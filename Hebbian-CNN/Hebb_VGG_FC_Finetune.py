# -*- coding: utf-8 -*-
# Finetune VGG FC layers on Hebbian-boosted inputs.
# Keeps conv backbone frozen; only FC layers and final classifier are trained.

import os
import sys
import time
import random
import zipfile
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision import transforms

computer_name = os.environ.get('COMPUTERNAME', '')
if computer_name == 'JACK-GP68HX':
    imagenet_root = r'C:\Users\liang\Documents\ImageNet'
    map_location = 'cuda'
    local_log_root = r'C:\Users\liang\Documents\Python Scripts\CNN_Hebbian_Run\_Local_Log'
elif computer_name == 'COLLES-161930':
    imagenet_root = r'C:\Users\jxl1870\Downloads\ImageNet'
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    map_location = 'cpu'
    local_log_root = r'C:\Users\jxl1870\Desktop\CNN_Hebbian_Run\_Local_Log'
else:
    imagenet_root = r'C:\Users\liang\Documents\ImageNet'
    map_location = 'cuda'
    local_log_root = r'.'

SEED = 114514
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

sys.path.append("..")
from predify2021.model_factory.get_model import get_model
from predify_hebb_inject import inject_hebb_into_pcoder_rep
from Hebbian_VGG_Lib import Hebbian_VGG_Classifier, Hebb_VGG_Channel_Boost, get_default_hebb_layer_params

from torchvision.models import VGG16_Weights
weights = VGG16_Weights.IMAGENET1K_V1
transform = weights.transforms()
mean, std = transform.mean, transform.std

# ------------------------------
# Finetune config
# ------------------------------
FC_INIT_WEIGHT_SOURCE = 'torchvision'  # 'torchvision' or 'finetuned'
FC_INIT_CKPT_PATH = r''                # set when FC_INIT_WEIGHT_SOURCE == 'finetuned'

FC_HEBB_COMPUTE_ENABLED = False
HEBB_UPDATE_ENABLED = True
FC_FORWARD_MODE = 'backbone_direct'  # 'predify' or 'backbone_direct'

FC_FINETUNE_EPOCHS = 1
FC_FINETUNE_BATCH_SIZE = 64
FC_FINETUNE_LR = 1e-3
FC_FINETUNE_WEIGHT_DECAY = 1e-4
FC_FINETUNE_NUM_WORKERS = 4
FC_FINETUNE_TIME_STEPS = 1
FC_FINETUNE_EVAL_EVERY = 1
FC_FINETUNE_SAVE_BEST_ONLY = True
FC_FINETUNE_RESET_HEBB_PER_BATCH = True
FC_FINETUNE_DEFER_HEBB_UPDATE = True
FC_FINETUNE_MAX_TRAIN_SAMPLES = None
FC_FINETUNE_MAX_VAL_SAMPLES = None
FC_FINETUNE_TAG = None

if 'local_log_root' in globals():
    FC_FINETUNE_OUT_DIR = os.path.join(local_log_root, 'Weights')
else:
    FC_FINETUNE_OUT_DIR = os.path.join('Weights')

# ------------------------------
# Helpers
# ------------------------------
def _get_classifier_pipe(classifier):
    return classifier.pipe if hasattr(classifier, "pipe") else classifier


def _load_fc_checkpoint(classifier, ckpt_path, map_location):
    if not ckpt_path:
        raise ValueError("FC_INIT_CKPT_PATH is empty.")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = ckpt.get("classifier_state", ckpt)
    _get_classifier_pipe(classifier).load_state_dict(state_dict)
    return ckpt


def _save_code_archive(archive_path, source_paths):
    archive_dir = os.path.dirname(archive_path)
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in source_paths:
            if src and os.path.isfile(src):
                zf.write(src, arcname=os.path.basename(src))
    return archive_path


def _build_imagenet_loaders(root, mean, std, batch_size, num_workers, pin_memory,
                            max_train_samples=None, max_val_samples=None):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = ImageNet(root, "train", transform=train_transform)
    val_dataset = ImageNet(root, "val", transform=val_transform)

    if max_train_samples is not None:
        max_train_samples = min(max_train_samples, len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(max_train_samples)))
    if max_val_samples is not None:
        max_val_samples = min(max_val_samples, len(val_dataset))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(max_val_samples)))

    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def _set_fc_trainable(model):
    for param in model.parameters():
        param.requires_grad = False
    classifier = _get_classifier_pipe(model.backbone.classifier)
    for param in classifier.parameters():
        param.requires_grad = True
    return [p for p in classifier.parameters() if p.requires_grad]


def _forward_with_timesteps(model, inputs, time_steps, hebb_layers=None, defer_hebb_update=False, forward_mode='predify'):
    steps = max(1, int(time_steps))
    model.reset()
    if forward_mode == 'predify':
        prev_update = []
        if hebb_layers and defer_hebb_update:
            for layer in hebb_layers:
                if hasattr(layer, "update_enabled"):
                    prev_update.append((layer, layer.update_enabled))
                    layer.update_enabled = False

        out = None
        for step in range(steps):
            out = model(inputs if step == 0 else None)

        if hebb_layers and defer_hebb_update:
            for layer, prev in prev_update:
                if prev and hasattr(layer, "commit_update"):
                    layer.commit_update()
                layer.update_enabled = prev
        return out

    if forward_mode != 'backbone_direct':
        raise ValueError(f"Unsupported FC_FORWARD_MODE: {forward_mode}")

    model.input_mem = inputs
    if getattr(model, 'random_init', False):
        with torch.no_grad():
            _ = model.backbone(model.input_mem)

    out = None
    for step in range(steps):
        with torch.no_grad():
            feats = model.backbone.features(model.input_mem)
            feats = model.backbone.avgpool(feats)
            feats = torch.flatten(feats, 1)
        out = model.backbone.classifier(feats)

    if hebb_layers and defer_hebb_update:
        for layer in hebb_layers:
            if hasattr(layer, "commit_update") and getattr(layer, "update_enabled", True):
                layer.commit_update()
    return out


def _eval_top1(model, loader, device, time_steps, hebb_layers=None,
               reset_hebb_per_batch=False, defer_hebb_update=False, forward_mode='predify'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if reset_hebb_per_batch and hebb_layers:
                for layer in hebb_layers:
                    if hasattr(layer, "zero_boost_weight"):
                        layer.zero_boost_weight()
            outputs = _forward_with_timesteps(
                model,
                images,
                time_steps,
                hebb_layers=hebb_layers,
                defer_hebb_update=defer_hebb_update,
                forward_mode=forward_mode,
            )
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(1, total)


def _save_fc_checkpoint(model, output_dir, tag=None, extra_state=None):
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"{tag}_" if tag else ""
    ckpt_path = os.path.join(output_dir, f"fc_finetune_{tag_part}{stamp}.pth")

    classifier = _get_classifier_pipe(model.backbone.classifier)
    ckpt = {"classifier_state": classifier.state_dict()}
    if extra_state:
        ckpt.update(extra_state)

    code_archive = os.path.splitext(ckpt_path)[0] + "_code.zip"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _save_code_archive(code_archive, [
        os.path.abspath(__file__),
        os.path.join(script_dir, "Hebbian_VGG_Lib.py"),
        os.path.join(script_dir, "Hebb_VGG_Pred_Run.py"),
    ])
    ckpt["code_archive"] = os.path.basename(code_archive)

    torch.save(ckpt, ckpt_path)
    return ckpt_path


def finetune_fc_layers(model, device, hebb_layers, out_dir, tag=None):
    pin_memory = device.type == "cuda"
    train_loader, val_loader = _build_imagenet_loaders(
        imagenet_root,
        mean,
        std,
        FC_FINETUNE_BATCH_SIZE,
        FC_FINETUNE_NUM_WORKERS,
        pin_memory,
        max_train_samples=FC_FINETUNE_MAX_TRAIN_SAMPLES,
        max_val_samples=FC_FINETUNE_MAX_VAL_SAMPLES,
    )

    model.build_graph = (FC_FORWARD_MODE == 'predify')
    model.train()
    trainable_params = _set_fc_trainable(model)

    optimizer = torch.optim.SGD(
        trainable_params,
        lr=FC_FINETUNE_LR,
        momentum=0.9,
        weight_decay=FC_FINETUNE_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = None
    for epoch in range(FC_FINETUNE_EPOCHS):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            if FC_FINETUNE_RESET_HEBB_PER_BATCH and hebb_layers:
                for layer in hebb_layers:
                    if hasattr(layer, "zero_boost_weight"):
                        layer.zero_boost_weight()

            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_with_timesteps(
                model,
                images,
                FC_FINETUNE_TIME_STEPS,
                hebb_layers=hebb_layers,
                defer_hebb_update=FC_FINETUNE_DEFER_HEBB_UPDATE,
                forward_mode=FC_FORWARD_MODE,
            )
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            if (batch_idx + 1) % 50 == 0:
                avg_loss = running_loss / max(1, total_samples)
                print(f"[FC FT] Epoch {epoch+1}/{FC_FINETUNE_EPOCHS} "
                      f"Iter {batch_idx+1}/{len(train_loader)} "
                      f"Loss {avg_loss:.4f}")

        avg_loss = running_loss / max(1, total_samples)
        print(f"[FC FT] Epoch {epoch+1}/{FC_FINETUNE_EPOCHS} Loss {avg_loss:.4f}")

        if val_loader and FC_FINETUNE_EVAL_EVERY > 0 and ((epoch + 1) % FC_FINETUNE_EVAL_EVERY == 0):
            val_acc = _eval_top1(
                model,
                val_loader,
                device,
                FC_FINETUNE_TIME_STEPS,
                hebb_layers=hebb_layers,
                reset_hebb_per_batch=FC_FINETUNE_RESET_HEBB_PER_BATCH,
                defer_hebb_update=FC_FINETUNE_DEFER_HEBB_UPDATE,
                forward_mode=FC_FORWARD_MODE,
            )
            print(f"[FC FT] Val top1 {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = _save_fc_checkpoint(
                    model,
                    out_dir,
                    tag=tag,
                    extra_state={"val_top1": val_acc, "epoch": epoch + 1},
                )
                print(f"[FC FT] Saved best checkpoint: {best_path}")
        elif not FC_FINETUNE_SAVE_BEST_ONLY:
            best_path = _save_fc_checkpoint(
                model,
                out_dir,
                tag=tag,
                extra_state={"epoch": epoch + 1},
            )
            print(f"[FC FT] Saved checkpoint: {best_path}")

    if best_path is None:
        best_path = _save_fc_checkpoint(model, out_dir, tag=tag, extra_state={"epoch": FC_FINETUNE_EPOCHS})
        print(f"[FC FT] Saved final checkpoint: {best_path}")

    model.build_graph = False
    return best_path


if __name__ == '__main__':
    device = torch.device(map_location)

    lws = [1, 1, 1, 1, 1]
    hps = [
        {"ffm": 0.8, "fbm": 0.1, "erm": 0.01 * lws[0]},
        {"ffm": 0.8, "fbm": 0.1, "erm": 0.01 * lws[1]},
        {"ffm": 0.8, "fbm": 0.1, "erm": 0.01 * lws[2]},
        {"ffm": 0.8, "fbm": 0.1, "erm": 0.01 * lws[3]},
        {"ffm": 0.8, "fbm": 0.1, "erm": 0.01 * lws[4]},
    ]

    model = get_model('pvgg', pretrained=True, deep_graph=False, hyperparams=hps).to(device)

    if FC_INIT_WEIGHT_SOURCE == 'finetuned':
        _load_fc_checkpoint(model.backbone.classifier, FC_INIT_CKPT_PATH, map_location=device)
    elif FC_INIT_WEIGHT_SOURCE != 'torchvision':
        raise ValueError(f"Unsupported FC_INIT_WEIGHT_SOURCE: {FC_INIT_WEIGHT_SOURCE}")

    hebb_pcoder4 = Hebb_VGG_Channel_Boost(in_channels=512, ori_mode=False).to(device)
    hebb_pcoder5 = Hebb_VGG_Channel_Boost(in_channels=512, ori_mode=False).to(device)
    inject_hebb_into_pcoder_rep(model, 4, hebb_pcoder4)
    inject_hebb_into_pcoder_rep(model, 5, hebb_pcoder5)

    model.backbone.classifier = Hebbian_VGG_Classifier(model.backbone.classifier, ori_mode=False).to(device)
    model.backbone.classifier.hebbian_1.compute_enabled = FC_HEBB_COMPUTE_ENABLED
    model.backbone.classifier.hebbian_2.compute_enabled = FC_HEBB_COMPUTE_ENABLED

    hebb_layer_list = [
        model.backbone.classifier.hebbian_2,
        model.backbone.classifier.hebbian_1,
        hebb_pcoder5,
        hebb_pcoder4,
    ]

    layer_para_list = get_default_hebb_layer_params()

    for i, layer in enumerate(hebb_layer_list):
        if hasattr(layer, "set_para"):
            layer.set_para(**layer_para_list[i])
        if hasattr(layer, "update_enabled"):
            layer.update_enabled = HEBB_UPDATE_ENABLED

    print("[FC FT] Start.")
    start = time.time()
    ckpt_path = finetune_fc_layers(
        model,
        device,
        hebb_layer_list,
        FC_FINETUNE_OUT_DIR,
        tag=FC_FINETUNE_TAG,
    )
    print(f"[FC FT] Done. Saved: {ckpt_path}")
    print(f"[FC FT] Time cost (s): {time.time() - start:.2f}")
