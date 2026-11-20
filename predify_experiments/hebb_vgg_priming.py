from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import pickle
from pathlib import Path
import random
import sys
import time
from typing import Any

from .config import ExperimentConfig
from .sequence import reset_hebbian_state, run_recurrent_sequence


RESULT_SCHEMA_VERSION = 3


def _add_legacy_module_path(project_root: Path) -> None:
    legacy_dir = project_root / "Hebbian-CNN"
    legacy_str = str(legacy_dir)
    if legacy_str not in sys.path:
        sys.path.insert(0, legacy_str)


def _resolve_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment")
    return torch.device(requested)


def _seed_everything(seed: int, np: Any, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _topk_correct(outputs: Any, labels: list[int], k: int) -> list[bool]:
    top_indices = outputs.argsort(axis=-1)[..., -k:]
    return [int(label) in top_indices[index].tolist() for index, label in enumerate(labels)]


def _condition_accuracy(correct: list[bool]) -> float:
    repeated_trials = correct[1::2]
    if not repeated_trials:
        return float("nan")
    return repeated_trials.count(True) / len(repeated_trials)


def _degradation_suffix(mode: str, level: float) -> str:
    if mode == "none":
        return ""
    level_text = format(level, ".12g").replace("-", "m").replace(".", "p")
    label = "sigma" if mode == "predify_noise" else "intensity"
    return f"_{mode}_{label}{level_text}"


def _numpy_stats(stats_obj: Any) -> dict[str, dict[str, Any]]:
    if not stats_obj.count:
        return {}
    return {
        layer: {name: value.cpu().numpy() for name, value in values.items()}
        for layer, values in stats_obj.finalize().items()
    }


class _RunRecorder:
    def __init__(
        self,
        *,
        torch: Any,
        device: Any,
        sample_count: int,
        max_time_steps: int,
        hebbian_layers: list[Any],
        pcoders: list[Any],
        representation_layers: dict[str, Any],
    ) -> None:
        self.torch = torch
        self.sample_index = 0
        self.timestep = 0
        self.representation_cache: dict[str, Any] = {}
        self.quantiles = torch.tensor([0.25, 0.5, 0.75], device=device)
        self.layer_metrics = torch.zeros(
            (sample_count, max_time_steps, len(hebbian_layers), 6), device=device
        )
        self.pcoder_errors = torch.zeros(
            (sample_count, max_time_steps, len(pcoders)), device=device
        )
        self.handles: list[Any] = []

        for layer_index, layer in enumerate(hebbian_layers):
            self.handles.append(layer.register_forward_hook(self._layer_hook(layer_index)))
        for pcoder_index, pcoder in enumerate(pcoders):
            self.handles.append(pcoder.register_forward_hook(self._pcoder_hook(pcoder_index)))
        for name, layer in representation_layers.items():
            self.handles.append(layer.register_forward_hook(self._representation_hook(name)))

    def _layer_hook(self, layer_index: int):
        def hook(layer: Any, _inputs: Any, _output: Any) -> None:
            x = layer.x_tmp.detach()
            values = self.torch.cat(
                (
                    self.torch.mean(x).view(1),
                    (self.torch.count_nonzero(x) / self.torch.numel(x)).view(1),
                    self.torch.max(x).view(1),
                    self.torch.quantile(x, q=self.quantiles),
                )
            )
            self.layer_metrics[self.sample_index, self.timestep, layer_index] = values

        return hook

    def _pcoder_hook(self, pcoder_index: int):
        def hook(pcoder: Any, _inputs: Any, _output: Any) -> None:
            self.pcoder_errors[self.sample_index, self.timestep, pcoder_index] = (
                pcoder.prediction_error.detach()
            )

        return hook

    def _representation_hook(self, name: str):
        def hook(_layer: Any, _inputs: Any, output: Any) -> None:
            self.representation_cache[name] = output.detach()

        return hook

    def set_position(self, sample_index: int, timestep: int) -> None:
        self.sample_index = sample_index
        self.timestep = timestep

    def clear_representations(self) -> None:
        self.representation_cache.clear()

    def update_representation_stats(
        self, stats_obj: Any, label: Any, representation_order: list[str]
    ) -> None:
        features: dict[str, Any] = {}
        for name in representation_order:
            if name not in self.representation_cache:
                continue
            value = self.representation_cache[name]
            if value.dim() > 2:
                value = value.flatten(start_dim=1)
            elif value.dim() == 1:
                value = value.unsqueeze(0)
            features[name] = value
        if features:
            stats_obj.update(feats=features, labels=label)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _build_model(config: ExperimentConfig, device: Any, torch: Any) -> tuple[Any, list[Any], list[int]]:
    from predify2021.model_factory.get_model import get_model
    from Hebbian_VGG_Lib import (
        Hebb_VGG_Channel_Boost,
        Hebbian_VGG_Classifier,
        get_default_hebb_layer_params,
    )
    from predify_hebb_inject import inject_hebb_into_pcoder_rep

    model = get_model(
        config.model.name,
        pretrained=config.model.pretrained_pcoders,
        deep_graph=False,
        hyperparams=list(config.model.hyperparameters) or None,
        weight_dir=config.paths.pcoder_weight_dir,
        map_location=device,
        random_init=config.model.random_init,
    ).to(device)

    if config.model.fc_weight_source == "finetuned":
        checkpoint = torch.load(config.paths.fc_checkpoint, map_location=device)
        state_dict = checkpoint.get("classifier_state", checkpoint)
        model.backbone.classifier.load_state_dict(state_dict)

    original_mode = config.model.original_mode
    hebb_pcoder4 = Hebb_VGG_Channel_Boost(in_channels=512, ori_mode=original_mode).to(device)
    hebb_pcoder5 = Hebb_VGG_Channel_Boost(in_channels=512, ori_mode=original_mode).to(device)
    inject_hebb_into_pcoder_rep(model, 4, hebb_pcoder4)
    inject_hebb_into_pcoder_rep(model, 5, hebb_pcoder5)

    model.backbone.classifier = Hebbian_VGG_Classifier(
        model.backbone.classifier, ori_mode=original_mode
    ).to(device)
    model.backbone.classifier.hebbian_1.compute_enabled = (
        config.model.fc_hebb_compute_enabled
    )
    model.backbone.classifier.hebbian_2.compute_enabled = (
        config.model.fc_hebb_compute_enabled
    )
    model.eval()

    hebbian_layers = [
        model.backbone.classifier.hebbian_2,
        model.backbone.classifier.hebbian_1,
        hebb_pcoder5,
        hebb_pcoder4,
    ]
    for layer, parameters in zip(
        hebbian_layers, get_default_hebb_layer_params(), strict=True
    ):
        layer.set_para(**parameters)
    return model, hebbian_layers, [4, 5]


def run_experiment(config: ExperimentConfig) -> list[Path]:
    config.validate_for_run()
    _add_legacy_module_path(config.paths.project_root)

    import numpy as np
    import torch
    from torchvision.datasets import ImageNet
    from torchvision.models import VGG16_Weights

    from Hebb_Priming_Utils import make_priming_stimulus
    from OnlineSharpeningStats_Demo import OnlineSharpeningStats

    device = _resolve_device(config.runtime.device, torch)
    _seed_everything(config.runtime.seed, np, torch)
    print(f"[experiment] device={device} config={config.source_file}")

    weights = VGG16_Weights.IMAGENET1K_V1
    dataset = ImageNet(
        str(config.paths.imagenet_root), "val", transform=weights.transforms()
    )
    model, hebbian_layers, pcoder_indices = _build_model(config, device, torch)
    pcoders = [getattr(model, f"pcoder{index}") for index in pcoder_indices]

    with config.paths.sample_file.open("rb") as handle:
        sample_dict = pickle.load(handle)
    all_indices = list(sample_dict["out_list"])
    sample_count = len(all_indices)
    if config.runtime.max_samples is not None:
        sample_count = min(sample_count, config.runtime.max_samples)
    indices = all_indices[:sample_count]
    full_class_list = list(sample_dict["full_clss_list"][:sample_count])

    class_source = sample_dict["full_clss_list"][
        : config.experiment.representation_class_items
    ]
    representation_classes = sorted(set(int(value) for value in class_source))
    if not representation_classes:
        raise ValueError("No representation classes found in sample file")
    compact_labels = {label: index for index, label in enumerate(representation_classes)}

    representation_order = ["hebbian_2", "hebbian_1", "hebb_pcoder5", "hebb_pcoder4"]
    representation_layers = {
        "hebbian_2": model.backbone.classifier.hebbian_2,
        "hebbian_1": model.backbone.classifier.hebbian_1,
        "hebb_pcoder5": hebbian_layers[2],
        "hebb_pcoder4": hebbian_layers[3],
    }
    conditions: list[bool] = []
    if config.experiment.run_priming:
        conditions.append(True)
    if config.experiment.run_nonpriming:
        conditions.append(False)

    output_paths: list[Path] = []
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    runs = [
        (degradation_mode, degradation_level, priming)
        for degradation_mode, degradation_level in config.stimulus.degradation_conditions()
        for priming in conditions
    ]
    for degradation_mode, degradation_level, priming in runs:
        # Keep image placement and stochastic degradation paired across levels
        # and experimental conditions.
        _seed_everything(config.runtime.seed, np, torch)
        recorder = _RunRecorder(
            torch=torch,
            device=device,
            sample_count=sample_count,
            max_time_steps=config.runtime.max_time_steps,
            hebbian_layers=hebbian_layers,
            pcoders=pcoders,
            representation_layers=representation_layers,
        )
        try:
            labels: list[int] = []
            outputs_by_time: list[list[Any]] = [
                [] for _ in range(config.runtime.max_time_steps)
            ]
            stats_by_time = [
                OnlineSharpeningStats(
                    layer_names=representation_order,
                    num_classes=len(representation_classes),
                    device=device,
                )
                for _ in range(config.runtime.max_time_steps)
            ]

            with torch.no_grad():
                for sample_index, dataset_index in enumerate(indices):
                    recorder.clear_representations()
                    degradation_seed = config.stimulus.degradation_seed
                    if degradation_mode == "fractal_occlusion":
                        degradation_seed += sample_index
                    image, metadata = make_priming_stimulus(
                        dataset,
                        dataset_index,
                        sample_index,
                        priming=priming,
                        # Preserve the historical control-stimulus schedule even
                        # when max_samples truncates the processed trials.
                        idx_log_list=all_indices,
                        canvas_scale=config.stimulus.canvas_scale,
                        contrast_scale=config.stimulus.contrast_scale,
                        prime_layout=config.stimulus.prime_layout,
                        degradation_mode=degradation_mode,
                        degradation_level=degradation_level,
                        degradation_seed=degradation_seed,
                        fractal_octaves=config.stimulus.fractal_octaves,
                        blur_sigma=config.stimulus.blur_sigma,
                        downsample_after_blur=config.stimulus.downsample_after_blur,
                        device=device,
                    )
                    if metadata["mode"] == "single":
                        true_label = int(metadata["y"])
                    elif priming:
                        true_label = int(metadata["y_priming"])
                    else:
                        true_label = -1

                    compact_label = compact_labels.get(true_label)
                    label_tensor = (
                        torch.tensor(compact_label, device=device)
                        if compact_label is not None
                        else None
                    )
                    should_record_representation = (
                        sample_index % 2 == 1 and label_tensor is not None
                    )

                    if sample_index % 2 == 0:
                        if metadata["mode"] == "single":
                            raise RuntimeError("Expected half-field stimulus on an even trial")
                        reset_hebbian_state(hebbian_layers)

                    step_count = (
                        config.runtime.reduced_time_steps
                        if sample_index % 2 == 0
                        else config.runtime.max_time_steps
                    )
                    next_index = sample_index + 1
                    next_trial_will_reset = (
                        next_index < sample_count and next_index % 2 == 0
                    )

                    def before_step(timestep: int) -> None:
                        recorder.set_position(sample_index, timestep)

                    def after_step(timestep: int, output: Any) -> None:
                        outputs_by_time[timestep].append(output.detach().cpu())
                        if should_record_representation:
                            recorder.update_representation_stats(
                                stats_by_time[timestep], label_tensor, representation_order
                            )

                    step_outputs = run_recurrent_sequence(
                        model,
                        image.unsqueeze(0).to(device),
                        step_count,
                        hebbian_layers=hebbian_layers,
                        commit_update=not next_trial_will_reset,
                        before_step=before_step,
                        after_step=after_step,
                    )
                    final_output = step_outputs[-1].detach().cpu()
                    for timestep in range(step_count, config.runtime.max_time_steps):
                        outputs_by_time[timestep].append(final_output)
                    labels.append(true_label)

            outputs = np.stack(
                [torch.cat(values, dim=0).numpy() for values in outputs_by_time], axis=1
            )
            top1 = _topk_correct(outputs[:, -1, :], labels, 1)
            top5 = _topk_correct(outputs[:, -1, :], labels, 5)
            accuracy = (_condition_accuracy(top1), _condition_accuracy(top5))
            print(
                f"[experiment] degradation={degradation_mode} "
                f"level={degradation_level:g} priming={priming} accuracy={accuracy}"
            )

            stats_timeline = [_numpy_stats(stats) for stats in stats_by_time]
            result = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "run_metadata": {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "config_file": str(config.source_file),
                    "device": str(device),
                    "seed": config.runtime.seed,
                    "condition": "priming" if priming else "nonpriming",
                    "prime_layout": config.stimulus.prime_layout,
                    "degradation_mode": degradation_mode,
                    "degradation_level": degradation_level,
                    "degradation_seed": config.stimulus.degradation_seed,
                    "hebbian_condition": (
                        "nohebb" if config.model.original_mode else "hebb"
                    ),
                    "config": asdict(config),
                },
                "task_name": config.experiment.task_name,
                "gap": config.experiment.gap,
                "max_time_step": config.runtime.max_time_steps,
                "idx_log_list": indices,
                "full_clss_list": full_class_list,
                "y_list": labels,
                "Out_list": outputs,
                "layer_metrics_arr": recorder.layer_metrics.cpu().numpy(),
                "pcoder_error_arr": recorder.pcoder_errors.cpu().numpy()[..., None],
                "pcoder_indices_to_record": pcoder_indices,
                "repr_layer_names": representation_order,
                "repr_class_subset": representation_classes,
                "repr_stats_timeline": stats_timeline,
                "repr_stats": stats_timeline[-1],
                "acc_top_1": top1,
                "acc_top_5": top5,
                "acc_result": accuracy,
            }
            suffix = "_Prime" if priming else "_nonPrime"
            degradation_suffix = _degradation_suffix(
                degradation_mode, degradation_level
            )
            layout_suffix = (
                "" if config.stimulus.prime_layout == "parafoveal_half_field"
                else f"_{config.stimulus.prime_layout}"
            )
            output_file = config.paths.output_dir / (
                f"VGG_Result_{config.experiment.task_name}{config.experiment.gap}"
                f"_new{layout_suffix}{degradation_suffix}{suffix}.pckl"
            )
            with output_file.open("wb") as handle:
                pickle.dump(result, handle)
            output_paths.append(output_file)
            print(f"[experiment] wrote {output_file}")
        finally:
            recorder.close()

    print(f"[experiment] completed in {time.time() - started:.1f}s")
    return output_paths
