# -*- coding: utf-8 -*-
# 用于测试平移不变性（para-fovea priming）
# 进一步修复错误的左右拼接
# In[]
import os
computer_name = os.environ['COMPUTERNAME']

if computer_name == 'JACK-GP68HX':
    # os.chdir(r'C:\Users\liang\Documents\Python Scripts\CORnet-master')
    imagenet_root = r'C:\Users\liang\Documents\ImageNet'
    map_location = 'cuda'
    
    local_log_root = r'C:\Users\liang\Documents\Python Scripts\CNN_Hebbian_Run\_Local_Log'

if computer_name == 'COLLES-161930':
    imagenet_root = r'C:\Users\jxl1870\Downloads\ImageNet'
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    map_location = 'cpu'
    
    local_log_root = r'C:\Users\jxl1870\Desktop\CNN_Hebbian_Run\_Local_Log'

# In[]
# import sys
# from pathlib import Path

# parent_dir = str(Path(__file__).resolve().parent.parent)
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)


# In[]
import numpy as np
import torch
import torch.nn.functional as F
import time
import pickle
from itertools import product
import random
# 固定随机种子（全局唯一）
SEED = 114514
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)    
#################
from Hebbian_VGG_Lib import Hebbian_VGG_Features,Hebbian_VGG_Classifier
from Utils.Utility import Cal_Accurate_
# import torch.nn as nn
from torchvision.datasets import ImageNet

import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from predify2021.model_factory.get_model import get_model
from predify_hebb_inject import inject_hebb_into_pcoder_rep

# In[] 20250523
from torchvision.models import VGG16_Weights
weights = VGG16_Weights.IMAGENET1K_V1

transform = weights.transforms()

mean, std = transform.mean, transform.std

dataset = ImageNet(
    imagenet_root, 'val', 
    transform = weights.transforms()
    )

def denormalize_imagenet(img_tensor, mean, std):
    """
    将经过 ImageNet Normalize 的图像还原为 0~1 范围可视化格式
    img_tensor: (3,H,W), tensor
    """
    mean = torch.tensor(mean, device=img_tensor.device).view(3,1,1)
    std  = torch.tensor(std, device=img_tensor.device).view(3,1,1)
    img = img_tensor * std + mean
    return torch.clamp(img, 0, 1)

# In[]
def apply_parafovea_partial_out(img: torch.Tensor, trial_idx: int,
                                canvas_scale: float = 1,
                                shift_px: int = 112,dataset=None) -> torch.Tensor:
    """
    将 (3,224,224) 的图像放入更大画布 (3,336,336)，
    奇数试次居中，偶数试次随机 ±shift_px 平移，
    并自动处理越界（保留留在画布内的部分）。
    """
    C, H, W = img.shape
    canvas_size = int(round(W * canvas_scale))  # 336
    canvas = torch.zeros((C, canvas_size, canvas_size), dtype=img.dtype, device=img.device)

    # 居中位置
    top_center = (canvas_size - H) // 2
    left_center = (canvas_size - W) // 2

    if not trial_idx % 2 == 0:
        x_offset = 0
    else:
        # 随机 ±112
        x_offset = shift_px if torch.rand(1).item() > 0.5 else -shift_px

    # 实际放置的左上角坐标（可能会 <0 或 >canvas-W）
    left_real = left_center + x_offset
    top_real = top_center

    # 计算源图像和目标画布的交集范围（避免越界）
    src_x0 = max(0, -left_real)                # 如果越界，跳过左边的部分
    src_x1 = min(W, canvas_size - left_real)   # 如果右边超出，截掉右侧
    dst_x0 = max(0, left_real)                 # 如果左边负数，画布从0开始
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    src_y0, src_y1 = 0, H
    dst_y0, dst_y1 = top_real, top_real + H

    # 贴到画布上（只贴不越界的部分）
    canvas[:, dst_y0:dst_y1, dst_x0:dst_x1] = img[:, src_y0:src_y1, src_x0:src_x1]

    return canvas

# In[]
def adjust_contrast_per_image(img: torch.Tensor, contrast_scale: float):
    """
    按每张图自身统计独立调整对比度。
    img: (3,H,W), 经过 ImageNet 预处理（不一定0均值）
    """
    mean = img.mean(dim=(1, 2), keepdim=True)
    std = img.std(dim=(1, 2), keepdim=True) + 1e-6
    return mean + (img - mean) * contrast_scale


def gaussian_blur(img: torch.Tensor, sigma: float):
    """
    对 (3,H,W) 图像做高斯模糊
    """
    if sigma <= 0:
        return img
    radius = int(3 * sigma)
    size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=img.device)
    kernel1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel1d /= kernel1d.sum()
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    kernel2d = kernel2d.expand(img.shape[0], 1, size, size)
    return F.conv2d(img.unsqueeze(0), kernel2d, padding=radius, groups=img.shape[0]).squeeze(0)


def make_half_field_stimulus_v5(
    dataset, idx, trial_idx,
    priming=True,
    canvas_scale: float = 2.0,
    contrast_scale: float = 0.5,
    blur_sigma: float = 1.0,
    downsample_after_blur: bool = True,
    dtype=torch.float32,
    device=None
):
    """
    Half-field priming 版本 5：
    - 偶数 trial：左右并排两张图（宽度扩展，高度保持224）
      -> 先调整对比度，再拼接，最后整体模糊
    - 奇数 trial：单张图（原始224x224）
    """
    C, H, W = 3, 224, 224
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 奇数 trial ===
    if trial_idx % 2 == 1:
        img, y = dataset[idx]
        img = img.to(device)
        return img, {"y": int(y), "mode": "single"}

    # === 偶数 trial ===
    # priming half（与下一 trial 相同）
    if priming:
        img_priming, y_priming = dataset[idx]
        last_rand_idx = idx
    else:
        # rand_idx = idx
        # while rand_idx == idx:
        #     tmp = random.randint(0, len(idx_log_list) - 1)
        #     rand_idx = idx_log_list[tmp]
        rand_idx = idx_log_list[(trial_idx+2)%len(idx_log_list)]
        img_priming, y_priming = dataset[rand_idx]
        # last_rand_idx = rand_idx

    # other half
    # rand_idx = idx
    # while rand_idx == idx or rand_idx==last_rand_idx:
    #     tmp = random.randint(0, len(idx_log_list) - 1)
    #     rand_idx = idx_log_list[tmp]
    rand_idx = idx_log_list[(trial_idx+4)%len(idx_log_list)]
    img_other, y_other = dataset[rand_idx]

    # 随机左右位置
    priming_on_left = random.random() > 0.5

    # 对比度调整（各自独立）
    img_priming = adjust_contrast_per_image(img_priming.to(device), contrast_scale)
    img_other = adjust_contrast_per_image(img_other.to(device), contrast_scale)

    # 拼接
    new_width = int(round(W * canvas_scale))
    half_w = new_width // 2
    canvas = torch.zeros((C, H, new_width), dtype=dtype, device=device)

    # 左右各自填充（按宽度裁剪或居中）
    def _paste_horiz(dst, img_src, x0, x1):
        dst_w = x1 - x0
        src_w = img_src.shape[2]
        if dst_w >= src_w:
            dst_start = x0 + (dst_w - src_w) // 2
            src_start = 0
            copy_w = src_w
        else:
            src_start = (src_w - dst_w) // 2
            dst_start = x0
            copy_w = dst_w
        dst[:, :, dst_start:dst_start+copy_w] = img_src[:, :, src_start:src_start+copy_w]

    if priming_on_left:
        _paste_horiz(canvas, img_priming, 0, half_w)
        _paste_horiz(canvas, img_other, new_width - half_w, new_width)
    else:
        _paste_horiz(canvas, img_other, 0, half_w)
        _paste_horiz(canvas, img_priming, new_width - half_w, new_width)

    # 模糊（整个画布）
    canvas = gaussian_blur(canvas, sigma=blur_sigma)
    
    # ✅ 降采样（节省计算）
    if downsample_after_blur:
        canvas = F.interpolate(
            canvas.unsqueeze(0),
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    return canvas, {
        "y_priming": int(y_priming),
        "y_other": int(y_other),
        "priming_on_left": priming_on_left,
        "canvas_size": canvas.shape[1:],
        "mode": "half-field"
    }

# In[]

if_store_log_locally = True
# map_location = 'cpu'

# In[]
device = torch.device(map_location)

lws = [1,1,1,1,1]
hps  = [
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[0]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[1]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[2]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[3]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[4]},
    ]

MAX_TIME_STEP = 5

model = get_model('pvgg',pretrained=True,deep_graph=False,hyperparams=hps).to(device)

# Hebb 模块：本脚本开启 Hebb 机制（ori_mode=False），注入到 PCoder rep 中
from Hebbian_VGG_Lib import Hebb_VGG_Channel_Boost, Hebb_Boost_C2
hebb_pcoder4 = Hebb_VGG_Channel_Boost(in_channels=512, ori_mode=False).to(device)
hebb_pcoder5 = Hebb_VGG_Channel_Boost(in_channels=512, ori_mode=False).to(device)
inject_hebb_into_pcoder_rep(model, 4, hebb_pcoder4)
inject_hebb_into_pcoder_rep(model, 5, hebb_pcoder5)

# Wrap classifier with Hebbian heads (active)
model.backbone.classifier = Hebbian_VGG_Classifier(model.backbone.classifier,ori_mode=False).to(device)

model.eval()

hebb_layer_list = [
    model.backbone.classifier.hebbian_2,
    model.backbone.classifier.hebbian_1,
    hebb_pcoder5,
    hebb_pcoder4
    ] # Top-down order

layer_para_list = [
    {'decay':0.5, 'coeff':0.05,'cut_perc':0.05},
    {'decay':0.5, 'coeff':0.05,'cut_perc':0.05},
    {'decay':0.5, 'coeff':0.2, 'inh_c':2},
    {'decay':0.5, 'coeff':0.2, 'inh_c':2},
]


# In[]
def _store_layer_metrics(layer, inp, out):
    global layer_metrics_arr,sample_idx,timestep_idx
    results = []
    for func_name in layer_metric_func:
        results.append(layer_metric_func[func_name](layer.x_tmp))
    
    layer_metrics_arr[sample_idx, timestep_idx, layer.layer_index] = torch.cat(results)

###################################################    
for i,layer in enumerate(hebb_layer_list):
    layer.layer_index = i
    if hasattr(layer,"set_para"):
        layer.set_para(**layer_para_list[i])
    layer.register_forward_hook(_store_layer_metrics)

layer_num = len(hebb_layer_list)

quantile_q = torch.tensor([0.25, 0.5, 0.75],device=device)


def NonZeroPercentage(x):
    return np.count_nonzero(x)/x.size

layer_metric_func = {
    'mean':lambda x:torch.mean(x).unsqueeze(0),
    'non-zero-per':lambda x:(torch.count_nonzero(x)/torch.numel(x)).unsqueeze(0),
    'max':lambda x:torch.max(x).unsqueeze(0),
    'quantiles':lambda x:torch.quantile(x, q=quantile_q),
    }

metric_num = 1 + 1 + 1 + len(quantile_q)
# In[]
time0 = time.time()

task_name = 'N_Back-'
# task_name = 'Class_N_Back-'
# task_name = 'Cluster-'
for gap in [1]:    
    with open('Log/DatasetSample/' + task_name + str(gap) +'.pckl','rb') as f:
        sample_dict = pickle.load(f)
    idx_log_list = sample_dict['out_list']
    
    sample_num = len(idx_log_list) 
    
    layer_metrics_arr = torch.zeros((sample_num,MAX_TIME_STEP,layer_num,metric_num),device=device)

    
    for priming in [True,False]:
        y_list = []
        Out_list = [[] for _ in range(MAX_TIME_STEP)]
        
        priming_sufix = '_Prime' if priming else '_nonPrime'
    
        with torch.no_grad():
            
            for sample_idx,idx in enumerate(idx_log_list):
                
                # if non_priming and (sample_idx % 2 == 0):
                #     # === 非 priming 条件：偶数试次换图 ===
                #     rand_idx = idx
                #     while rand_idx == idx:
                #         rand_idx = random.randint(0, len(dataset) - 1)
                #     img, y = dataset[rand_idx]
                # else:
                #     # === priming 条件：用当前样本 ===
                #     img, y = dataset[idx]
                # img = apply_parafovea_partial_out(img, sample_idx)
                
                img, meta = make_half_field_stimulus_v5(dataset, idx, sample_idx, priming=priming,canvas_scale=2)
                # print('img统计： ',img.mean(),img.std())
                # if 'canvas_size' in meta:
                #     print(meta['canvas_size'])
                if meta["mode"] == "single":
                    # 奇数 trial
                    y_true = meta["y"]
                else:
                    # 偶数 trial
                    if priming:
                        y_true = meta["y_priming"]
                    else:
                        # 非 priming 对照组不评分类别
                        y_true = -1

                # 偶数 trial（priming 拼接前）清零 Hebb boost 记忆
                if sample_idx % 2 == 0:
                    assert(meta["mode"]!='single')
                    for layer in hebb_layer_list:
                        if hasattr(layer,"zero_boost_weight"):
                            layer.zero_boost_weight()

                net_input = img.unsqueeze(0).to(device)
                model.reset()
                for layer in hebb_layer_list:
                    if hasattr(layer, "update_enabled"):
                        layer.update_enabled = False
                for timestep_idx in range(MAX_TIME_STEP):
                    if timestep_idx == 0:
                        out = model(net_input)
                    else:
                        out = model(None)
                    Out_list[timestep_idx].append(out.detach().cpu())
                # 网络动力学收敛后（最后一个 timestep 之后）再更新 Hebb 权重
                for layer in hebb_layer_list:
                    if hasattr(layer, "commit_update"):
                        layer.commit_update()
                y_list.append(int(y_true))
                
                
                # img_vis = denormalize_imagenet(img, mean, std)
                # img_np = np.transpose(img_vis.cpu().numpy(), (1,2,0))
                # plt.imshow(img_np)
                # plt.axis('off')
                # plt.show()
                
                if sample_idx >= 40 - 1:
                    # 4/0
                    break
                   
        outputs_per_timestep = [torch.cat(step_outs,dim=0) for step_outs in Out_list]
        Out_arr = torch.stack(outputs_per_timestep,dim=1).numpy()
        
        acc_top_1 = Cal_Accurate_(Out_arr[:,-1,:],y_list,1)
        acc_top_5 = Cal_Accurate_(Out_arr[:,-1,:],y_list,5)
        acc_result = (acc_top_1.count(True)/len(acc_top_1), acc_top_5.count(True)/len(acc_top_5))
        
        print('Acc: ',acc_result)
        
        SaveDict = {
            'task_name':task_name,
            'gap':gap,
            'max_time_step':MAX_TIME_STEP,
            
            'idx_log_list':sample_dict['out_list'],
            'full_clss_list':sample_dict['full_clss_list'],
            'y_list':y_list,
            'Out_list':Out_arr,
            'layer_metrics_arr':layer_metrics_arr.cpu().numpy(),
            'acc_top_1':acc_top_1,
            'acc_top_5':acc_top_5,
            'acc_result':acc_result,
            }
        
        ####
        # break
        
        if not if_store_log_locally:
            # save_output_path = os.path.join('Log','HebbianModelResult_')
            pass
        else:
            save_output_path = os.path.join(local_log_root,'Log','VGG_Result_')
        
        save_output_filename = save_output_path + task_name + str(gap) +'_new' +priming_sufix+ '.pckl'
        
        print('Dump result: ',save_output_filename)
        
        with open(save_output_filename,'wb') as f:
            pickle.dump(SaveDict,f)
        
        print('######################################')
        print()

time1 = time.time() - time0
print('Time Cost (s): ',time1)
print('\a')
print('\a')
