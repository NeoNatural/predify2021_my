# -*- coding: utf-8 -*-
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
import time
import pickle
from itertools import product
import sys

from Hebbian_VGG_Lib import Hebbian_VGG_Features,Hebbian_VGG_Classifier, Hebb_VGG_Channel_Boost, Hebb_Boost_C2
from Utils.Utility import Cal_Accurate_
from torchvision.datasets import ImageNet
from torchvision.models import VGG16_Weights

sys.path.append("..")
from predify2021.model_factory.get_model import get_model

# In[] 20250523
weights = VGG16_Weights.IMAGENET1K_V1

dataset = ImageNet(
    imagenet_root, 'val', 
    transform = weights.transforms()
    )

# In[]

if_store_log_locally = False
# map_location = 'cpu'
device = torch.device(map_location)

lws = [1,1,1,1,1]
hps  = [
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[0]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[1]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[2]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[3]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[4]},
    ]
MAX_TIME_STEP = 10

model = get_model('pvgg',pretrained=True,deep_graph=False,hyperparams=hps).to(device)

# Hebb 层以 ori_mode=True（仅记录，不更新权重）
hebb_feat_1 = Hebb_VGG_Channel_Boost(ori_mode=True).to(device)
hebb_feat_2 = Hebb_VGG_Channel_Boost(ori_mode=True).to(device)

def _attach_hebb_feat(layer, hebb_layer):
    def hook_fn(m, inp, out):
        return hebb_layer(out)
    layer.register_forward_hook(hook_fn)

# VGG16 (no BN) pool4 at idx 23, pool5 at idx 30
_attach_hebb_feat(model.backbone.features[23], hebb_feat_1)
_attach_hebb_feat(model.backbone.features[30], hebb_feat_2)

# classifier 包装为 Hebbian 版本（ori_mode=True）
model.backbone.classifier = Hebbian_VGG_Classifier(model.backbone.classifier,ori_mode=True).to(device)

model.eval()

hebb_layer_list = [
    model.backbone.classifier.hebbian_2,
    model.backbone.classifier.hebbian_1,
    hebb_feat_2,
    hebb_feat_1
    ] # Top-down order

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
    layer.register_forward_hook(_store_layer_metrics)

layer_num = len(hebb_layer_list)

quantile_q = torch.tensor([0.25, 0.5, 0.75],device=device)

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
    
    y_list = []
    Out_list = [[] for _ in range(MAX_TIME_STEP)]

    with torch.no_grad():
        
        for sample_idx,idx in enumerate(idx_log_list):
            
            img, y = dataset[idx]

            # 每 2 个 trial 清零一次 Hebb boost 记忆
            if sample_idx % 2 == 0:
                for layer in hebb_layer_list:
                    if hasattr(layer,"zero_boost_weight"):
                        layer.zero_boost_weight()

            net_input = img.unsqueeze(0).to(device)
            model.reset()
            for timestep_idx in range(MAX_TIME_STEP):
                if timestep_idx == 0:
                    out = model(net_input)
                else:
                    out = model(None)
                Out_list[timestep_idx].append(out.detach().cpu())
            y_list.append(int(y))
               
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
        save_output_path = os.path.join('Log','OriModelResult_')
        
    else:
        # save_output_path = os.path.join(local_log_root,'Log','VGG_Result_')
        pass
    
    save_output_filename = save_output_path + task_name + str(gap) + '.pckl'
    
    print('Dump result: ',save_output_filename)
    
    with open(save_output_filename,'wb') as f:
        pickle.dump(SaveDict,f)
    
    print('######################################')
    print()

time1 = time.time() - time0
print('Time Cost (s): ',time1)
