# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:51:55 2024

@author: jxl1870
"""

# # Referenced from Cornet-master run.py
# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     with torch.no_grad():
#         _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = [correct[:k].sum().item() for k in topk]
#         return res
import numpy as np
import os
import pickle
from scipy.special import softmax
from torch.nn import CrossEntropyLoss
from torch import Tensor,LongTensor

# In[]

def ResortList(new_order, old_order, in_list):
    out = []
    for idx in new_order:
        old_idx = old_order.index(idx)
        out.append(in_list[old_idx])
        
    if isinstance(in_list,list):
        return out
    else:
        return np.array(out)


# In[]

def CalMetricVectors(Out_list, y_list):
    avr = Out_list.mean(axis = 1)
    negative_avr = np.array([Out_list[i][Out_list[i]<0].mean() for i in range(len(Out_list))])
    positive_avr = np.array([Out_list[i][Out_list[i]>0].mean() for i in range(len(Out_list))])
    
    soft_max = softmax(Out_list, axis = 1).max(axis = 1)
    hard_max = Out_list.max(axis = 1)
    loss = CrossEntropyLoss(reduction='none')
    return {
        'avr':avr,
        'negative_avr':negative_avr,
        'positive_avr':positive_avr,
        'hard_max':hard_max,
        'soft_max':soft_max,
        'CrossEntropyLoss':loss(Tensor(Out_list), LongTensor(y_list)).numpy(),
        }

# def CalMetricVectors_logits(Out_list, y_list):
#     # avr = Out_list.mean(axis = 1)
#     # negative_avr = np.array([Out_list[i][Out_list[i]<0].mean() for i in range(len(Out_list))])
#     # positive_avr = np.array([Out_list[i][Out_list[i]>0].mean() for i in range(len(Out_list))])
#     softmax_ = softmax(Out_list, axis = 1)
#     soft_max_top1 = softmax_.max(axis = 1)
#     soft_max_top5 = softmax_.max(axis = 1)
#     # hard_max = Out_list.max(axis = 1)
#     # loss = CrossEntropyLoss(reduction='none')
#     return {
#         # 'avr':avr,
#         # 'negative_avr':negative_avr,
#         # 'positive_avr':positive_avr,
#         # 'hard_max':hard_max,
#         'soft_max':soft_max,
#         # 'CrossEntropyLoss':loss(Tensor(Out_list), LongTensor(y_list)).numpy(),
#         }

# def CalMetricVectors_SubUnits(Out_list):
#     avr = Out_list.mean(axis = 1)
#     # negative_avr = np.array([Out_list[i][Out_list[i]<0].mean() for i in range(len(Out_list))])
#     # positive_avr = np.array([Out_list[i][Out_list[i]>0].mean() for i in range(len(Out_list))])
    
#     # soft_max = softmax(Out_list, axis = 1).max(axis = 1)
#     hard_max = Out_list.max(axis = 1)
#     # loss = CrossEntropyLoss(reduction='none')
#     return {
#         'avr':avr,
#         # 'negative_avr':negative_avr,
#         # 'positive_avr':positive_avr,
#         'hard_max':hard_max,
#         # 'soft_max':soft_max,
#         # 'CrossEntropyLoss':loss(Tensor(Out_list), LongTensor(y_list)).numpy(),
#         }    

# In[] 

# def CalMetrics(Out_list):
    
#     avr = Out_list.mean()
    
#     negative_avr = Out_list[Out_list<0].mean()
    
#     positive_avr = Out_list[Out_list>0].mean()
    
#     hard_max = Out_list.max(axis = 1).mean()
    
#     # soft_max = softmax(Out_list, axis = 1).max(axis = 1).mean()
    
#     return {
#         'avr':avr,
#         'negative_avr':negative_avr,
#         'positive_avr':positive_avr,
#         'hard_max':hard_max,
#         # 'soft_max':soft_max,
#         }


# In[]

def Get_N_Back_Slice(N, rep, size=4800, num_trials = 2):
    out = []
    for i in range(size*num_trials):
        if not bool((i // N) % 2) ^ rep:
            out.append(i)
    
    return out
    
def Choose_Correct_Trials(src,correct_list, if_correct):
    out = []
    for idx in src:
        if correct_list[idx] == if_correct:
            out.append(idx)
    
    return out

def Choose_Correct_Trials_2(correct_list, slc, if_correct):
    out = []
    
    tmp_list = [correct_list[i] for i in slc]
    
    for idx,item in enumerate(tmp_list):
        if item == if_correct:
            out.append(idx)
    
    return out


# In[]

def non_negative_(in_arr):
    out = in_arr.copy()
    out[out<0] = 0
    return out

# In[]
def GetDictItemFromIndex(in_dict, index):
    pos = in_dict['idx_log_list'].index(index)
    
    out_dict = {}
    
    for key in in_dict.keys():
        if len(in_dict[key]) != len(in_dict['idx_log_list']):
            continue
        new_key = key.split('_')[0]
        out_dict[new_key] = in_dict[key][pos]
    
    return out_dict

# In[]

def isAccurate(item,top_k = 1):
    y = item['y']
    out = item['Out']
    tops = list(out.argsort()[-top_k:])
    if y in tops:
        return True
    else:
        return False

def Cal_Accurate_list(in_dict, top_k = 1):
    tops = in_dict['Out_list'].argsort()[...,-top_k:]
    
    acc_list = []
    
    for i in range(len(in_dict['y_list'])):
        if in_dict['y_list'][i] in list(tops[i]):
            acc_list.append(True)
        else:
            acc_list.append(False)
    
    return acc_list

def Cal_Accurate_(out_list, y_list, top_k = 1):
    tops = out_list.argsort()[...,-top_k:]
    
    acc_list = []
    
    for i in range(len(y_list)):
        if y_list[i] in list(tops[i]):
            acc_list.append(True)
        else:
            acc_list.append(False)
    
    return acc_list

# In[]

def SaveObject(save_path, name,obj):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(os.path.join(save_path, name+'.pckl'),'wb') as f:
        pickle.dump(obj,f)


# In[]  add for VGG run

def GetFullRunName(run):
    sort_name = run['name'] + '-' + str(run['layer_index']) if 'layer_index' in run else run['name']
    return sort_name