# -*- coding: utf-8 -*-
# 用于测试平移不变性（para-fovea priming）
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
import numpy as np
import sys
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
from cycler import cycler


import seaborn as sns
# from statannotations.Annotator import Annotator
# import seaborn as sns
# from scipy.stats import chi2_contingency
import pickle

from Utils.Utility import Get_N_Back_Slice,CalMetricVectors,ResortList,Choose_Correct_Trials_2,GetFullRunName
from Utils.Utility import Cal_Accurate_
# In[]
if_store_log_locally = True

task_name = 'N_Back-'
# gap_list = [1,2,4]

if not if_store_log_locally:
    save_output_path = os.path.join('Log')
else:
    save_output_path = os.path.join(local_log_root,'Log')

# In[]

used_gap = 1

USED_NUM = 2000

MAX_TIME_STEP = 10

# In[]

def CalcSoftmaxMetrics_GTlable():
    return

def GetRawLogits_GTlable():
    return

def GetPCError(Out_ori,Out_first,Out_rep):
    
    return {
        'first':Out_first[:,0].mean(),
        'repetition':Out_rep[:,0].mean(),
        'ori':Out_ori[:,0].mean()
        }

def GetRawLogits_Top1(Out_ori,Out_first,Out_rep):
    arg_max = np.argmax(Out_ori,axis=1)
    idx=(range(len(arg_max)),arg_max)
    
    return {
        'first':Out_first[idx].mean(),
        'repetition':Out_rep[idx].mean(),
        'ori':Out_ori[idx].mean()
        }

# [:,time_idx,layer_index,:]

def CalcSoftmaxMetrics_Top1(Out_ori,Out_first,Out_rep):
    softmax_ori = softmax(Out_ori, axis = 1)
    arg_max = np.argmax(softmax_ori,axis=1)
    idx=(range(len(arg_max)),arg_max)
    
    softmax_first = softmax(Out_first, axis = 1)
    softmax_rep = softmax(Out_rep, axis = 1)
    
    return {
        'first':softmax_first[idx].mean(),
        'repetition':softmax_rep[idx].mean(),
        'ori':softmax_ori[idx].mean()
        }
    
    # return {
    #     'first':Out_first[idx].mean(),
    #     'repetition':Out_rep[idx].mean(),
    #     'ori':softmax_ori[idx].mean()
    #     }

def GetRawLogits_Top5(Out_ori,Out_first,Out_rep):
    
    idx= (np.arange(len(Out_ori))[:, None],np.argsort(Out_ori,axis=1)[:,-5:-1]) #2-5,not including top1
    # idx= (np.arange(len(softmax_ori))[:, None],np.argsort(softmax_ori,axis=1)[:,-5:])
    
    _ori = Out_ori[idx].mean(axis=1)
    
    _first = Out_first[idx].mean(axis=1)
    _rep = Out_rep[idx].mean(axis=1)
    
    return {
        'first':_first.mean(),
        'repetition':_rep.mean(),
        'ori':_ori.mean()
        } 

def CalcSoftmaxMetrics_Top5(Out_ori,Out_first,Out_rep):
    softmax_ori = softmax(Out_ori, axis = 1)
    
    idx= (np.arange(len(softmax_ori))[:, None],np.argsort(softmax_ori,axis=1)[:,-5:-1]) #2-5,not including top1
    # idx= (np.arange(len(softmax_ori))[:, None],np.argsort(softmax_ori,axis=1)[:,-5:])
    
    softmax_ori = softmax_ori[idx].mean(axis=1)
    
    softmax_first = softmax(Out_first, axis = 1)[idx].mean(axis=1)
    softmax_rep = softmax(Out_rep, axis = 1)[idx].mean(axis=1)
    
    return {
        'first':softmax_first.mean(),
        'repetition':softmax_rep.mean(),
        'ori':softmax_ori.mean()
        } 



def GetLayerMetric(Out_ori,Out_first,Out_rep,metric_idx):
    # mean_ori = np.mean(Out_ori, axis = 1)
    
    # mean_first = np.mean(Out_first, axis = 1)
    # mean_rep = np.mean(Out_rep, axis = 1)
    
    # return {
    #     'first':Out_first[metric_idx].mean(),
    #     'repetition':Out_rep[metric_idx].mean(),
    #     'ori':Out_ori[metric_idx].mean()
    #     }
    return {
        'first':Out_first[:,metric_idx].mean(),
        'repetition':Out_rep[:,metric_idx].mean(),
        'ori':Out_ori[:,metric_idx].mean()
        }

def CalLayerMetric(metric_idx):
    def fun(*args, **kwargs):
        return GetLayerMetric(*args, **kwargs,metric_idx=metric_idx)
    
    return fun

target_index = 0 # 0:mean ;1:non-zero-per

metric_name_dict = {
    0:'Mean Value',
    1:'Non-zero Percentage'
    }

run_list = [
    {'name':'Top1_Logits','metric_function':GetRawLogits_Top1,'test_target':'Out_list'},
    {'name':'Top5_Logits','metric_function':GetRawLogits_Top5,'test_target':'Out_list'},
    {'name':'Top1_Softmax','metric_function':CalcSoftmaxMetrics_Top1,'test_target':'Out_list'},
    {'name':'Top5_Softmax','metric_function':CalcSoftmaxMetrics_Top5,'test_target':'Out_list'},
    ########  Layer activation
    {'name':'FC-2 Activation','layer_index':0,'metric_function':CalLayerMetric(target_index),'test_target':'layer_metrics_arr'}, # according to the order defined in layer_metric_func
    {'name':'FC-1 Activation','layer_index':1,'metric_function':CalLayerMetric(target_index),'test_target':'layer_metrics_arr'},
    {'name':'Conv-5 Activation','layer_index':2,'metric_function':CalLayerMetric(target_index),'test_target':'layer_metrics_arr'},
    {'name':'Conv-4 Activation','layer_index':3,'metric_function':CalLayerMetric(target_index),'test_target':'layer_metrics_arr'},
    ######## PCoder prediction error (notice the order of layer_index is reversed)
    {'name':'Conv-5 PC Error','layer_index':1,'metric_function':CalLayerMetric(0),'test_target':'pcoder_error_arr'},
    {'name':'Conv-4 PC Error','layer_index':0,'metric_function':CalLayerMetric(0),'test_target':'pcoder_error_arr'},
    ]

# In[]  

  
def Cal_First_Rep_Ori(SaveDict, SaveDict_Ori,test_target, metric_function, gap=1, layer_index=None, crop_num=None, time_idx=2):
    
    ret_dict = {}
    
    if not crop_num:
        crop_num = 4800
    
    for if_correct in [True,False]:
        
        slc_first = Get_N_Back_Slice(gap,False,crop_num)
        # slc_first = list(range(0,9600,2))
        
        # new_order = np.array(SaveDict['idx_log_list'])[slc_first]
        
        # old_order = SaveDict_Ori['idx_log_list']

        ##
        slc_ifcorrect = Choose_Correct_Trials_2(SaveDict_Ori['acc_top_1'],slc_first,if_correct)     
        ##         
        
        slc_rep = Get_N_Back_Slice(gap,True,crop_num)
        # slc_rep = list(range(1,9600,2))
        
        # new_order2 = np.array(SaveDict['idx_log_list'])[slc_rep]
        
        # for i in range(len(new_order2)):
        #     assert(new_order[i]==new_order2[i])

        # ##
        # slc_rep_ifcorrect = Choose_Correct_Trials(
        #     slc_rep,
        #     ResortList(new_order, old_order, SaveDict_Ori['acc_top_1']),
        #     if_correct
        #     )     
        # ##                 
        '''
        A major edit below. [:,layer_index,:] -> [:,axis_of_timestep,layer_index,:]
        due to time iteration in predify model.
        E.g. ori = SaveDict_Ori[test_target][slc_first][:,layer_index,:] if layer_index else SaveDict_Ori[test_target][slc_first] -> ori = SaveDict_Ori[test_target][slc_first][:,-1,layer_index,:] if layer_index else SaveDict_Ori[test_target][slc_first][:,-1:]
        '''        
        ori = SaveDict_Ori[test_target][slc_first][:,time_idx,layer_index,:] if layer_index else SaveDict_Ori[test_target][slc_first][:,time_idx,:]
        
        ori_=ori[slc_ifcorrect]
        
        first = SaveDict[test_target][slc_first][:,time_idx,layer_index,:] if layer_index else SaveDict[test_target][slc_first][:,time_idx,:]
        first_ = first[slc_ifcorrect]
        
        rep = SaveDict[test_target][slc_rep][:,time_idx,layer_index,:] if layer_index else SaveDict[test_target][slc_rep][:,time_idx,:]
        rep_ = rep[slc_ifcorrect]
        
        # ret_dict['correct' if if_correct else 'error'] = metric_function(ori_[0:1000],first_[0:1000],rep_[0:1000])
        ret_dict['correct' if if_correct else 'error'] = metric_function(ori_,first_,rep_)
    
    ret_dict['all'] = metric_function(ori,first,rep)
        
    return ret_dict

# In[]

# for time_idx in range(10):
    
#     y_list = SaveDict_Ori['y_list']
    
#     Out_arr = SaveDict_Ori['Out_list']
    
#     acc_top_1 = Cal_Accurate_(Out_arr[:,time_idx,:],y_list,1)
#     acc_top_5 = Cal_Accurate_(Out_arr[:,time_idx,:],y_list,5)
    
#     # count only odd trials 
#     acc_result = (acc_top_1[1::2].count(True)/len(acc_top_1) * 2, acc_top_5[1::2].count(True)/len(acc_top_5)*2)
    
#     print('Acc: ',acc_result)

# In[]
ori_filepath = os.path.join('Log','OriModelResult_'+task_name + str(used_gap) + '.pckl')

with open(ori_filepath,'rb') as f:
    SaveDict_Ori = pickle.load(f)

# === baseline (Vanilla / Ori model) repr stats ===
def _get_repr_stats_timeline(sd):
    if 'repr_stats_timeline' in sd:
        return sd['repr_stats_timeline']
    # fallback: replicate final stats if timeline missing
    base = sd.get('repr_stats', {})
    return [base for _ in range(MAX_TIME_STEP)]

repr_layer_order_baseline = SaveDict_Ori.get('repr_layer_names', [])
repr_within_timeline_ori = []
trace_timeline_ori = []
for stats_dict in _get_repr_stats_timeline(SaveDict_Ori):
    layer_means_t = {}
    layer_trace_means_t = {}
    for lname in repr_layer_order_baseline:
        if lname not in stats_dict:
            continue
        stats = stats_dict[lname]
        within = np.array(stats['within_cosine'])
        trace_var = np.array(stats['trace_var'])
        layer_means_t[lname] = np.nanmean(within)
        layer_trace_means_t[lname] = np.nanmean(trace_var)
    repr_within_timeline_ori.append(layer_means_t)
    trace_timeline_ori.append(layer_trace_means_t)


Cond_Sort_prime = {}
repr_mean_stats = {}
trace_mean_stats = {}
repr_within_timeline = {False: [], True: []}   # list over time, each element: dict layer->mean within
trace_timeline = {False: [], True: []}
for priming in [False,True]:
    priming_sufix = '_Prime' if priming else '_nonPrime'
    # filelist = os.listdir(save_output_path)

    file_path = os.path.join(save_output_path,'VGG_Result_N_Back-1_new'+priming_sufix+ '.pckl')
    with open(file_path,'rb') as f:
        SaveDict = pickle.load(f)

    # ==== 平均 sharpening 统计（最后时间步，200 类） ====
    layer_means = {}
    layer_trace_means = {}
    if 'repr_stats' in SaveDict and 'repr_layer_names' in SaveDict:
        for lname in SaveDict['repr_layer_names']:
            stats = SaveDict['repr_stats'][lname]
            within = np.array(stats['within_cosine'])
            trace_var = np.array(stats['trace_var'])
            layer_means[lname] = np.nanmean(within)
            layer_trace_means[lname] = np.nanmean(trace_var)
    repr_mean_stats[priming] = layer_means
    trace_mean_stats[priming] = layer_trace_means
    repr_layer_order = SaveDict.get('repr_layer_names', [])
    
    # ==== timeline sharpening 统计（全时间步） ====
    repr_stats_ts = SaveDict.get('repr_stats_timeline')
    if repr_stats_ts is None:
        # 回退：若无 timeline，复制最终统计以保持长度
        repr_stats_ts = [SaveDict.get('repr_stats', {}) for _ in range(MAX_TIME_STEP)]
    for time_idx in range(min(MAX_TIME_STEP, len(repr_stats_ts))):
        layer_means_t = {}
        layer_trace_means_t = {}
        stats_dict = repr_stats_ts[time_idx]
        for lname in repr_layer_order:
            if lname not in stats_dict:
                continue
            stats = stats_dict[lname]
            within = np.array(stats['within_cosine'])
            trace_var = np.array(stats['trace_var'])
            layer_means_t[lname] = np.nanmean(within)
            layer_trace_means_t[lname] = np.nanmean(trace_var)
        repr_within_timeline[priming].append(layer_means_t)
        trace_timeline[priming].append(layer_trace_means_t)
        
    ##############################
    time_step_list = []

    for time_idx in range(MAX_TIME_STEP):
        ##############################
        Cond_Sort = {}
        
        for run in run_list:
            Cond_Sort[run['name']] = {}
            # for inh_c in inh_c_list:
            #     Cond_Sort[GetFullRunName(run)][inh_c] = {}
         
        for run in run_list:        
            one_cond_metric = {}
            if 'Top1_Logits' in run['name']:
                acc_top_1 = Cal_Accurate_(SaveDict['Out_list'][:,time_idx,:],SaveDict['y_list'],1)
                acc_top_1_ori = Cal_Accurate_(SaveDict_Ori['Out_list'][:,time_idx,:],SaveDict_Ori['y_list'],1)
                # one_cond_metric['acc_top1'] = 
                
                tmp = {
                    'repetition':acc_top_1[1::2].count(True)/len(acc_top_1) * 2,
                    'ori':acc_top_1_ori[1::2].count(True)/len(acc_top_1_ori) * 2,
                    }
                
                Cond_Sort['Top1_Acc'] = {'all':tmp}           
            
            if 'Top5_Logits' in run['name']:
                acc_top_5 = Cal_Accurate_(SaveDict['Out_list'][:,time_idx,:],SaveDict['y_list'],5)
                acc_top_5_ori = Cal_Accurate_(SaveDict_Ori['Out_list'][:,time_idx,:],SaveDict_Ori['y_list'],5)
                # one_cond_metric['acc_top5'] = acc_top_5[1::2].count(True)/len(acc_top_5)*2
                tmp = {
                    'repetition':acc_top_5[1::2].count(True)/len(acc_top_5) * 2,
                    'ori':acc_top_5_ori[1::2].count(True)/len(acc_top_5_ori) * 2,
                    }
                Cond_Sort['Top5_Acc'] = {'all':tmp}
            
            layer_index = run['layer_index'] if 'layer_index' in run else None
            one_cond_metric.update(
                Cal_First_Rep_Ori(SaveDict, SaveDict_Ori,run['test_target'], 
                                  run['metric_function'], gap=used_gap,layer_index=layer_index,crop_num=USED_NUM//2,time_idx=time_idx)
                )
            
            sort_name = run['name']
            Cond_Sort[sort_name] = one_cond_metric
        time_step_list.append(Cond_Sort)
        
        ##############################
    Cond_Sort_prime[priming] = time_step_list
    
    
    # para_name_trans_dict = {
    #     'decay':'γ',
    #     'coeff':'α',
    #     # 'inh_c':'β',
    #     # 'decay':'Decay',
    #     # 'coeff':'Coeff',
    #     # 'offset':'Offset',
    #     'acc_top1_drop':'Accuracy-Top1 Drop',
    #     'acc_top5_drop':'Accuracy-Top5 Drop',
    #     }

# In[]

for key in Cond_Sort.keys():
    plt.figure()
    sort_name = key
    title = sort_name
    plt.title(title)
    curve_list = []
    for time_idx in range(MAX_TIME_STEP):
        values_np = Cond_Sort_prime[False][time_idx][title]['all']
        values_p = Cond_Sort_prime[True][time_idx][title]['all']
        curve_list.append((values_np['repetition'],values_p['repetition'],values_p['ori']))
    
    plt.gca().set_prop_cycle(cycler(color=['lightblue', 'orange','dimgray'])+cycler(linestyle=['-','-','--']))
    plt.plot(np.array(curve_list),
             
             label = ['NonPrime','Prime','Vanilla'],
             )
    
    if 'Error' in title:
        y_label = 'MSE'
    elif 'Activation' in title:    
        y_label = metric_name_dict[target_index]
    else:
        y_label = None
    
    if y_label:
        plt.ylabel(y_label)
    plt.xlabel('Time steps')
    plt.legend()
    plt.savefig(os.path.join('Fig',title+'.jpg'),dpi=600)

    plt.show()
    # In[]
    
    # title_list = ['Top1 Softmax',
    #               'Top2-5 Softmax',
    #               'Sub-Layer Activation'
    #               ]

time_idx = 0
    
fig, axs = plt.subplots(1,len(run_list),layout="constrained",figsize=(14,5))

# para_selected = (0.3,0.05)

for i in range(len(run_list)):
    ax = axs[i]
    sort_name = run_list[len(run_list)-i-1]['name']
    title = sort_name
    values_np = Cond_Sort_prime[False][time_idx][title]['all']
    values_p = Cond_Sort_prime[True][time_idx][title]['all']
    print([values_np['repetition'],values_p['repetition']])
    ax.bar(range(2),[values_np['repetition'],values_p['repetition']],
           color=['lightblue', 'orange'],
           width = 0.7,
           label = ['NonPrime','Prime'],
           tick_label  = ['NonPrime','Prime']
           )
    # ax.set_xlim([-1,4])
    ax.axhline(values_p['ori'],color='k',linestyle='--',label = 'Vanilla')
    
    # ax.set_title(title_list[i])
    ax.set_title(sort_name)
    

ax.legend(loc=5,borderaxespad = -10)

fig.suptitle('NonPrime Vs Prime')
fig.savefig(os.path.join('Fig','NonPrime Vs Prime'+'.jpg'),dpi=600)

fig.show()

# ===== Repr sharpening（Prime/NonPrime 相对 Vanilla 的百分比变化）=====
# 约定顺序：Conv4, Conv5, FC-1, FC-2
layer_key_order = ['hebb_pcoder4','hebb_pcoder5','hebbian_1','hebbian_2']
layer_labels = ['Conv4','Conv5','FC-1','FC-2']

means_np  = np.array([repr_mean_stats.get(False, {}).get(k, np.nan) for k in layer_key_order], dtype=float)
means_p   = np.array([repr_mean_stats.get(True,  {}).get(k, np.nan) for k in layer_key_order], dtype=float)
means_ori = np.array([repr_within_timeline_ori[-1].get(k, np.nan) for k in layer_key_order], dtype=float)
trace_np  = np.array([trace_mean_stats.get(False, {}).get(k, np.nan) for k in layer_key_order], dtype=float)
trace_p   = np.array([trace_mean_stats.get(True,  {}).get(k, np.nan) for k in layer_key_order], dtype=float)
trace_ori = np.array([trace_timeline_ori[-1].get(k, np.nan) for k in layer_key_order], dtype=float)

def pct_change(baseline, changed):
    denom = np.where(np.abs(baseline) > 1e-12, np.abs(baseline), np.nan)
    return (changed - baseline) / denom * 100.0

pct_within_np_vs_ori = pct_change(means_ori, means_np)
pct_within_p_vs_ori  = pct_change(means_ori, means_p)
pct_trace_np_vs_ori  = pct_change(trace_ori,  trace_np)
pct_trace_p_vs_ori   = pct_change(trace_ori,  trace_p)

width = 0.6
x = np.arange(len(layer_key_order))

# within_cosine 百分比变化（对 Vanilla）
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width/4, pct_within_np_vs_ori, width/2, color='lightblue', label='NonPrime vs Vanilla')
ax.bar(x + width/4, pct_within_p_vs_ori,  width/2, color='orange',    label='Prime vs Vanilla')
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(layer_labels, rotation=20)
ax.set_ylabel('Δ% vs Vanilla (within_cosine)')
ax.set_title('Sharpening Δ% vs Vanilla (final timestep, 200-class mean)')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join('Fig','Sharpening_within_pct.jpg'), dpi=300)
fig.show()

# trace_var 百分比变化（对 Vanilla）
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width/4, pct_trace_np_vs_ori, width/2, color='lightblue', label='NonPrime vs Vanilla')
ax.bar(x + width/4, pct_trace_p_vs_ori,  width/2, color='orange',    label='Prime vs Vanilla')
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(layer_labels, rotation=20)
ax.set_ylabel('Δ% vs Vanilla (trace_var)')
ax.set_title('Sharpening Δ% vs Vanilla (trace_var, final timestep, 200-class mean)')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join('Fig','Sharpening_trace_pct.jpg'), dpi=300)
fig.show()
# ===== Repr sharpening 时间轨迹（每层每指标，Prime/NonPrime/Vanilla，对 vanilla@t0 的百分比） =====
timeline_len = min(len(repr_within_timeline_ori), len(repr_within_timeline[False]), len(repr_within_timeline[True]), MAX_TIME_STEP)
time_axis = np.arange(timeline_len)

def pct_series_vs_base(series, base0):
    base = np.full_like(series, base0, dtype=float)
    return pct_change(base, series)

for metric_name, timeline_dict, fname_suffix, ori_dict in [
    ("within_cosine", repr_within_timeline, "within", repr_within_timeline_ori),
    ("trace_var", trace_timeline, "trace", trace_timeline_ori),
]:
    for lname, label in zip(layer_key_order, layer_labels):
        np_series  = np.array([timeline_dict[False][t].get(lname, np.nan) for t in range(timeline_len)], dtype=float)
        p_series   = np.array([timeline_dict[True][t].get(lname,  np.nan) for t in range(timeline_len)], dtype=float)
        ori_series = np.array([ori_dict[t].get(lname, np.nan) for t in range(timeline_len)], dtype=float)
        base0 = ori_series[0]

        pct_np  = pct_series_vs_base(np_series,  base0)
        pct_p   = pct_series_vs_base(p_series,   base0)
        pct_ori = pct_series_vs_base(ori_series, base0)

        plt.figure(figsize=(6,4))
        plt.plot(time_axis, pct_np,  marker='o', linestyle='-',  color='steelblue', label='NonPrime')
        plt.plot(time_axis, pct_p,   marker='o', linestyle='-',  color='coral',     label='Prime')
        plt.plot(time_axis, pct_ori, marker='o', linestyle='--', color='dimgray',   label='Vanilla')
        plt.axhline(0, color='k', linewidth=0.8)
        plt.xlabel('Time steps')
        plt.ylabel(f'Δ% vs Vanilla@t0 ({metric_name})')
        plt.title(f'{label} {metric_name} timeline')
        plt.legend()
        plt.tight_layout()
        fname = f'Sharpening_timeline_{fname_suffix}_{lname}.jpg'
        plt.savefig(os.path.join('Fig', fname), dpi=300)
        plt.show()

# 4/0
    # In[]
    
    
fig, axs = plt.subplots(1,len(run_list),layout="constrained",figsize=(20,5))

for i in range(len(run_list)):
    ax = axs[i]
    
    sort_name = run_list[len(run_list)-i-1]['name']
    title = sort_name
    
    width = 0.5
    
    pos_list_correct = [0,0.5]
    pos_list_incorrect = [1.5,2]
    
    pos_tick = [0.25,1.75]
    
    ax.bar(pos_list_correct,[Cond_Sort_prime[False][time_idx][title]['correct']['repetition'],
                             Cond_Sort_prime[True][time_idx][title]['correct']['repetition']], # NonPrime vs Prime
           color=['lightblue', 'orange'],
           width = width,
           # label = ['First','Repetition'],
           # tick_label  = ['First','Repetition']
           )
    
    ax.bar(pos_list_incorrect,[Cond_Sort_prime[False][time_idx][title]['error']['repetition'],
                             Cond_Sort_prime[True][time_idx][title]['error']['repetition']],
           color=['lightblue', 'orange'],
           width = width,
           label = ['NonPrime','Prime'],
           # tick_label  = ['First','Repetition']
           )
    ax.set_xlim([-0.5,2.5])
    ax.axhline(Cond_Sort[title]['all']['ori'],color='k',linestyle='--',label = 'Vanilla')
    ax.set_xticks(pos_tick,['Correct','Incorrect'])
    # ax.axhline(Cond_Sort[title][offset][para_selected]['correct']['ori'],color='k',linestyle='--',label = 'Vanilla Correct')
    # ax.axhline(Cond_Sort[title][offset][para_selected]['error']['ori'],color='k',linestyle='--',label = 'Vanilla Incorrect')
    
    # ax.set_title(title_list[i])
    ax.set_title(sort_name)
    
ax.legend(loc=5,borderaxespad = -10)

fig.suptitle('Correct Vs Incorrect'+priming_sufix)
fig.savefig(os.path.join('Fig','Correct Vs Incorrect'+priming_sufix+'.jpg'),dpi=600)
fig.show()

# In[]
4/0
# In[]

ori_acc_top1 = SaveDict_Ori['acc_result'][0]
ori_acc_top5 = SaveDict_Ori['acc_result'][1]

def GetHeatMap(ax, axis_0,axis_1,keyword,Cond_list,para_list_dict,para_name_trans_dict,fmt):
    
    score_mat = np.zeros((len(para_list_dict[axis_0]),len(para_list_dict[axis_1])))
    
    for Cond in Cond_list:
        pos_0 = para_list_dict[axis_0].index(Cond[0])
        pos_1 = para_list_dict[axis_1].index(Cond[1])
        if keyword == 'acc_top1':
            score_mat[pos_0][pos_1] = Cond_list[Cond][keyword] - ori_acc_top1
        elif keyword == 'acc_top5':
            score_mat[pos_0][pos_1] = Cond_list[Cond][keyword] - ori_acc_top5
        else:
            score_mat[pos_0][pos_1] = (Cond_list[Cond][keyword[0]][keyword[1]] - Cond_list[Cond]['all']['ori']) / Cond_list[Cond]['all']['ori']

    # print(score_mat)
    df = pd.DataFrame(score_mat,
                      index=pd.Series(para_list_dict[axis_0],name=para_name_trans_dict[axis_0]),
                      columns=pd.Series(para_list_dict[axis_1],name=para_name_trans_dict[axis_1])
                      )
    
    # print(df)
    
    sns.heatmap(df,annot=True, cmap='RdPu_r',fmt=fmt ,ax=ax,annot_kws={"size": 8})
    
    # ax = sns.heatmap(df,annot=True, cmap='RdPu_r',fmt=fmt)    
    # return ax

metric_list = [
    ['correct','first'],
    ['correct','repetition'],
    ['error','first'],
    ['error','repetition'],
]

# In[]   
# for metric in metric_list:
for run in run_list:
    sort_name = GetFullRunName(run)
    
    fig, axs = plt.subplots(1,
                            5 if sort_name=='Top1_Softmax' or sort_name=='Top5_Softmax' else 4,
                            layout="constrained",
                            figsize=(16,4) if sort_name=='Top1_Softmax' or sort_name=='Top5_Softmax'else(16,4))

    Cond_list = Cond_Sort[sort_name]
    
    extra_list = []
    
    if sort_name=='Top1_Softmax':
        extra_list.append('acc_top1')
    
    if run['name']=='Top5_Softmax':
        extra_list.append('acc_top5')
    
    col=0
    for metric in extra_list+metric_list:

        # fmt = ".1%" if metric in ['acc_top1','acc_top5'] else '.3g'
        fmt = ".1%" 
        GetHeatMap(axs[col],'decay', 'coeff', metric,Cond_list, para_list_dict, para_name_trans_dict,fmt)
        title_name = metric[0] + '-' +metric[1]
        
        title_name = para_name_trans_dict[title_name] if title_name in para_name_trans_dict else title_name
        
        axs[col].set_title(title_name)
        
        col+=1

    fig.suptitle(sort_name)
    # fig.savefig(os.path.join(r'D:\Music&Video&Picture&Download\Downloads\Fig',run['name']+'.jpg'),dpi=300)
    fig.savefig(os.path.join('Fig',sort_name+'.jpg'),dpi=300)
    fig.show()
    
    # break
        


# In[]
xx = model.classifier.hebbian_2.x_tmp.numpy()
xx2=model.features.hebb_channel_2.x_tmp.numpy()
xx3=model.features.hebb_channel_2.x_full.numpy()

# In[]

plt.figure()
plt.hist(xx.flatten(),bins=80,density=True)
plt.xlabel('Activation')
plt.ylabel('Histo Density')

# In[]

metri = layer_metrics_arr.cpu().numpy()

# In[]


# In[]

