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
import seaborn as sns
# from statannotations.Annotator import Annotator
# import seaborn as sns
# from scipy.stats import chi2_contingency
import pickle

from Utils.Utility import Get_N_Back_Slice,CalMetricVectors,ResortList,Choose_Correct_Trials_2,GetFullRunName
# In[]
if_store_log_locally = True

task_name = 'N_Back-'
gap_list = [1,2,4]

if not if_store_log_locally:
    save_output_path = os.path.join('Log')
else:
    save_output_path = os.path.join(local_log_root,'Log')

# In[]

used_gap = 1

# In[]

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

run_list = [
    {'name':'Top1_Softmax','metric_function':CalcSoftmaxMetrics_Top1,'test_target':'Out_list'},
    {'name':'Top5_Softmax','metric_function':CalcSoftmaxMetrics_Top5,'test_target':'Out_list'},
    {'name':'Layer_Mean','layer_index':0,'metric_function':CalLayerMetric(target_index),'test_target':'layer_metrics_arr'}, # according to the order defined in layer_metric_func
    {'name':'Layer_Mean','layer_index':1,'metric_function':CalLayerMetric(target_index),'test_target':'layer_metrics_arr'},
    {'name':'Layer_Mean','layer_index':2,'metric_function':CalLayerMetric(0),'test_target':'layer_metrics_arr'},
    {'name':'Layer_Mean','layer_index':3,'metric_function':CalLayerMetric(0),'test_target':'layer_metrics_arr'},

    ]

# In[]  

  
def Cal_First_Rep_Ori(SaveDict, SaveDict_Ori,test_target, metric_function, gap=1, layer_index=None):
    
    ret_dict = {}
    
    
    
    for if_correct in [True,False]:
        
        slc_first = Get_N_Back_Slice(gap,False)
        # slc_first = list(range(0,9600,2))
        
        # new_order = np.array(SaveDict['idx_log_list'])[slc_first]
        
        # old_order = SaveDict_Ori['idx_log_list']

        ##
        slc_ifcorrect = Choose_Correct_Trials_2(SaveDict_Ori['acc_top_1'],slc_first,if_correct)     
        ##         
        
        slc_rep = Get_N_Back_Slice(gap,True)
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
        
        ori = SaveDict_Ori[test_target][slc_first][:,layer_index,:] if layer_index else SaveDict_Ori[test_target][slc_first]
        
        ori_=ori[slc_ifcorrect]
        
        first = SaveDict[test_target][slc_first][:,layer_index,:] if layer_index else SaveDict[test_target][slc_first]
        first_ = first[slc_ifcorrect]
        
        rep = SaveDict[test_target][slc_rep][:,layer_index,:] if layer_index else SaveDict[test_target][slc_rep]
        rep_ = rep[slc_ifcorrect]
        
        # ret_dict['correct' if if_correct else 'error'] = metric_function(ori_[0:1000],first_[0:1000],rep_[0:1000])
        ret_dict['correct' if if_correct else 'error'] = metric_function(ori_,first_,rep_)
    
    ret_dict['all'] = metric_function(ori,first,rep)
        
    return ret_dict

# In[]

ori_filepath = os.path.join('Log','OriModelResult_'+task_name + str(used_gap) + '.pckl')

with open(ori_filepath,'rb') as f:
    SaveDict_Ori = pickle.load(f)

Cond_Sort_prime = {}
for priming in [False,True]:
    priming_sufix = '_Prime' if priming else '_nonPrime'
    # filelist = os.listdir(save_output_path)
    
    Cond_Sort = {}
    
    for run in run_list:
        Cond_Sort[GetFullRunName(run)] = {}
        # for inh_c in inh_c_list:
        #     Cond_Sort[GetFullRunName(run)][inh_c] = {}
    
    file_path = os.path.join(save_output_path,'VGG_Result_N_Back-1_new'+priming_sufix+ '.pckl')
    with open(file_path,'rb') as f:
        SaveDict = pickle.load(f)
         
    for run in run_list:        
        one_cond_metric = {}
        if run['name']=='Top1_Softmax':
            one_cond_metric['acc_top1']=SaveDict['acc_result'][0]  # - 0.4816
        
        if run['name']=='Top5_Softmax':
            one_cond_metric['acc_top5']=SaveDict['acc_result'][1] # - 0.7066
        
        layer_index = run['layer_index'] if 'layer_index' in run else None
        one_cond_metric.update(
            Cal_First_Rep_Ori(SaveDict, SaveDict_Ori,run['test_target'], 
                              run['metric_function'], gap=used_gap,layer_index=layer_index)
            )
        
        sort_name = GetFullRunName(run)
        Cond_Sort[sort_name] = one_cond_metric
    Cond_Sort_prime[priming] = Cond_Sort
    
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
    
    # title_list = ['Top1 Softmax',
    #               'Top2-5 Softmax',
    #               'Sub-Layer Activation'
    #               ]
    
fig, axs = plt.subplots(1,len(run_list),layout="constrained",figsize=(14,5))

# para_selected = (0.3,0.05)

for i in range(len(run_list)):
    ax = axs[i]
    sort_name = GetFullRunName(run_list[i])
    title = sort_name
    values_np = Cond_Sort_prime[False][title]['all']
    values_p = Cond_Sort_prime[True][title]['all']
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
# fig.legend(loc=5,borderaxespad = -5)
fig.show()

# 4/0
    # In[]
    
    
fig, axs = plt.subplots(1,len(run_list),layout="constrained",figsize=(20,5))

for i in range(len(run_list)):
    ax = axs[i]
    
    sort_name = GetFullRunName(run_list[i])
    title = sort_name
    
    width = 0.5
    
    pos_list_correct = [0,0.5]
    pos_list_incorrect = [1.5,2]
    
    pos_tick = [0.25,1.75]
    
    ax.bar(pos_list_correct,[Cond_Sort_prime[False][title]['correct']['repetition'],
                             Cond_Sort_prime[True][title]['correct']['repetition']], # NonPrime vs Prime
           color=['lightblue', 'orange'],
           width = width,
           # label = ['First','Repetition'],
           # tick_label  = ['First','Repetition']
           )
    
    ax.bar(pos_list_incorrect,[Cond_Sort_prime[False][title]['error']['repetition'],
                             Cond_Sort_prime[True][title]['error']['repetition']],
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




