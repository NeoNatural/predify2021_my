# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:33:32 2025

@author: jxl1870
"""
import torch
import torch.nn as nn
# from torchvision.models import vgg16_bn,VGG16_BN_Weights

# In[] soft E%-max one-hot mask extractor

def emax_soft_gate(x, E=0.95, alpha=0.01, eps=1e-7):
    vmax = x.max()
    s = (x/(vmax+eps) - E) / alpha # x' in [0,1]
    return torch.sigmoid(s)


def get_default_hebb_layer_params():
    return [
        ######################################## FC1\FC2
        {'decay': 0, 'coeff': 0, 'cut_perc': 0},
        {'decay': 0, 'coeff': 0, 'cut_perc': 0},

        ################################ Conv4\Conv5
        {'decay': 0.5, 'coeff': 0.05, 'inh_c': 4},
        {'decay': 0.5, 'coeff': 0.05, 'inh_c': 4},
    ]



# In[] 20250521
class Hebb_Boost_C2(nn.Module): # 
    def __init__(self,in_channels=4096, decay = 0.5, coeff = 0.1,inh_c = 2,
                 sparse_thres = 0.95, cut_perc=0.05,ori_mode=False):
        super().__init__()        
               
        self.x_tmp = None
        self.layer_index = None
        self.ori_mode = ori_mode
        self.update_enabled = True
        self.compute_enabled = True
        self._pending_x_tmp = None
        self._pending_x_sparse = None

        
        if not ori_mode:
            self.decay = decay
            self.coeff = coeff
            self.inh_c = inh_c
            self.sparse_thres = sparse_thres
            self.cut_perc = cut_perc
            # self.refrac_coeff = 0.3
            
            self.nonlin = nn.ReLU()
            # self.nonlin = nn.Sigmoid()
            self.register_buffer('boost_weight', torch.zeros((in_channels, in_channels)))
            # self.register_buffer('refrac_value', torch.zeros((in_channels)))
  
    def forward(self,x):
        if self.ori_mode or not self.compute_enabled:
            x_view = x
            if x_view.dim() == 1:
                x_view = x_view.unsqueeze(0)
            elif x_view.dim() > 2:
                x_view = x_view.flatten(start_dim=1)
            self.x_tmp = x_view
            self._pending_x_tmp = None
            self._pending_x_sparse = None
            return x

        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.flatten(start_dim=1)

        boost_vec = torch.matmul(x, self.boost_weight.t())
        
        # inh_value = boost_vec.mean()
        # if boost_vec.max() >0:
        #     inh_value = boost_vec[boost_vec>(self.sparse_thres*boost_vec.max())].mean()
        # else:
        #     inh_value=0
        
        x_tmp = (x + boost_vec) #* (1 - self.refrac_value)
        max_val = x_tmp.max(dim=1, keepdim=True).values
        
        # self.refrac_value += self.x_tmp/max_val * self.refrac_coeff
        
        # self.x_tmp *= x.max() /self.x_tmp.max()
        
        x_tmp = x_tmp.masked_fill(x_tmp < max_val * self.cut_perc, 0)
        
        
        # threshold = torch.quantile(self.x_tmp, 0.9) # quantile thres
        threshold = max_val * self.sparse_thres # E%-max
        x_sparse = x_tmp.masked_fill(x_tmp < threshold, 0)

        pos_mask = x_sparse > 0
        pos_sum = (x_sparse * pos_mask).sum(dim=1, keepdim=True)
        pos_count = pos_mask.sum(dim=1, keepdim=True)
        denom = pos_sum / pos_count.clamp_min(1)
        if torch.isfinite(denom).all():
            x_sparse = torch.where(pos_count > 0, x_sparse / denom, x_sparse)

        self.x_tmp = x_tmp
        self._pending_x_tmp = x_tmp.detach()
        self._pending_x_sparse = x_sparse.detach()
        
        if self.update_enabled:
            self.commit_update()

        return x_tmp

    def commit_update(self):
        if self.ori_mode or not self.compute_enabled or not hasattr(self, "boost_weight") or self._pending_x_tmp is None or self._pending_x_sparse is None:
            return
        with torch.no_grad():
            if self._pending_x_tmp.dim() == 2:
                weight_tmp = self._pending_x_tmp.t().mm(self._pending_x_sparse) / self._pending_x_tmp.shape[0]
            else:
                weight_tmp = torch.outer(self._pending_x_tmp, self._pending_x_sparse)
            # weight_tmp.fill_diagonal_(0)
            self.boost_weight.mul_(self.decay).add_(self.coeff * weight_tmp)

    def zero_boost_weight(self):
        # 在 ori_mode 下没有 boost_weight，直接跳过
        if self.ori_mode or not hasattr(self, "boost_weight"):
            return
        self.boost_weight *= 0
        # self.refrac_value *= 0
        self._pending_x_tmp = None
        self._pending_x_sparse = None
    
    def set_para(self,decay, coeff,cut_perc=0.1, inh_c=4):
        self.decay = decay
        self.coeff = coeff
        self.inh_c = inh_c
        self.cut_perc = cut_perc

class Hebbian_VGG_Classifier(nn.Module):
    def __init__(self,in_classifier,ori_mode=False):
        super().__init__()
        self.pipe = in_classifier
        
        self.hebbian_1 = Hebb_Boost_C2(ori_mode=ori_mode)
        self.hebbian_2 = Hebb_Boost_C2(ori_mode=ori_mode)
        
        # for i, layer in enumerate(self.pipe.children()):
        #     if isinstance(layer,torch.nn.ReLU):
                
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        
        for i, layer in enumerate(self.pipe.children()):
            x = layer(x)
            
            if i==1:
                # assert(isinstance(layer,nn.ReLU))
                x = self.hebbian_1(x)
            
            if i==4:
                # assert(isinstance(layer,nn.ReLU))
                x = self.hebbian_2(x)
            
        return x
    
    def zero_boost_weight(self):
        self.hebbian_1.zero_boost_weight()
        self.hebbian_2.zero_boost_weight()
        
        
# In[] 20250523

class Hebb_VGG_Channel_Boost(nn.Module):
    def __init__(self,in_channels=512, decay = 0.5, coeff = 0.05, 
                 inh_c = 2,sparse_thres = 0.9,ori_mode=False):
        super().__init__()       
        
        self.ori_mode = ori_mode
        self.x_tmp = None
        self.x_full = None
        self.layer_index = None
        self.avgpool = nn.AdaptiveAvgPool2d(1)   
        self.update_enabled = True
        self.compute_enabled = True
        self._pending_x_tmp = None
        self._pending_x_sparse = None
        
        if not ori_mode:
            self.register_buffer('boost_weight', torch.zeros((in_channels, in_channels)))      
            self.decay = decay
            self.coeff = coeff
            self.inh_c = inh_c     
            self.sparse_thres = sparse_thres
            self.nonlin = nn.ReLU() 
            # self.nonlin = nn.Sigmoid()
        
    def forward(self,x):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x_avr = self.avgpool(x).flatten(1) # (B, C)
        # self.x_tmp = x_avr
        # self.x_full = x.detach()
        # return x

        if self.ori_mode or not self.compute_enabled:
            self.x_tmp = x_avr
            self.x_full = x.detach()
            self._pending_x_tmp = None
            self._pending_x_sparse = None
            return x

        boost_vec = torch.matmul(x_avr, self.boost_weight.t())
        inh_value = boost_vec.mean(dim=1, keepdim=True)
        x_tmp = self.nonlin(x_avr + boost_vec - self.inh_c * inh_value)

        max_val = x_tmp.max(dim=1, keepdim=True).values
        threshold = max_val * self.sparse_thres # E%-max
        x_sparse = x_tmp.masked_fill(x_tmp < threshold, 0)

        self.x_tmp = x_tmp
        self._pending_x_tmp = x_tmp.detach()
        self._pending_x_sparse = x_sparse.detach()

        if self.update_enabled:
            self.commit_update()
            
        scale = (x_tmp / (x_avr + 1e-12)).view(x.shape[0], -1, 1, 1)
        return x * scale

    def commit_update(self):
        if self.ori_mode or not self.compute_enabled or not hasattr(self, "boost_weight") or self._pending_x_tmp is None or self._pending_x_sparse is None:
            return
        with torch.no_grad():
            if self._pending_x_tmp.dim() == 2:
                weight_tmp = self._pending_x_tmp.t().mm(self._pending_x_sparse) / self._pending_x_tmp.shape[0]
            else:
                weight_tmp = torch.outer(self._pending_x_tmp, self._pending_x_sparse)
            # weight_tmp.fill_diagonal_(0)
            norm = torch.norm(weight_tmp)
            if torch.isfinite(norm) and norm > 0:
                weight_tmp = weight_tmp / norm
            self.boost_weight.mul_(self.decay).add_(self.coeff * weight_tmp)
    
    def zero_boost_weight(self):
        if self.ori_mode or not hasattr(self, "boost_weight"):
            return
        self.boost_weight *= 0
        self._pending_x_tmp = None
        self._pending_x_sparse = None
    
    def set_para(self,decay, coeff, inh_c=2):
        self.decay = decay
        self.coeff = coeff
        self.inh_c = inh_c

class Hebbian_VGG_Features(nn.Module):
    def __init__(self,in_features,ori_mode=False):
        super().__init__()
        self.pipe = in_features
        
        self.hebb_channel_1 = Hebb_VGG_Channel_Boost(ori_mode=ori_mode)
        self.hebb_channel_2 = Hebb_VGG_Channel_Boost(ori_mode=ori_mode)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.pipe.children()):
            x = layer(x)
            
            if i==33:
                # assert(isinstance(layer,nn.MaxPool2d))
                x = self.hebb_channel_1(x)
            
            if i==43:
                # assert(isinstance(layer,nn.MaxPool2d))
                x = self.hebb_channel_2(x)
                
        return x
    
    def zero_boost_weight(self):
        self.hebb_channel_1.zero_boost_weight()
        self.hebb_channel_2.zero_boost_weight()

# In[]

if __name__ == '__main__':
    
    from torchvision.models import vgg16_bn,VGG16_BN_Weights
    
    import time
    
    map_location = 'cuda'
    
    weights = VGG16_BN_Weights.IMAGENET1K_V1
    
    model = vgg16_bn(weights=weights).to(map_location)

    model.features = Hebbian_VGG_Features(model.features).to(map_location)

    model.classifier = Hebbian_VGG_Classifier(model.classifier).to(map_location)
    
    test_input = torch.rand((3,224,224)).to(map_location)

    model.eval()
    
    time0 = time.time()
    with torch.no_grad():
        for i in range(50):
            out1 = model(test_input.unsqueeze(0))        
            model.features.zero_boost_weight()
            model.classifier.zero_boost_weight()
    
    time1 = time.time() - time0
    
    print('Time cost per sample: ',time1/50)

    
