#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:39:39 2019

@author: will
"""

import torch
import torch.nn as nn
from torch.nn import Sequential,Linear
from torch.nn import GRU
from functools import partial
from torch.nn.functional import glu

class biasLayer(nn.Module):
    def __init__(self,dims):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dims))

    def forward(self, x):
        return x + self.bias
    
def LinearLeaky(in_features, out_features, bias=True):
    return Sequential(Linear(in_features,out_features,bias),
                      nn.LeakyReLU(inplace=True))

def ConvBatchRelu(in_channel,out_channel,kernel_size,**kwargs):
    kwargs['bias'] = False
    return Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,**kwargs),
                       nn.BatchNorm2d(out_channel),
                       nn.ReLU(inplace=True))
    
def ConvBatchLeaky(in_channel,out_channel,kernel_size,**kwargs):
    kwargs['bias'] = False
    return Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,**kwargs),
                       nn.BatchNorm2d(out_channel),
                       nn.LeakyReLU(inplace=True))
    
def ConvBatchRelu1D(in_channel,out_channel,kernel_size,**kwargs):
    kwargs['bias'] = False
    return Sequential(nn.Conv1d(in_channel,out_channel,kernel_size,**kwargs),
                       nn.BatchNorm1d(out_channel),
                       nn.ReLU(inplace=True))
    
def ConvBatchLeaky1D(in_channel,out_channel,kernel_size,**kwargs):
    kwargs['bias'] = False
    return Sequential(nn.Conv1d(in_channel,out_channel,kernel_size,**kwargs),
                       nn.BatchNorm1d(out_channel),
                       nn.LeakyReLU(0.1,inplace=True))   
    
def ConvBatchRRelu1D(in_channel,out_channel,kernel_size,**kwargs):
    kwargs['bias'] = False
    return Sequential(nn.Conv1d(in_channel,out_channel,kernel_size,**kwargs),
                       nn.BatchNorm1d(out_channel),
                       nn.RReLU(inplace=True))    
    
def ConvGLU(in_channel,out_channel,kernel_size,**kwargs):
    kwargs['bias'] = False
    return Sequential(nn.Conv1d(in_channel,2*out_channel,kernel_size,**kwargs),
                       LambdaLayer(partial(glu,dim=1)),
                       nn.BatchNorm1d(out_channel))

def Conv2dGLU(in_channel,out_channel,kernel_size,**kwargs):
    kwargs['bias'] = False    
    return Sequential(nn.Conv2d(in_channel,2*out_channel,kernel_size,**kwargs),
                       LambdaLayer(partial(glu,dim=1)),
                       nn.BatchNorm2d(out_channel))    
    
class LambdaLayer(nn.Module):
    def __init__(self, lambda_):
        super(LambdaLayer, self).__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return self.lambda_(x)
    

class GRU_NCL(GRU):
    # wrap over GRU, input/output with shape (batch, channel, length).
    # function as drop-in replacement for conv1d
    def __init__(self,*args,returnH=False,**kwargs):
        kwargs['batch_first'] = True
        super().__init__(*args,**kwargs)
        self.returnH = returnH
        
    def forward(self,input_,h_0=None):
        output, h_n = super().forward(input_.transpose(1,2),h_0)
        return (output.transpose(1,2), h_n.transpose(0,1)) if self.returnH else output.transpose(1,2)

class skipConnectWrap1d(nn.Module):
    # add ResNet like skip connection
    # assume model does not change length    
    # model(x) + x. model(x), x should have shape (n,c1,len) and (n,c2,len)
    # if c1 > c2, add model(x)[:,:c2] + x
    # if c1 == c2, model(x) + x
    # else x[:,:c1] + model(x)
    def __init__(self,model):
        super().__init__()
        self.model = model 

    def forward(self,x):
        out = self.model(x)
        c1,c2 = out.shape[1], x.shape[1]
        if c1>c2:
            out[:,:c2] = out[:,:c2] + x
            return out
        elif c1==c2:
            return out + x
        else:
            x[:,:c1] = x[:,:c1] + out
            return x
        
class ConcatWrap1d(nn.Module):
    # add DenseNet like concat
    # assume model does not change length
    def __init__(self,model):
        super().__init__()
        self.model = model 

    def forward(self,x):
        out = self.model(x)
        return torch.cat([out,x],1)      