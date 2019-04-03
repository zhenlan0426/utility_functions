#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:39:39 2019

@author: will
"""

import torch.nn as nn
from torch.nn import Sequential
from torch.nn import GRU

def ConvBatchRelu(in_channel,out_channel,kernel_size,**kwargs):
    return Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,**kwargs),
                       nn.BatchNorm2d(out_channel),
                       nn.ReLU(inplace=True))
    
def ConvBatchRelu1D(in_channel,out_channel,kernel_size,**kwargs):
    return Sequential(nn.Conv1d(in_channel,out_channel,kernel_size,**kwargs),
                       nn.BatchNorm1d(out_channel),
                       nn.ReLU(inplace=True))
    
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
        return (output.transpose(1,2), h_n) if self.returnH else output.transpose(1,2)
    
    
    
