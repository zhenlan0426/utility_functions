#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:39:39 2019

@author: will
"""

import torch.nn as nn
from torch.nn import Sequential

def ConvBatchRelu(in_channel,out_channel,kernel_size,**kwargs):
    return Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,**kwargs),
                       nn.BatchNorm2d(out_channel),
                       nn.ReLU(inplace=True))
    
