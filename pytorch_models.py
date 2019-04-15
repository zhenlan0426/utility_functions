#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:39:39 2019

@author: will
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential,Linear
from torch.nn import GRU
from functools import partial
from torch.nn.functional import glu
from torch.autograd import Variable
import math
import numpy as np

class biasLayer(nn.Module):
    def __init__(self,dims):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(*dims))

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

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y
    

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 


class SincConv_fast2(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, in_channels, out_channels, kernel_size, sample_rate=16000, 
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 10
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        diff = high_hz - low_hz
        interval = diff//out_channels
        
        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.rand((out_channels,in_channels,1))*diff+low_hz)

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.rand((out_channels,in_channels,1))*interval)

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)
        
        f_times_t_low = torch.mul(low, self.n_)
        f_times_t_high = torch.mul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band
        band_pass_right= torch.flip(band_pass_left,dims=[2])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=2)

        
        band_pass = band_pass / (band_pass_center)
        

        self.filters = (band_pass).view(
            self.out_channels, self.in_channels, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)    
