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
import math
import numpy as np

class biasLayer(nn.Module):
    def __init__(self,dims):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(*dims))

    def forward(self, x):
        return x + self.bias

def SincBatchLeaky(in_channel,out_channel,kernel_size,**kwargs):
    kwargs['bias'] = False
    return Sequential(SincConv_fast2(in_channel,out_channel,kernel_size,**kwargs),
                       nn.BatchNorm1d(out_channel),
                       nn.LeakyReLU(inplace=True))
    
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


class weightedAverage(nn.Module):
    # d is number of models. Calculated w1*f1(x) + w2*f2(x)...
    def __init__(self,d):
        super().__init__()
        self.d = d
        self.score = nn.Parameter(torch.rand(d)/2)

    def forward(self,xs):
        # xs is of shape (n,...,d), where d is number of model to average
        weight = self.get_weight()
        return torch.matmul(xs,weight)  
   
    def get_weight(self):
        return F.softmax(self.score,0)

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
    Differ from the SincConv_fast in that it can take multiple input channels.
    Parameters
    ----------
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, in_channels, out_channels, kernel_size, sample_rate=16000, 
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super().__init__()

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
        
        
        
''' self-attention '''

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout),SublayerConnection(size, dropout)])
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)    
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "query, key, value have shape (N,L,C)"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))    
    
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))    
    
    
    
    
    
    
    
    
    
    
    