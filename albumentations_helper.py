#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 17:32:03 2018

@author: will
"""

def create_transform(fun):
    # fun should be albumentations object like Compose([HorizontalFlip(p=0.5),ShiftScaleRotate(0.1,0.1,45,p=1)])
    def transform(x):
        return fun(image=x)['image']
    return transform

def create_transform_with_mask(fun):
    # fun should be albumentations object like Compose([HorizontalFlip(p=0.5),ShiftScaleRotate(0.1,0.1,45,p=1)])
    def transform(x,y):
        aug = fun(image=x,mask=y)
        return aug['image'],aug['mask']
    return transform