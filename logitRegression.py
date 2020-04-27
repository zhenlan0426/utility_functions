#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:11:21 2020

@author: will
"""


from sklearn.linear_model import LogisticRegression
import numpy as np

class continous2weight(object):
    @staticmethod
    def convert2weight(X,y,weight):
        # y is in [0,1]
        X = np.concatenate([X,X],0)
        if weight is not None:
            weight = np.concatenate([weight,weight],0)
            weight = weight * np.concatenate([y,1-y],0)
        else:
            weight = np.concatenate([y,1-y],0)
        y = np.concatenate([np.ones_like(y),np.zeros_like(y)],0)
        return X,y,weight
    
class LogitRegession(LogisticRegression,continous2weight):
    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = self.convert2weight(X,y,sample_weight)
        super().fit(X, y, sample_weight)
        
        