#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:11:21 2020

@author: will
"""


from sklearn.linear_model import LogisticRegression
import numpy as np

class continous2weight(object):
    ''' this class allow estimator that only takes in {0,1} to be fitted with tgt in [0,1]
    '''
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
    
    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = self.convert2weight(X,y,sample_weight)
        super().fit(X, y, sample_weight)
        
class LogitRegession(continous2weight,LogisticRegression):pass
    

if __name__ == '__main__':
    X = np.random.rand(1000,10)
    beta = np.random.rand(10)
    p = X@beta
    p = 1/(1+np.exp(-p))
    model = LogitRegession(C=1e5)
    model.fit(X,p)
    assert np.allclose(model.coef_,beta,atol=1e-4)