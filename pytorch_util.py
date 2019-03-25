import torch.nn as nn
from pretrainedmodels import utils
import torch
import time
import numpy as np


''' functions related to fine-tuning pretrained model '''
def gather_parameter_byName(name,model):
    return [w for n,w in model.named_parameters() if name in n]

def gather_parameter_nameList(nameList,model):
    parameter = []
    for name in nameList:
        parameter += gather_parameter_byName(name,model)
    return parameter

def trainable_parameter(model):
    return [p for p in model.parameters() if p.requires_grad]

def set_requires_grad(model, require_grad):
    if isinstance(model,(list,tuple)):
        for m in model:
            set_requires_grad_paraList(m.parameters(),require_grad)
    else:
        set_requires_grad_paraList(model.parameters(),require_grad)

def set_requires_grad_paraList(parameterList, require_grad):
    for para in parameterList:
        para.requires_grad = require_grad
            
def fine_tune_pretrainedmodels(model, outDim):
    # works only for models in pretrainedmodels as it depends on its API
    # switch out last linear layer
    # set grad to False for all layers except last linear
    set_requires_grad(model,False)
    InDim = model.last_linear.in_features
    model.last_linear = nn.Linear(InDim, outDim)
    return model

def feature_extract_pretrainedmodels(model):
    # works only for models in pretrainedmodels as it depends on its API
    # replace last linear with identity function
    # set grad to False
    model.last_linear = utils.Identity()
    set_requires_grad(model,False)
    return model

#''' functions related to data pipeline '''
#
#to_tensor = torch.from_numpy
#
#def switch_channel(x):
#    # swith HWC to CHW
#    dim = x.ndim
#    if dim == 3:
#        return x.transpose(2,0,1)
#    elif dim == 4:
#        return x.transpose(0,3,1,2)
#    
#to_device = lambda x:x.to('cuda:0')
#
#class Compose(object):
#    def __init__(self, funs):
#        self.funs = funs
#
#    def __call__(self, img):
#        for t in self.funs:
#            img = t(img)
#        return img
#    
#CHW_tensor_device = Compose([switch_channel,to_tensor,to_device])
#tensor_device = Compose([to_tensor,to_device])
#
#def combine_fun(f1,f2):
#    def fun(x,y):
#        return f1(x),f2(y)
#    return fun
#
#numpy2torch = combine_fun(CHW_tensor_device,tensor_device)
#
#class FunctionWrapOverDataset(Dataset):
#    def __init__(self, dataset, fun):
#        # dataset: Dataset object
#        # fun: fun to apply to output of dataset
#        self.dataset = dataset
#        self.fun = fun
#
#    def __len__(self):
#        return len(self.dataset)
#
#    def __getitem__(self, idx):
#        return self.fun(*self.dataset[idx])
#
#class FunctionWrapOverDataLoader(object):
#    def __init__(self, dl, func):
#        self.dl = dl
#        self.func = func
#
#    def __len__(self):
#        return len(self.dl)
#
#    def __iter__(self):
#        batches = iter(self.dl)
#        for b in batches:
#            yield (self.func(*b))    

''' functions related to training, evaluating, and predictions '''

def data2cuda(data):
    return [i.to('cuda:0') for i in data] if isinstance(data,(list,tuple)) else data.to('cuda:0')

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    # assume loss_func returns mean rather than sum
    since = time.time()
    opt.zero_grad()
    train_batch = len(train_dl.dataset)//train_dl.batch_size
    val_batch = len(valid_dl.dataset)//valid_dl.batch_size
    
    for epoch in range(epochs):
        # training #
        model.train()
        train_loss = 0
        for data in train_dl:
            loss = loss_func(model,data2cuda(data))
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
        
        # evaluating #
        val_loss = evaluate(model,valid_dl,loss_func,val_batch)
        print('epoch:{}, train_loss:{}, val_loss:{}'.format(epoch,train_loss/train_batch,val_loss))
    
    time_elapsed = time.time() - since
    print('Training completed in {}s'.format(time_elapsed))
    return model

def evaluate(model,dataloader,loss_func,n_batchs=None):
    model.eval()
    if n_batchs is None:
        n_batchs = len(dataloader.dataset)//dataloader.batch_size
    with torch.no_grad():
        loss = np.sum([loss_func(model,data2cuda(data)) for data in dataloader])
    return loss/n_batchs

def predict(model,dataloader,to_numpy=True):
    # dataloader return Xs only
    model.eval()
    with torch.no_grad():
        if to_numpy:
            return np.concatenate([model(data2cuda(data)).detach().numpy() for data in dataloader])
        else:
            return torch.cat([model(data2cuda(data)) for data in dataloader])











