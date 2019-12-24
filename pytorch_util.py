import torch.nn as nn
#from pretrainedmodels import utils
import torch
import time
import numpy as np
from torch.nn.utils import clip_grad_value_
import copy
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
import math

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

# def feature_extract_pretrainedmodels(model):
#     # works only for models in pretrainedmodels as it depends on its API
#     # replace last linear with identity function
#     # set grad to False
#     model.last_linear = utils.Identity()
#     set_requires_grad(model,False)
#     return model

def differential_lr(model,base_lr,factor=2.6):
    # assume model is instance of Sequential
    # lr decrease by factor as you go to early layer
    # return value be consumed by optimizer
    # Note iterate over direct children
    filter_list = [m for m in model.children() if len(trainable_parameter(m))]
    length = len(filter_list)
    return [{"params": trainable_parameter(m), "lr": base_lr/(factor**(length-i))} 
            for i,m in enumerate(filter_list)]
    
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

def HWC2CHW(np_array):
    ndim = np_array.ndim 
    if ndim == 4:
        return np_array.transpose(0,3,1,2)
    elif ndim == 3:
        return np_array.transpose(2,0,1)
    else:
        print('wrong dims: {}'.format(ndim))
        
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def fit(epochs, model, loss_func, opt, train_dl, valid_dl=None,clip=0,clip_fun=clip_grad_value_,lossBest=1e6,patience=0):
    # assume loss_func returns mean rather than sum
    # within patience number of time, do not save model
    # if continue training, needs to pass in previous best val loss
    # if lossBest is None, do not save model
    since = time.time()
    opt.zero_grad()
    train_batch = len(train_dl.dataset)//(train_dl.batch_size if \
                                          train_dl.batch_size is not None else \
                                          train_dl.batch_sampler.batch_size)
    if valid_dl is not None:
        val_batch = len(valid_dl.dataset)//(valid_dl.batch_size if \
                                            valid_dl.batch_size is not None else \
                                            valid_dl.batch_sampler.batch_size)
        if lossBest is not None:
            best_model_wts = copy.deepcopy(model.state_dict())
        
    if clip!=0:
        paras = trainable_parameter(model)
        
    for epoch in range(epochs):
        # training #
        model.train()
        np.random.seed()
        train_loss = 0
        for data in train_dl:
            loss = loss_func(model,data2cuda(data))
            loss.backward()
            if clip!=0:
                clip_fun(paras,clip)
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
        
        # evaluating #
        if valid_dl is not None:
            val_loss = evaluate(model,valid_dl,loss_func,val_batch)
            print('epoch:{}, train_loss:{}, val_loss:{}'.format(epoch,train_loss/train_batch,val_loss))
        else:
            print('epoch:{}, train_loss:{}'.format(epoch,train_loss/train_batch))
        
        # save model
        if (valid_dl is not None) and (lossBest is not None):
            if val_loss<lossBest:
                lossBest = val_loss
                if epoch >= patience:
                    best_model_wts = copy.deepcopy(model.state_dict())
                
    if (valid_dl is not None) and (lossBest is not None): model.load_state_dict(best_model_wts)    
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
        out = torch.cat([model(data2cuda(data)) for data in dataloader])
        return out.cpu().detach().numpy() if to_numpy else out


''' data pipeline '''

class numpyArray(Dataset):
    def __init__(self,npArray):
        self.npArray = npArray
        
    def __len__(self):
        return self.npArray.shape[0]

    def __getitem__(self, idx):
        return self.npArray[idx]

class mixupWrapper(Dataset):
    """apply mixup over dataset
       Assume dataset return x,y
    """
    def __init__(self, dataset,alpha=1e-1):
        # dataset is Dataset instance
        # bigger alpha means more mix
        self.dataset = dataset
        self.n = len(dataset)
        self.alpha = alpha
        
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x1,y1 = self.dataset[idx]
        x2,y2 = self.dataset[np.random.randint(self.n)]
        w1 = np.random.beta(self.alpha,self.alpha)
        w2 = 1-w1
        return np.float32(w1*x1+w2*x2),np.float32(w1*y1+w2*y2)

def wrapTrainGen2TestGen(trainGen,arg=0):
    # given a gen that yields x,y..., yield the first arg
    # turn training gen into testing gen for prediction
    for data in iter(trainGen):
        yield data[arg]


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss