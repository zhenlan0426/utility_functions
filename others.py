from tensorflow.keras import backend as K
import gc
import tensorflow as tf
import os

def clear_all():
    K.clear_session()
    tf.reset_default_graph()
    K.get_session().close()
    gc.collect()
    
def creat_folder(root):
    # e.g. creat_folder('/home/will/Desktop/kaggle/WT')
    if not os.path.isdir(root+'/Data'):
        os.mkdir(root+'/Data')
    if not os.path.isdir(root+'/Model'):
        os.mkdir(root+'/Model')
    if not os.path.isdir(root+'/Code'):
        os.mkdir(root+'/Code')
    if not os.path.isdir(root+'/Submission'):
        os.mkdir(root+'/Submission')