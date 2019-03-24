#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 09:38:52 2018

@author: will
"""

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend 
from tensorflow.keras.applications import DenseNet121,DenseNet169,DenseNet201

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet_greyscale(blocks,input_shape,pooling,trainable):
    
    if blocks == 121:
        blocks = [6, 12, 24, 16]
    elif blocks == 169:
        blocks == [6, 12, 32, 32]
    elif blocks == 201:
        blocks == [6, 12, 48, 32]
        
    img_input = layers.Input(shape=input_shape)
    bn_axis = 3

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = models.Model(img_input, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = models.Model(img_input, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = models.Model(img_input, x, name='densenet201')

    # Load weights
    if blocks == [6, 12, 24, 16]:
        pretrained_model = DenseNet121(include_top=False,pooling=pooling)
    elif blocks == [6, 12, 32, 32]:
        pretrained_model = DenseNet169(include_top=False,pooling=pooling)
    elif blocks == [6, 12, 48, 32]:
        pretrained_model = DenseNet201(include_top=False,pooling=pooling)
    
    w = pretrained_model.layers[2].get_weights()[0].sum(2,keepdims=True)
    model.layers[2].set_weights([w])
    model.layers[2].trainable = trainable
    model.trainable = trainable
    
    for l1,l2 in zip(model.layers[3:],pretrained_model.layers[3:]):
        l1.set_weights(l2.get_weights())
        l1.trainable = trainable
    return model

#test = DenseNet_greyscale(121,(224,224,1),'max',False)