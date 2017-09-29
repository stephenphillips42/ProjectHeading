#!/usr/bin/python

import tensorflow as tf
from operator import mul

# TODO: add deconvolution layer

def leakyRelu(x):
  alpha=0.01
  with tf.name_scope('leakyRelu'):
    return tf.maximum(x,tf.mul(x,alpha))

def relu_square(x):
  with tf.name_scope('relu_square'):
    return tf.pow(tf.maximum(x, 0.0),2.0)

def activation(activationType):
  if activationType == 'relu':
    return tf.nn.relu
  elif activationType == 'leakyrelu':
    return leakyRelu
  elif activationType == 'tanh':
    return tf.tanh
  elif activationType == 'relusq':
    return relu_square

def fully_connected_layer(inputs, mid_size, out_size, activation, regularizer=None):
  """Default caffe style MRSA Fully Connected Layer. Assumed x.get_shape(0) == None."""
  with tf.name_scope('fc_layer'): 
    layer = tf.layers.dense(inputs=inputs,
                            units=mid_size,
                            activation=activation,
                            kernel_regularizer=regularizer)
    return tf.layers.dense(inputs=layer, units=out_size, kernel_regularizer=regularizer)

def linear_layer(x, mid_size, out_size, regularizer=None):
  """Default caffe style MRSA Fully Connected Layer. Assumed x.get_shape(0) == None."""
  with tf.name_scope('linear_layer'): 
    in_size = reduce(mul, x.get_shape()[1:].as_list(), 1)
    x2 = tf.reshape(x,[-1,in_size])
    return tf.layers.dense(inputs=x2, units=outSize, kernel_regularizer=regularizer)

def resnet_conv(inputs,
                filters,
                kernel_size,
                strides=(1,1),
                activation=None,
                batch_norm=True,
                kernel_regularizer=None):
  with tf.variable_scope('resconv_1'):
    conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=filters[0],
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='SAME',
                             activation=activation,
                             kernel_regularizer=kernel_regularizer)
    if batch_norm:
      conv1 = tf.layers.batch_normalization(conv1)
  with tf.variable_scope('resconv_2'):
    conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=filters[1],
                             kernel_size=kernel_size,
                             strides=[1, 1],
                             padding='SAME',
                             activation=None,
                             kernel_regularizer=kernel_regularizer)
    if batch_norm:
      conv2 = tf.layers.batch_normalization(conv2)
  with tf.variable_scope('resconv_skip'):
    skip = tf.layers.conv2d(inputs=inputs,
                            filters=filters[1],
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='SAME',
                            activation=None,
                            kernel_regularizer=kernel_regularizer)
    if batch_norm:
      skip = tf.layers.batch_normalization(skip)
  with tf.variable_scope('resconv_output'):
    output = activation(conv2 + skip)
  return output



