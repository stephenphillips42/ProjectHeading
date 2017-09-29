#!/usr/bin/python

import math
import tensorflow as tf
import numpy as np
import utils

import hourglass

def build_heading_network(opts, inputs, regularizer=None):
  """Build generic heading estimation architecture.
  
  Args:
    opts: dictionary with required members (nlayers,noutputs), and optional
          fields (filters,strides,kernelSizes,paddings,activations) in it.
            filters: List of positive integer, number of hidden features per layer
            strides: List of positive integer, strides of convolution per layer
            kernelSizes: List of positive integer, width of filters per layer
            padding: List of strings in  ('VALID', 'SAME')
            activations: List of strings in ('relu', 'leakyRelu', or 'reluSq')
    inputs: Input tensor for the network
    regularization: Regularization funtion to apply to weights
  Returns:
    List of tf.Tensors, each corresponding to layer of network. Final entry of the
    list is the output."""
  nlayers = opts.architecture.get('nlayers')
  noutputs = opts.architecture.get('noutputs')
  filters = opts.architecture.get('filters',[ 2.0**(6+i) for i in range(nlayers) ])
  strides = opts.architecture.get('strides', [ 2 for i in range(nlayers) ])
  kernel_sizes = opts.architecture.get('kernel_sizes', [ 3 for i in range(nlayers) ])
  activations = opts.architecture.get('activations', [ 'relu' for i in range(nlayers) ])
  layers = [inputs]

  # Build network
  with tf.variable_scope('weights'):
    for i in range(nlayers):
      with tf.variable_scope('block%02d' % i):
        conv = tf.layers.conv2d(inputs=layers[-1],
                                filters=filters[i],
                                kernel_size=kernel_sizes[i],
                                strides=[strides[i]] * 2,
                                padding=paddings[i],
                                activation=utils.activation(activations[i]),
                                kernel_regularizer=regularizer)
        layers.append(conv)
        if opts.use_batch_norm:
          layers.append(tf.layers.batch_normalization(layers[-1]))
    with tf.name_scope('linear_block'):
      inSize = reduce(mul, layers[-1].get_shape()[1:].as_list(), 1)
      reshape = tf.reshape(layers[-1],[-1,inSize])
      linear_layer = tf.layers.dense(inputs=reshape,
                                     units=noutputs,
                                     kernel_regularizer=regularizer)
      layers.append(reshape)
      layers.append(linear_layer)
    # TODO: Check for another fully connected output

    if opts.output_type == 'foe':
      with tf.name_scope('normalize'):
        layers.append(tf.nn.l2_normalize(layers[-1],1))
    elif opts.output_type == 'foeomega':
      with tf.name_scope('halfnormalize'):
        o = layers[-1]
        layers.append(tf.concat([ tf.nn.l2_normalize(o[:,:3],1), o[:,3:] ], 1))

  return layers

def build_heading_resnet(opts, inputs, regularizer=None):
  """Build generic heading estimation architecture.
  
  Args:
    opts: dictionary with required members (nlayers,noutputs), and optional
          fields (filters,strides,kernelSizes,paddings,activations) in it.
            filters: List of positive integer, number of hidden features per layer
            strides: List of positive integer, strides of convolution per layer
            kernelSizes: List of positive integer, width of filters per layer
            padding: List of strings in  ('VALID', 'SAME')
            activations: List of strings in ('relu', 'leakyRelu', or 'reluSq')
    inputs: Input tensor for the network
    regularization: Regularization funtion to apply to weights
  Returns:
    List of tf.Tensors, each corresponding to layer of network. Final entry of the
    list is the output."""
  nlayers = opts.architecture.get('nlayers')
  noutputs = opts.architecture.get('noutputs')
  filters = opts.architecture.get('filters',[ [2.0**(6+i)]*2 for i in range(nlayers) ])
  strides = opts.architecture.get('strides', [ 2 for i in range(nlayers) ])
  kernel_sizes = opts.architecture.get('kernel_sizes', [ 3 for i in range(nlayers) ])
  paddings = opts.architecture.get('paddings', [ "VALID" for i in range(nlayers) ])
  activations = opts.architecture.get('activations', [ 'relu' for i in range(nlayers) ])
  layers = [inputs]

  # Build network
  with tf.variable_scope('weights'):
    for i in range(nlayers):
      with tf.variable_scope('block%02d' % i):
        layer = utils.resnet_conv(inputs=layers[-1],
                                  filters=filters[i],
                                  kernel_size=kernel_sizes[i],
                                  strides=[strides[i]] * 2,
                                  activation=utils.activation(activations[i]),
                                  batch_norm=opts.use_batch_norm,
                                  kernel_regularizer=regularizer)
        layers.append(layer)
    with tf.name_scope('linear_block'):
      inSize = reduce(mul, layers[-1].get_shape()[1:].as_list(), 1)
      reshape = tf.reshape(layers[-1],[-1,inSize])
      linear_layer = tf.layers.dense(inputs=reshape,
                                     units=noutputs,
                                     kernel_regularizer=regularizer)
      layers.append(reshape)
      layers.append(linear_layer)
    # TODO: Check for another fully connected output

    if opts.output_type == 'foe':
      with tf.name_scope('normalize'):
        layers.append(tf.nn.l2_normalize(layers[-1],1))
    elif opts.output_type == 'foeomega':
      with tf.name_scope('halfnormalize'):
        o = layers[-1]
        layers.append(tf.concat([ tf.nn.l2_normalize(o[:,:3],1), o[:,3:] ], 1))

  return layers

def build_heading_resnet_2(opts, inputs, regularizer=None):
  """Build generic heading estimation resnet architecture (type 2)."""
  nlayers = 2
  noutputs = opts.noutputs
  filters = [ [ 256 ]*2 , [ 512 ]*2 ]
  strides = [ 4 ] * nlayers
  kernel_sizes = [ 7 ] * nlayers
  paddings = [ 'SAME' ] * nlayers
  activations = [ 'relu' ] * nlayers
  layers = [inputs]

  # Build network
  with tf.variable_scope('weights'):
    for i in range(nlayers):
      with tf.variable_scope('block%02d' % i):
        print('block%02d' % i)
        layer = utils.resnet_conv(inputs=layers[-1],
                                  filters=filters[i],
                                  kernel_size=kernel_sizes[i],
                                  strides=[strides[i]] * 2,
                                  activation=utils.activation(activations[i]),
                                  batch_norm=opts.use_batch_norm,
                                  kernel_regularizer=regularizer)
        layers.append(layer)
    in_size = reduce(mul, layers[-1].get_shape()[1:].as_list(), 1)
    reshape = tf.reshape(layers[-1],[-1,in_size])
    layers.append(reshape)
    if opts.use_fully_connected:
      print("fully_connected_layer")
      fc = utils.fully_connected_layer(inputs=layers[-1],
                                       mid_size=opts.fully_connected_size,
                                       out_size=noutputs,
                                       activation=utils.activation(activations[-1]),
                                       regularizer=regularizer)
      layers.append(fc)
    else:
      with tf.name_scope('linear_block'):
        linear_layer = tf.layers.dense(inputs=layers[-1],
                                       units=noutputs,
                                       kernel_regularizer=regularizer)
        layers.append(linear_layer)

    if opts.output_type == 'foe':
      with tf.name_scope('normalize'):
        layers.append(tf.nn.l2_normalize(layers[-1],1))
    elif opts.output_type == 'foeomega':
      with tf.name_scope('halfnormalize'):
        o = layers[-1]
        layers.append(tf.concat([ tf.nn.l2_normalize(o[:,:3],1), o[:,3:] ], 1))

  return layers

def build_architecture(opts, sample):
  """Build network from options."""
  opts.architecture = {}
  if opts.network_type == 'heading_network':
    opts.architecture['nlayers'] = opts.nlayers
    opts.architecture['filters'] = [ 2**(6+i) for i in range(opts.nlayers) ]
    opts.architecture['strides'] = [ 2 ] * opts.nlayers
    opts.architecture['kernel_sizes'] = [ 3 ] * opts.nlayers
    opts.architecture['paddings'] = [ 'VALID' ] * opts.nlayers
    opts.architecture['activations'] = [ 'relu' ] * opts.nlayers
  elif opts.network_type == 'heading_resnet':
    opts.architecture['nlayers'] = opts.nlayers
    opts.architecture['filters'] = [ [2**(6+i)]*2 for i in range(opts.nlayers) ]
    opts.architecture['strides'] = [ 2 ] * opts.nlayers
    opts.architecture['kernel_sizes'] = [ 3 ] * opts.nlayers
    opts.architecture['activations'] = [ 'relu' ] * opts.nlayers
    # TODO: Add back polymix
    # if opts['activation-type'] == "polymix":
    #   opts['activations'][0] = "relusq"
  elif opts.network_type == 'hourglass':
    opts.architecture['nb_stack'] = 1
    opts.architecture['nlayers'] = opts.nlayers
    opts.architecture['width'] = 256
    opts.architecture['bn_decay'] = 0.9
    opts.architecture['epsilon'] = 1e-5

  opts.architecture['noutputs'] = opts.noutputs

  regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=opts.weight_l1_decay,
																										scale_l2=opts.weight_decay)
  if opts.network_type == 'heading_network':
    return build_heading_network(opts, sample["flow"], regularizer=regularizer)
  elif opts.network_type == 'heading_resnet':
    return build_heading_resnet(opts, sample["flow"], regularizer=regularizer)
  elif opts.network_type == 'heading_resnet_2':
    return build_heading_resnet_2(opts, sample["flow"], regularizer=regularizer)
  elif opts.network_type == 'hourglass':
    hg = hourglass.single_hourglass(opts,
                                    verbose=False,
                                    name='network',
                                    regularizer=regularizer)
    return hg(sample["image"])
  else:
    return None # TODO: Proper error handling



