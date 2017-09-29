#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import collections
import datetime
import glob
import numpy as np
import os
import scipy.io as sio
import tensorflow as tf

import options
import data_util
import architecture

slim = tf.contrib.slim

def get_target(opts, sample):
  target = None
  if opts.output_type == 'foe':
    target = sample['foe']
  elif opts.output_type == 'omega':
    target = sample['omega']
  elif opts.output_type == 'foeomega':
    target = tf.concat([sample['foe'], sample['omega']], 1)
  elif opts.output_type == 'foescaled':
    target = sample['foe']
  elif opts.output_type == 'invdepth':
    target = tf.one_hot(sample['invdepths'], opts.nclasses, axis=-1)
  return target

def get_loss(opts, target, output):
  """Get loss from inputs and outputs"""
  # Error
  if opts.output_type in ['foe', 'foeomega', 'omega', 'foescaled']:
    mse = tf.losses.mean_squared_error(output,target) # Mean Square Error
    # Cosine Error
    if opts.output_type == 'foe':
      ce = tf.losses.cosine_distance(output,tf.nn.l2_normalize(target,1),1)
    elif opts.output_type == 'foeomega':
      ce = tf.losses.cosine_distance(output[:,:3],tf.nn.l2_normalize(target[:,:3]))
    if opts.full_tensorboard:
      print("Creating summaries...")
      tf.summary.scalar('mean_square_error', mse)
      if opts.output_type in ['foe', 'foeomega']:
        tf.summary.scalar('cosine_distance', ce)
      tf.summary.scalar('weight_regularization', tf.losses.get_regularization_loss())
  if opts.output_type == 'invdepth':
    output_reshape = tf.reshape(output, [-1, opts.nclasses])
    target_reshape = tf.reshape(target, [-1, opts.nclasses])
    cross_entropy = tf.losses.softmax_cross_entropy(target_reshape, output_reshape)
    if opts.full_tensorboard:
      print("Creating summaries...")
      tf.summary.scalar('cross_entropy', cross_entropy)
      
	# This includes regularization losses
  loss = tf.losses.get_total_loss()
  tf.summary.scalar('loss', loss)
  return loss

def build_optimizer(opts, global_step):
  # Learning parameters post-processing
  num_batches = 1.0 * opts.sample_sizes['train'] / opts.batch_size
  decay_steps = int(num_batches * opts.learning_rate_decay_epochs)
  if opts.learning_rate_decay_type == 'fixed':
    learning_rate = tf.constant(opts.learning_rate, name='fixed_learning_rate')
  elif opts.learning_rate_decay_type == 'exponential':
    learning_rate = tf.train.exponential_decay(opts.learning_rate,
                                               global_step,
                                               decay_steps,
                                               opts.learning_rate_decay_rate,
                                               staircase=True,
                                               name='learning_rate')
  elif opts.learning_rate_decay_type == 'polynomial':
    learning_rate = tf.train.polynomial_decay(opts.learning_rate,
                                              global_step,
                                              decay_steps,
                                              opts.min_learning_rate,
                                              power=1.0,
                                              cycle=False,
                                              name='learning_rate')

  if opts.full_tensorboard:
    tf.summary.scalar('learning_rate', learning_rate)
  # TODO: add individual adam options to these
  if opts.optimizer_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif opts.optimizer_type == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif opts.optimizer_type == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate)
  elif opts.optimizer_type == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  return optimizer

def train(opts):
  """Train dataset"""
  # Build dataset
  dataset = data_util.get_dataset(opts)
  sample = dataset.load_batch('train')
  # Build network
  netLayers = architecture.build_architecture(opts, sample)
  # print("\n".join([ str(x) for x in netLayers ]))
  output = netLayers[-1]
  target = get_target(opts, sample)
  loss = get_loss(opts, target, output)
  # Build optimizer
  global_step = tf.train.get_or_create_global_step()
  optimizer = build_optimizer(opts, global_step)
  train_op = slim.learning.create_train_op(total_loss=loss,
                                           optimizer=optimizer,
                                           global_step=global_step,
                                           clip_gradient_norm=5)
  # Start training
  tf.logging.set_verbosity(tf.logging.INFO)
  num_batches = 1.0 * opts.sample_sizes['train'] / opts.batch_size
  max_steps = int(num_batches * opts.num_epochs)
  # TODO: Implement this: init_fn=_get_init_fn(),
  slim.learning.train(
          train_op=train_op,
          logdir=opts.save_dir,
          number_of_steps=max_steps,
          log_every_n_steps=opts.log_steps,
          save_summaries_secs=opts.save_summaries_secs,
          save_interval_secs=opts.save_interval_secs)

def debug_train(opts):
  # Build dataset
  dataset = data_util.get_dataset(opts)
  sample = dataset.load_batch_other('train')
  # Build network
  netLayers = architecture.build_architecture(opts, sample)
  # print("\n".join([ str(x) for x in netLayers ]))
  output = netLayers[-1]
  target = get_target(opts, sample)
  loss = get_loss(opts, target, output)
  # Build optimizer
  global_step = tf.train.get_or_create_global_step()
  optimizer = build_optimizer(opts, global_step)
  train_op = slim.learning.create_train_op(total_loss=loss,
                                           optimizer=optimizer,
                                           global_step=global_step,
                                           clip_gradient_norm=5)
  """Check for bugs in dataset and network"""
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1):
      for k in sample:
        x = sess.run(sample[k])
        print("{}: Shape {}\n{}".format(k,x.shape,x[0]))
      for i in range(len(netLayers)):
        x = sess.run(netLayers[i])
        print("Net Layer {}".format(i))
        print(x.shape)
      x = sess.run(output)
      print("output: Shape {}\n{}".format(x.shape, x[0]))
      x = sess.run(target)
      print("target: Shape {}\n{}".format(x.shape, x[0]))
      x = sess.run(loss)
      print("loss: {}".format(x))
      print("optimizer: {}".format(optimizer))
      x = sess.run(train_op)
      print("train_op: {}".format(x))
    coord.request_stop()
    coord.join(threads)

def main(opts):
  """Train with appropriate options"""
  if opts.debug:
    debug_train(opts)
  else:
    train(opts)


if __name__ == "__main__":
  opts = options.get_opts()
  main(opts)

