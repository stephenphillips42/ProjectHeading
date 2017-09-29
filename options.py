#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
import tensorflow as tf
import yaml


DatasetType = collections.namedtuple('DatasetType',
                                     ['img_size','num_train','num_test','use_invdepth'])
DATASET_TYPES = {
  'cubes_full': DatasetType(img_size=[96,320],
                            num_train=27000,
                            num_test=3000,
                            use_invdepth=False),
  'cubes128': DatasetType(img_size=[128,128],
                          num_train=22000,
                          num_test=3000,
                          use_invdepth=False),
  'cubes128_depth': DatasetType(img_size=[128,128],
                                num_train=22000,
                                num_test=3000,
                                use_invdepth=True),
}
NETWORK_TYPES = [
  'heading_network',
  'heading_resnet',
  'heading_resnet_2',
  'hourglass'
]

def get_opts():
  """Parse arguments from command line and get all options for training."""
  parser = argparse.ArgumentParser(description='Train motion estimator')
  # Logging options
  parser.add_argument('--debug',
                      default=False,
                      type=bool,
                      help='Run in debug mode')
  parser.add_argument('--full_tensorboard',
                      default=True,
                      type=bool,
                      help='Display everything on Tensorboard')
  parser.add_argument('--log_steps',
                      default=10,
                      type=int,
                      help='Number of steps between logging')
  parser.add_argument('--save_summaries_secs',
                      default=20,
                      type=int,
                      help='Time between saving summaries for tensorboard (in seconds)')
  parser.add_argument('--save_interval_secs',
                      default=360,
                      type=int,
                      help='Time between saving models (in seconds)')

  # Directory and dataset options
  parser.add_argument('--save_dir',
                      default='/scratch/tfcheckpoints/test',
                      help='Directory to save out logs and checkpoints')
  # TODO: add other dataset types (namely kitti)
  parser.add_argument('--dataset_type',
                      default='cubes_full',
                      choices=DATASET_TYPES.keys(),
                      help='Dataset type to use for training')
  parser.add_argument('--dataset_dir',
                      default='/scratch/synthscene/cubes_full',
                      help='Directory where dataset is located')
  parser.add_argument('--output_type',
                      default='foe',
                      choices=['foe','foescaled','omega','foeomega','invdepth'],
                      help='Directory where dataset is located')
  parser.add_argument('--tf_type',
                      default='float32',
                      choices=['float32', 'float64', 'float128'],
                      help='Size of the image (overwritten)')
  parser.add_argument('--img_size',
                      default=None,
                      help='Size of the image (overwritten)')
  parser.add_argument('--sample_sizes',
                      default=None,
                      help='Size of the train/test datasets (overwritten)')

  # Dataset loading options
  parser.add_argument('--num_readers',
                      default=3,
                      type=int,
                      help='Number of readers for the queue')
  parser.add_argument('--shuffle_data',
                      default=True,
                      type=bool,
                      help='Shuffle data for loading')
  parser.add_argument('--num_preprocessing_threads',
                      default=3,
                      type=int,
                      help='Number of threads used to create the batches')

  # Architecture parameters
  # TODO: Add more network types
  parser.add_argument('--network_type',
                      default='heading_network',
                      choices=NETWORK_TYPES,
                      help='Network architecture to use')
  parser.add_argument('--nlayers',
                      default=4,
                      type=int,
                      help='Number of layers in the architecture')
  parser.add_argument('--noutputs',
                      default=3,
                      type=int,
                      help='Dimensionality of the output (overwritten)')
  parser.add_argument('--nclasses',
                      default=24,
                      type=int,
                      help='Number of classes')
  parser.add_argument('--activation_type',
                      default=None,
                      choices=['relu','leakyrelu','tanh','relusq'],
                      help='What type of activation to use')
  parser.add_argument('--use_fully_connected',
                      default=False,
                      type=bool,
                      help='Use fully connected layer at the end')
  parser.add_argument('--fully_connected_size',
                      default=1024,
                      type=int,
                      help='Size of the last fully connected layer')
  parser.add_argument('--use_batch_norm',
                      default=True,
                      type=bool,
                      help='Decision whether to use batch norm or not')
  parser.add_argument('--architecture',
                      default=None,
                      help='Helper variable for building the architecture type from network_type')

  # Machine learning parameters
  parser.add_argument('--num_epochs',
                      default=400,
                      type=int,
                      help='Number of epochs to run training')
  parser.add_argument('--batch_size',
                      default=128,
                      type=int,
                      help='Size for batches')
  parser.add_argument('--noise_level',
                      default=1e-2,
                      type=float,
                      help='Standard devation of white noise to add to input')
  parser.add_argument('--weight_decay',
                      default=4e-5,
                      type=float,
                      help='Weight decay regularization')
  parser.add_argument('--weight_l1_decay',
                      default=3e-5,
                      type=float,
                      help='L1 weight decay regularization')
  parser.add_argument('--optimizer_type',
                      default='adam',
                      choices=['adam','adadelta','momentum','sgd'],
                      help='Optimizer type for adaptive learning methods')
  parser.add_argument('--learning_rate',
                      default=1e-3,
                      type=float,
                      help='Learning rate for gradient descent')
  parser.add_argument('--learning_rate_decay_type',
                      default='exponential',
                      choices=['fixed','exponential','polynomial'],
                      help='Learning rate decay policy')
  parser.add_argument('--min_learning_rate',
                      default=1e-5,
                      type=float,
                      help='Minimum learning rate after decaying')
  parser.add_argument('--learning_rate_decay_rate',
                      default=0.95,
                      type=float,
                      help='Learning rate decay rate')
  parser.add_argument('--learning_rate_decay_epochs',
                      default=4,
                      type=int,
                      help='Number of epochs before learning rate decay')

  opts = parser.parse_args()

  # Post processing
  # Save out options
  if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
  with open(os.path.join(opts.save_dir, 'options.yaml'), 'w') as yml:
    yml.write(yaml.dump(opts.__dict__))
  # Architecture post-processing
  opts.img_size = DATASET_TYPES[opts.dataset_type].img_size
  opts.sample_sizes = {
    'train': DATASET_TYPES[opts.dataset_type].num_train,
    'test': DATASET_TYPES[opts.dataset_type].num_test
  }
  if DATASET_TYPES[opts.dataset_type].use_invdepth:
    opts.output_type = 'invdepth'

  if opts.output_type in ['foe','foescaled','omega']:
    opts.noutputs = 3
  elif opts.output_type in ['foeomega']:
    opts.noutputs = 6

  # Finished, return options
  return opts


