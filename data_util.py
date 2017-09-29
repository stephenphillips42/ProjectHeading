"""Module for generating or reorganizing various datasets.

This generates various datasets (dot, egomotion, etc.) and converts them to tfrecords (.tfrecords)
format as well as a numpy format (.npy).

TODO:
  * Implement dot dataset
  * Implement the egomotion dataset
"""

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
import scipy.misc as misc
import tensorflow as tf

import options

slim = tf.contrib.slim


# Tensorflow features
def _bytes_feature(value):
  """Create arbitrary tensor Tensorflow feature."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Classes
# TODO: Make class for float and feature
class TensorFeature(slim.tfexample_decoder.ItemHandler):
  """Custom class used for decoding serialized tensors."""
  def __init__(self, key, shape, dtype, description):
    super(TensorFeature, self).__init__(key)
    self._key = key
    self._shape = shape
    self._dtype = dtype
    self._description = description

  def get_feature_write(self, value):
    v = value.astype(self._dtype).tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

  def get_feature_read(self):
    return tf.FixedLenFeature([], tf.string)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    tensor = tf.decode_raw(tensor, out_type=self._dtype)
    return tf.reshape(tensor, self._shape)


class Dataset(object):
  """Class that manages creating, loading, and formatting the dataset."""
  # Constants
  MAX_IDX = 5000

  def __init__(self, opts):
    self.opts = opts
    self.dataset_dir = opts.dataset_dir
    self.type = opts.tf_type
    self.features = {
      'flow': TensorFeature(key='flow',
                            shape=opts.img_size + [4],
                            dtype=self.type,
                            description='Array of flow (u,v,x,y) values'),
      'foe': TensorFeature(key='foe',
                           shape=[3],
                           dtype=self.type,
                           description='Heading direction value'),
      'omega': TensorFeature(key='omega',
                             shape=[3],
                             dtype=self.type,
                             description='Angular velocity value'),
    }

  def process_features(self, loaded_features):
    features = {}
    for k, feat in self.features.iteritems():
      features[k] = feat.get_feature_write(loaded_features[k])
    return features

  def convert_dataset(self, out_dir, mode):
    """Writes synthetic flow data in .mat format to a TF record file."""
    outfile = lambda idx: os.path.join(out_dir, '{}-{:02d}.tfrecords'.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)
    matfiles = glob.glob(os.path.join(self.dataset_dir, mode, "[0-9]*.mat"))

    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    for index in tqdm(range(len(matfiles))):
      if file_idx > self.MAX_IDX:
        file_idx = 0
        if writer: writer.close()
        writer = tf.python_io.TFRecordWriter(outfile(record_idx))
        record_idx += 1
      loaded_features = sio.loadmat(matfiles[index])
      features = self.process_features(loaded_features)
      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      file_idx += 1

    if writer: writer.close()

    # And save out a file with the creation time for versioning
    with open(os.path.join(out_dir, '{}_timestamp.txt'.format(mode)), 'w') as date_file:
      date_file.write('TFrecord created {}'.format(str(datetime.datetime.now())))

  def load_batch_other(self, mode):
    assert mode in self.opts.sample_sizes, "Mode {} not supported".format(mode)
    data_sources = glob.glob(os.path.join(self.dataset_dir, mode + '-[0-9][0-9].tfrecords'))
    with tf.name_scope('input'):
      filename_queue = tf.train.string_input_producer(data_sources)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    keys_to_features = { k: v.get_feature_read() for k, v in self.features.iteritems() }
    items_to_descriptions = { k: v._description for k, v in self.features.iteritems() }
    features = tf.parse_single_example(serialized_example, features=keys_to_features)
    keys = self.features.keys()
    values = [ self.features[k].tensors_to_item({k:features[k]}) for k in keys ]
    values = tf.train.batch(
                values,
                batch_size=self.opts.batch_size,
                num_threads=self.opts.num_preprocessing_threads,
                capacity=5 * self.opts.batch_size)
    return dict(zip(keys,values))

  def load_batch(self, mode):
    """Return batch loaded from this dataset"""
    assert mode in self.opts.sample_sizes, "Mode {} not supported".format(mode)
    data_sources = glob.glob(os.path.join(self.dataset_dir, mode + '-[0-9][0-9].tfrecords'))
    # Build dataset provider
    keys_to_features = { k: v.get_feature_read() for k, v in self.features.iteritems() }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, self.features)
    items_to_descriptions = { k: v._description for k, v in self.features.iteritems() }
    dataset = slim.dataset.Dataset(
                data_sources=data_sources,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=self.opts.sample_sizes[mode],
                items_to_descriptions=items_to_descriptions)
    batch_size = self.opts.batch_size
    provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=self.opts.num_readers,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size,
                shuffle=self.opts.shuffle_data)
    # Extract features
    keys = self.features.keys()
    values = provider.get(keys)
    # Flow preprocessing here?
    values = tf.train.batch(
                values,
                batch_size=batch_size,
                num_threads=self.opts.num_preprocessing_threads,
                capacity=5 * batch_size)
    return dict(zip(keys,values))

class DepthDataset(Dataset):
  def __init__(self, opts):
    super(DepthDataset, self).__init__(opts)
    self.nclasses = opts.nclasses
    self.features = {
      'image': TensorFeature(key='image',
                            shape=opts.img_size + [3],
                            dtype=self.type,
                            description='Array of (r,g,b) values'),
      'invdepths': TensorFeature(key='invdepths',
                            shape=[ int(s/2) for s in opts.img_size],
                            dtype='int64',
                            description='Array of inverse depth values'),
    }

  def process_features(self, loaded_features):
    features = {}
    image = loaded_features['images'][0][:,:,:3].astype(self.type)
    features['image'] = self.features['image'].get_feature_write(image)
    newshape = [int(image.shape[0]/2), int(image.shape[1]/2)]
    invdepths = misc.imresize(loaded_features['invdepths'][:,:,0], newshape).astype('float32')
    invdepths_classes = np.round(self.nclasses*invdepths/255.).astype('int64')
    features['invdepths'] = self.features['invdepths'].get_feature_write(invdepths_classes)
    return features

def get_dataset(opts):
  if opts.dataset_type == 'cubes_full':
    return Dataset(opts)
  elif opts.dataset_type == 'cubes128':
    return Dataset(opts)
  elif opts.dataset_type == 'cubes128_depth':
    return DepthDataset(opts)

if __name__ == '__main__':
  opts = options.get_opts()
  if opts.debug:
    SEQ_TYPES = ['train',]
  else:
    SEQ_TYPES = ['test', 'train']
  dataset = get_dataset(opts)
  for idx, seq_type in enumerate(SEQ_TYPES):
    dataset.convert_dataset(opts.dataset_dir, seq_type)
