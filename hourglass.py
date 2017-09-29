import tensorflow as tf

# From: https://stackoverflow.com/questions/41211691/tensorflow-implementation-is-2x-slower-than-the-torchs-one
class stacked_hourglass(object):
  def __init__(self, opts, verbose=True, name='stacked_hourglass'):
    arch = opts.architecture
    self.noutputs = opts.noutputs
    self.nb_stack = arch['nb_stack']
    self.nlayers = arch['nlayers']
    self.width = arch['width']
    self.bn_decay = arch['bn_decay']
    self.epsilon = arch['epsilon']
    self.name = name
    self.verbose = verbose
    self.indent = 0

  def _print(self, x):
    if self.verbose:
      indent = "".join([" "] * self.indent)
      print("{}{}".format(indent, x))

  def __call__(self, x):
    with tf.name_scope(self.name) as scope:
      self._print("x: {}".format(x))
      padding = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], name='padding')
      self._print("padding: {}".format(padding))
      with tf.name_scope("preprocessing") as sc:
        conv1 = self._conv(padding, self.width/4, 7, 2, 'VALID', 'conv1')
        self._print("conv1: {}".format(conv1))
        norm1 = tf.contrib.layers.batch_norm(conv1, self.bn_decay, epsilon=self.epsilon, 
                            activation_fn=tf.nn.relu, scope=sc)
        self._print("norm1: {}".format(norm1))
        r1 = self._residual_block(norm1, self.width/2, 'r1')
        self._print("r1: {}".format(r1))
        pool = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], 'VALID', scope=scope)
        self._print("pool: {}".format(pool))
        r2 = self._residual_block(pool, self.width/2, 'r2')
        self._print("r2: {}".format(r2))
        r3 = self._residual_block(r2, self.width, 'r3')
        self._print("r3: {}".format(r3))
      hg = [None] * self.nb_stack
      ll = [None] * self.nb_stack
      ll_ = [None] * self.nb_stack
      out = [None] * self.nb_stack
      out_ = [None] * self.nb_stack
      sum_ = [None] * self.nb_stack
      with tf.name_scope('_hourglass_0_with_supervision') as sc:
        hg[0] = self._hourglass(r3, self.nlayers, self.width, '_hourglass')
        self._print("hg[{}]: {}".format(0,hg[0]))
        ll[0] = self._conv_bn_relu(hg[0], self.width, name='conv_1')
        self._print("ll[{}]: {}".format(0,ll[0]))
        ll_[0] = self._conv(ll[0],self.width,1,1,'VALID','ll')
        self._print("ll_[{}]: {}".format(0,ll_[0]))
        out[0] = self._conv(ll[0],self.noutputs,1,1,'VALID','out')
        self._print("out[{}]: {}".format(0,out[0]))
        out_[0] = self._conv(out[0],self.width,1,1,'VALID','out_')
        self._print("out_[{}]: {}".format(0,out_[0]))
        sum_[0] = tf.add_n([ll_[0], out_[0], r3])
        self._print("sum_[{}]: {}".format(0,sum_[0]))
      for i in range(1, self.nb_stack - 1):
        with tf.name_scope('_hourglass_' + str(i) + '_with_supervision') as sc:
          hg[i] = self._hourglass(sum_[i-1], self.nlayers, self.width, '_hourglass')
          self._print("hg[{}]: {}".format(i,hg[i]))
          ll[i] = self._conv_bn_relu(hg[i], self.width, name='conv_1')
          self._print("ll[{}]: {}".format(i,ll[i]))
          ll_[i] = self._conv(ll[i],self.nlayers,1,1,'VALID','ll')
          self._print("ll_[{}]: {}".format(i,ll_[i]))
          out[i] = self._conv(ll[i],self.noutputs,1,1,'VALID','out')
          self._print("out[{}]: {}".format(i,out[i]))
          out_[i] = self._conv(out[i],self.width,1,1,'VALID','out_')
          self._print("out_[{}]: {}".format(i,out_[i]))
          sum_[i] = tf.add_n([ll_[i], out_[i], sum_[i-1]])
          self._print("sum_[{}]: {}".format(i,sum_[i]))
      idx = self.nb_stack - 1
      with tf.name_scope('_hourglass_' + str(idx) + '_with_supervision') as sc:
        hg[self.nb_stack-1] = self._hourglass(sum_[idx - 1], self.nlayers, self.width, '_hourglass')
        self._print("hg[{}]: {}".format(idx,hg[idx]))
        ll[self.nb_stack-1] = self._conv_bn_relu(hg[idx], self.width, name='conv_1')
        self._print("ll[{}]: {}".format(idx,ll[idx]))
        out[self.nb_stack-1] = self._conv(ll[idx],self.nclasses,1,1,'VALID','out')
        self._print("out[{}]: {}".format(idx, out[idx]))
      return tf.stack(out)

  def _conv(self, inputs, nb_filter, kernel_size=1, strides=1, pad='VALID', name='conv'):
    with tf.name_scope(name) as scope:
      conv = tf.layers.conv2d(inputs=inputs,
                              filters=nb_filter,
                              kernel_size=kernel_size,
                              strides=[strides,strides],
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                              padding=pad,
                              kernel_regularizer=self.regularizer)
      return conv

  def _conv_bn_relu(self, inputs, nb_filter, kernel_size=1, strides=1, name=None):
    with tf.name_scope(name) as scope:
      conv = tf.layers.conv2d(inputs=inputs,
                              filters=nb_filter,
                              kernel_size=kernel_size,
                              strides=[strides,strides],
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                              kernel_regularizer=self.regularizer)
      norm = tf.contrib.layers.batch_norm(conv, self.bn_decay, epsilon=self.epsilon,
                                          activation_fn=tf.nn.relu, scope=scope)
      return norm

  def _conv_block(self, inputs, nb_filter_out, name='_conv_block'):
    with tf.name_scope(name) as scope:
      with tf.name_scope('norm_conv1') as sc:
        norm1 = tf.contrib.layers.batch_norm(inputs, self.bn_decay, epsilon=self.epsilon, 
                            activation_fn=tf.nn.relu, scope=sc)
        conv1 = self._conv(norm1, nb_filter_out / 2, 1, 1, 'SAME', name='conv1')
      with tf.name_scope('norm_conv2') as sc:
        norm2 = tf.contrib.layers.batch_norm(conv1, self.bn_decay, epsilon=self.epsilon, 
                            activation_fn=tf.nn.relu, scope=sc)
        conv2 = self._conv(norm2, nb_filter_out / 2, 3, 1, 'SAME', name='conv2')
      with tf.name_scope('norm_conv3') as sc:
        norm3 = tf.contrib.layers.batch_norm(conv2, self.bn_decay, epsilon=self.epsilon, 
                            activation_fn=tf.nn.relu, scope=sc)
        conv3 = self._conv(norm3, nb_filter_out, 1, 1, 'SAME', name='conv3')
      return conv3

  def _skip_layer(self, inputs, nb_filter_out, name='_skip_layer'):
    if inputs.get_shape()[3].__eq__(tf.Dimension(nb_filter_out)):
      return inputs
    else:
      with tf.name_scope(name) as scope:
        conv = self._conv(inputs, nb_filter_out, 1, 1, 'SAME', name='conv')
        return conv

  def _residual_block(self, inputs, nb_filter_out, name='_residual_block'):
    with tf.name_scope(name) as scope:
      _conv_block = self._conv_block(inputs, nb_filter_out)
      _skip_layer = self._skip_layer(inputs, nb_filter_out)
      return tf.add(_skip_layer, _conv_block)

  def _hourglass(self, inputs, n, nb_filter_res, name='_hourglass'):
    self.indent += 1
    with tf.name_scope(name) as scope:
      # Upper branch
      up1 = self._residual_block(inputs, nb_filter_res, 'up1')
      self._print("up1  {}: {}".format(n, up1))
      # Lower branch
      pool = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], 'VALID', scope=scope)
      self._print("pool {}: {}".format(n, pool))
      low1 = self._residual_block(pool, nb_filter_res, 'low1')
      self._print("low1 {}: {}".format(n, low1))
      if n > 1:
        low2 = self._hourglass(low1, n-1, nb_filter_res, 'low2')
      else:
        low2 = self._residual_block(low1, nb_filter_res, 'low2')
      self._print("low2 {}: {}".format(n, low2))
      low3 = self._residual_block(low2, nb_filter_res, 'low3')
      self._print("low3 {}: {}".format(n, low3))
      low4 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3] * 2,
                                              name='upsampling')
      self._print("low4 {}: {}".format(n, low4))
      if n < 4:
        output = tf.add(up1, low4, name='merge')
      else:
        output = self._residual_block(tf.add(up1, low4), nb_filter_res, 'low4')
      self._print("hourglass {}: {}".format(n, output))
      self.indent -= 1
      return output

class single_hourglass(stacked_hourglass):
  def __init__(self, opts, verbose=True, name='single_hourglass', regularizer=None):
    super(single_hourglass, self).__init__(opts, verbose, name)
    self.regularizer = regularizer
    self.nclasses = opts.nclasses

  def __call__(self, x):
    with tf.name_scope(self.name) as scope:
      self._print("x: {}".format(x))
      padding = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], name='padding')
      self._print("padding: {}".format(padding))
      with tf.name_scope("preprocessing") as sc:
        conv1 = self._conv(padding, 64, 7, 2, 'VALID', 'conv1')
        self._print("conv1: {}".format(conv1))
        norm1 = tf.contrib.layers.batch_norm(conv1, self.bn_decay, epsilon=self.epsilon, 
                            activation_fn=tf.nn.relu, scope=sc)
        self._print("norm1: {}".format(norm1))
        r1 = self._residual_block(norm1, self.width/2, 'r1')
        self._print("r1: {}".format(r1))
        # pool = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], 'VALID', scope=scope)
        # self._print("pool: {}".format(pool))
        r2 = self._residual_block(r1, self.width/2, 'r2')
        self._print("r2: {}".format(r2))
        r3 = self._residual_block(r2, self.width, 'r3')
        self._print("r3: {}".format(r3))
      with tf.name_scope('_hourglass_full') as sc:
        hg = self._hourglass(r3, 4, self.width, '_hourglass')
        self._print("hg: {}".format(hg))
        ll = self._conv_bn_relu(hg, self.width, name='conv_1')
        self._print("ll: {}".format(ll))
        ll_ = self._conv(ll,self.width,1,1,'VALID','ll')
        self._print("ll_: {}".format(ll_))
        out = self._conv(ll,self.nclasses,1,1,'VALID','out')
        self._print("out: {}".format(out))
      # TODO: Add other layers
      return [out]


if __name__ == "__main__":
  import os
  import sys
  import numpy as np
  import time
  import options
  opts = options.get_opts()
  n = 128
  arch = {}
  arch['nb_stack'] = 2
  arch['nlayers'] = opts.nlayers
  arch['width'] = 256
  arch['bn_decay'] = 0.9
  arch['epsilon'] = 1e-5
  opts.architecture = arch
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
  nb_stack = 1
  num_tests = 100
  batch_size = 32
  with tf.Graph().as_default():
    print "start build model..."
    _x = tf.placeholder(tf.float32, [None, n, n, 3])
    y = tf.placeholder(tf.float32, [None, n/2, n/2, opts.nclasses])
    output = single_hourglass(opts, 'single_hourglass')(_x)
    loss = tf.reduce_mean(tf.square(output - y))
    rmsprop = tf.train.RMSPropOptimizer(2.5e-4)
    print "build finished..."
    train_step = tf.Variable(0, name='global_step', trainable=False)
    train_rmsprop = rmsprop.minimize(loss, train_step)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      print "test..."
      xarr = np.random.rand(num_tests, batch_size, n, n, 3)
      yarr = np.random.rand(num_tests, batch_size, n/2, n/2, opts.nclasses)
      _time = time.clock()
      for u in range(0, num_tests):
        sess.run(loss, feed_dict={_x:xarr[u], y:yarr[u]})
      print "test forward:", (time.clock() - _time)/num_tests
      for u in range(0, num_tests):
        sess.run(train_rmsprop, feed_dict={_x:xarr[u], y:yarr[u]})
      print "test forward-backward:", (time.clock() - _time)/num_tests


