#!/bin/python

import os
import sys
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# from tqdm import tqdm

# Machining Learning parameters
mytype = tf.float64
batch_size = 100
num_steps = 10**6
learning_rate = 1e-4
logdir = '/home/stephen/Documents/Research/logdir'

# Tensorflow helper functions
def get_figure():
  fig = plt.figure(num=0, figsize=(6, 8), dpi=300)
  fig.clf()
  return fig

def fig2rgb_array(fig, expand=True):
  fig.canvas.draw()
  buf = fig.canvas.tostring_rgb()
  ncols, nrows = fig.canvas.get_width_height()
  shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
  return np.fromstring(buf, dtype=np.uint8).reshape(shape)


# Simulation functions
# Function parameters
npoints_plot = 800
top = 3
bottom = -3
slope = 2.4
def mu(x):
  return slope*x

def sigma(x):
  return 0.1 + 0.4*(1.1 - np.cos(((2*np.pi)/((top-bottom)/2.0))*(x-bottom)))**2

def tf_sigma(x):
  return 0.5*tf.square(1.1 - tf.cos(((2*np.pi)/((top-bottom)/2.0))*(x-bottom)))

# Sampling functions
npoints_test = 800
def x_sample(npoints):
  x = np.random.rand(npoints,1)*(top - bottom) + bottom
  n = np.random.randn(npoints,1)
  y = mu(x) + n*sigma(x)
  return (x, y)

def get_chance_loss(x_test, y_test):
  X = np.concatenate((x_test,np.ones_like(x_test)),1)
  Y = y_test.reshape(-1,1)
  w_hat = np.linalg.solve(np.dot(X.T, X), np.dot(X.T,Y))
  print(w_hat.reshape(-1))
  print([slope, 0.0])
  Y_hat = np.dot(X, w_hat)
  lin_loss = np.mean(np.square(Y_hat - Y))
  chance_loss = np.mean(np.square(Y - np.mean(Y)))
  # # Plot things
  # plt.scatter(x, y, c='b')
  # plt.scatter(x, Y_hat, c='r')
  # plt.show()
  return Y_hat, lin_loss, chance_loss

def get_plot(x_test, y_test, y_lin, y_test_hat):
  fig = get_figure()
  plt.subplot('211').scatter(x_test, y_test, c='g')
  plt.subplot('211').scatter(x_test, y_lin, c='r')
  plt.subplot('211').scatter(x_test, y_test_hat[:,0], c='b')
  # plt.subplot('211').scatter(x_test, y_test_hat[:,0] + 2*np.abs(y_test_hat[:,1]), c='c')
  # plt.subplot('211').scatter(x_test, y_test_hat[:,0] - 2*np.abs(y_test_hat[:,1]), c='c')
  plt.subplot('212').scatter(x_test, sigma(x_test), c='g')
  plt.subplot('212').scatter(x_test, np.abs(y_test_hat[:,1]), c='b')
  return fig

custom_init = False
def main():
  # Build network
  I = 1 # Input units
  H = 32 # Hiden units
  O = 2 # Output units
  # Heavy computing bit
  b_ = np.linspace(bottom,top,H)
  diff = lambda i: (mu(b_[i]) - mu(b_[i-1]))/(b_[i] - b_[i-1])
  mu_init = np.array([ diff(1) ] + 
                     [ diff(i) - diff(i-1) for i in range(2, len(b_)-1) ] + 
                     [ 0, 0 ]).reshape(-1,1)
  diff = lambda i: (sigma(b_[i]) - sigma(b_[i-1]))/(b_[i] - b_[i-1])
  s_init = np.array([ diff(1) ] + 
                    [ diff(i) - diff(i-1) for i in range(2, len(b_)-1) ] + 
                    [ 0, 0 ]).reshape(-1,1)
  # Hidden weights, computed in numpy
  w_1_np = np.ones([I,H])
  b_1_np = -b_
  w_2_np = np.concatenate((mu_init,s_init), 1)
  b_2_np = np.array([mu(b_[0]), sigma(b_[0])])
  # w_2_np = mu_init # np.concatenate((mu_init,s_init), 1)
  # b_2_np = np.array([mu(b_[0])])
  if custom_init:
    w_1 = tf.Variable(tf.constant(w_1_np, dtype=mytype), name="weight_1")
    b_1 = tf.Variable(tf.constant(b_1_np, dtype=mytype), name="bias_1")
    w_2 = tf.Variable(tf.constant(w_2_np, dtype=mytype), name="weight_2")
    b_2 = tf.Variable(tf.constant(b_2_np, dtype=mytype), name="bias_2")
  else:
    w_1 = tf.Variable(tf.truncated_normal([I,H], dtype=mytype), name="weight_1")
    b_1 = tf.Variable(tf.truncated_normal([H], dtype=mytype), name="bias_1")
    w_2 = tf.Variable(tf.truncated_normal([H, O], dtype=mytype, stddev=1.0/np.sqrt(H)),
                                          name="weight_2")
    b_2 = tf.Variable(tf.truncated_normal([O], dtype=mytype), name="bias_2")

  def fn(x):
    lin = b_1 + tf.matmul(x, w_1)
    hidden = tf.nn.relu(lin)
    return b_2 + tf.matmul(hidden, w_2)

  x_test_, y_test_ = x_sample(1000)
  inds = x_test_[:,0].argsort()
  x_test_np = x_test_[inds]
  y_test_np = y_test_[inds]
  def numpy_test(x, y, w_1, b_1, w_2, b_2):
    lin = b_1 + np.dot(x, w_1)
    hidden = np.maximum(0,lin)
    y_np = b_2 + np.dot(hidden,w_2)
    l_sqd_np = np.divide(np.square(y[:,0] - y_np[:,0]), np.square(y_np[:,1]))
    l_sq_np = np.square(y[:,0] - y_np[:,0])
    l_log_np = np.log(np.square(y_np[:,1]))
    l_np = l_sqd_np + l_log_np
    return y_np, [l_np, l_sq_np, l_sqd_np, l_log_np]

  y_np, l_ = numpy_test(x_test_np, y_test_np, w_1_np, b_1_np, w_2_np, b_2_np)
  fig = plt.figure()
  ax1 = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)
  ax1.scatter(x_test_np, y_test_np, c='g')
  ax1.plot(x_test_np, mu(x_test_np), c='r')
  ax1.plot(x_test_np, y_np[:,0], c='b')
  ax1.plot(x_test_np, y_np[:,0] + 2*y_np[:,1], c='c')
  ax1.plot(x_test_np, y_np[:,0] - 2*y_np[:,1], c='c')
  ax2.scatter(x_test_np, l_[0], c='b')
  ax2.scatter(x_test_np, l_[1], c='r')
  ax2.scatter(x_test_np, l_[3], c='m')
  print("Initial losses: {:+.4e} (Square: {:+.4e}, Norm. Square: {:+.4e}, Log: {:+.4e})".format(
          np.mean(l_[0]), np.mean(l_[2]), np.mean(l_[1]), np.mean(l_[3])))
  plt.show()
  fig = plt.figure()
  ax1 = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)
  ax1.scatter(l_[1], l_[2], c='b')
  split_line = [ np.minimum(l_[1].min(),l_[2].min()),
                 np.maximum(l_[1].max(),l_[2].max()) ]
  ax1.plot(split_line, split_line, c='r')
  ax1.set_xlabel("Square Error")
  ax1.set_ylabel("Normalized Square Error")
  ax2.scatter(l_[1], l_[0], c='b')
  split_line = [ np.minimum(l_[1].min(),l_[0].min()),
                 np.maximum(l_[1].max(),l_[0].max()) ]
  ax2.plot(split_line, split_line, c='r')
  ax2.set_xlabel("Square Error")
  ax2.set_ylabel("MLE Error")
  plt.show()
  # sys.exit()
  # Train set
  x = tf.placeholder(mytype, shape=(None, 1))
  y = tf.placeholder(mytype, shape=(None, 1))
  y_hat = fn(x)
  # Test set
  x_test_, y_test_ = x_sample(npoints_test)
  x_test = tf.constant(x_test_, dtype=mytype)
  y_test = tf.constant(y_test_, dtype=mytype)
  y_test_hat = fn(x_test)
  y_lin_, lin_loss, chance_loss = get_chance_loss(x_test_, y_test_)

  # Losses
  alpha = 1.0
  loss_0 = tf.reduce_mean(tf.square(y[:,0] - y_hat[:,0]))
  loss_1 = tf.reduce_mean(tf.divide(tf.square(y[:,0] - y_hat[:,0]), tf.square(y_hat[:,1])))
  loss_2 = tf.reduce_mean(tf.log(tf.square(y_hat[:,1])))
  # loss_2 = tf.constant(0.0)
  loss = loss_1 + alpha * loss_2
  test_loss_1 = tf.reduce_mean(tf.square(y_test - y_test_hat[:,0]))
  test_loss_2 = tf.reduce_mean(tf.square(tf_sigma(x_test) - y_test_hat[:,1]))
  # test_loss_2 = tf.constant(0.0)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)

  # Tensorboard summaries
  tf.summary.scalar('total_loss', loss)
  tf.summary.scalar('norm_square_loss', loss_1)
  tf.summary.scalar('log_loss', loss_2)
  tf.summary.scalar('test_square_loss', test_loss_1)
  tf.summary.scalar('test_log_loss', test_loss_2)
  # # Weights (debug)
  # for i in range(H):
  #   tf.summary.scalar('w_1_{}'.format(i), w_1[0,i] - w_1_np[0,i])
  #   tf.summary.scalar('b_1_{}'.format(i), b_1[i] - b_1_np[i])
  #   tf.summary.scalar('w_2_{}_0'.format(i), w_2[i,0] - w_2_np[i,0])
  #   tf.summary.scalar('w_2_{}_1'.format(i), w_2[i,1] - w_2_np[i,0])
  # tf.summary.scalar('b_2_0'.format(1), b_2[0] - b_2_np[0])
  # tf.summary.scalar('b_2_1'.format(1), b_2[1] - b_2_np[1])
  merged = tf.summary.merge_all()
  # Figure summary
  fig = get_figure()
  vis_placeholder = tf.placeholder(tf.uint8, fig2rgb_array(fig).shape)
  vis_summary = tf.summary.image('plot', vis_placeholder)

  # Begin session
  with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
    tf.global_variables_initializer().run()
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    (x_, y_) = x_sample(batch_size)
    l_tf = sess.run([loss, loss_0, loss_1, loss_2], feed_dict={x: x_, y: y_})
    for step in range(num_steps):
      print("{}: Train: Full {:+.4e} "
            "(Square: {:+.4e}, Norm. Square: {:+.4e}, Log: {:+.4e})".format(
              step, l_tf[0], l_tf[1], l_tf[2], l_tf[3]))
      # # Numpy test
      # w_1_, b_1_, w_2_, b_2_, y_tf = sess.run([w_1, b_1, w_2, b_2, y_hat],
      #                                         feed_dict={x: x_, y: y_})
      # y_np, l_ = numpy_test(x_, y_, w_1_, b_1_, w_2_, b_2_)
      # print("{}: Numpy: Full {:+.4e} "
      #       "(Square: {:+.4e}, Norm. Square: {:+.4e}, Log: {:+.4e})".format(
      #         step, np.mean(l_[0]), np.mean(l_[1]), np.mean(l_[2]), np.mean(l_[3])))
      # Testing
      test_loss_1_, test_loss_2_, y_test_hat_ = \
        sess.run([test_loss_1, test_loss_2, y_test_hat])
      print("Test: Square Loss {:+.4e}, Sigma Loss {:+.4e}".format(test_loss_1_, test_loss_2_))

      # Figure
      fig = get_plot(x_test_, y_test_, y_lin_, y_test_hat_)
      image = fig2rgb_array(fig)
      summary_writer.add_summary(
        vis_summary.eval(feed_dict={vis_placeholder: image}))
      # Optimize
      (x_, y_) = x_sample(batch_size)
      _, summary = sess.run([train_op, merged], feed_dict={x: x_, y: y_})
      summary_writer.add_summary(summary, step)
      l_tf = sess.run([loss, loss_0, loss_1, loss_2], feed_dict={x: x_, y: y_})

main()
