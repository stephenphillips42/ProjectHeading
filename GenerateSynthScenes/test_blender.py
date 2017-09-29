#!/bin/python

import os
import sys
import shutil
import subprocess
import tempfile
import time
import argparse
import yaml
from tqdm import tqdm

import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc as misc
import scipy.io as sio

def get_opts():
  parser = argparse.ArgumentParser(description='Generate synthetic scene datasets')
  parser.add_argument('--params',
                      default='params/params.yaml',
                      help='parameters for the dataset.') 
  parser.add_argument('--dataset_dir',
                      default='/scratch/synthscene',
                      help='Location to save the dataset.') 
  parser.add_argument('--mode',
                      default='/scratch/synthscene',
                      choices=['train', 'test'],
                      help='Location to save the dataset.') 
  parser.add_argument('--start',
                      default=0,
                      type=int,
                      help='Index to start dataset.') 
  parser.add_argument('--stop',
                      default=22000,
                      type=int,
                      help='Index to end dataset.') 
  return parser.parse_args()


def stats(x):
  s = "{} +/- {} [{}, {}]"
  print(s.format(np.mean(x), np.std(x), np.min(x), np.max(x)))

def load_depth(fname):
  """Open the OpenEXR file that """
  # Load
  pt = Imath.PixelType(Imath.PixelType.FLOAT)
  depths_exr = OpenEXR.InputFile(fname)
  # Meta-data about the image
  dw = depths_exr.header()['dataWindow']
  size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
  # Load the image data from EXR file
  depthsstr = depths_exr.channel('R', pt) # Only need 1 channel
  # Set up numpy array
  depths = np.fromstring(depthsstr, dtype = np.float32)
  depths.shape = (size[1], size[0]) # (row, col)
  return depths

def generate_scene(params, seed, timeout_time=8, debug=False):
  wdir = tempfile.mkdtemp()
  with open(os.path.join(wdir, 'params.yaml'), 'w') as f:
    f.write(yaml.dump(params['render_params']))

  scene = {}
  while not os.path.isfile(os.path.join(wdir, 'Image01.png')):
    with open(os.path.join(wdir, 'stdout.log'), 'w') as stdout_log:
      with open(os.path.join(wdir, 'stderr.log'), 'w') as stderr_log:
        status = subprocess.call([
            'timeout',
            str(timeout_time),
            'blender',
            '--background',
            '--python',
            'blender_scene_generator.py',
            '--',
            wdir,
            str(seed),
        ], stdout=stdout_log, stderr=stderr_log)
    if debug and not os.path.isfile(os.path.join(wdir, 'Image01.png')):
      with open(os.path.join(wdir, 'stderr.log'), 'r') as f:
        print("Error")
        print(f.read())
  # Load items
  scene['images'] = []
  scene['images'].append(misc.imread(os.path.join(wdir, 'Image01.png')))
  scene['images'].append(misc.imread(os.path.join(wdir, 'Image02.png')))
  scene['invdepths'] = misc.imread(os.path.join(wdir, 'InvDepth.png'))
  scene['labels'] = misc.imread(os.path.join(wdir, 'Labels.png'))
  scene['depths'] = load_depth(os.path.join(wdir, 'TrueDepth0001.exr'))
  misc_dict = sio.loadmat(os.path.join(wdir, 'misc.mat'))
  delkeys = ['__header__', '__globals__', '__version__']
  for i in range(len(delkeys)):
    del misc_dict[delkeys[i]]
  scene.update(misc_dict)
  # Delete temporary directory
  shutil.rmtree(wdir)
  # Return dict
  return scene

def generate_xy(scene):
  imshape = scene['images'][0].shape
  x_pixels, y_pixels = np.meshgrid(np.arange(imshape[0]), np.arange(imshape[1]))
  x_cal = (x_pixels - scene['center_x'])/scene['focal_x']
  y_cal = (y_pixels - scene['center_y'])/scene['focal_y']
  return x_cal, y_cal

def generate_flow(params, idx, mode='train'):
  seed = pow(params['seeds'][mode], idx, 4294967291)
  scene = generate_scene(params, seed)
  x_cal, y_cal = generate_xy(scene)
  scene['xy'] = np.stack((x_cal, y_cal), -1)
  # Compute flow
  # FOE component of flow
  foe = scene['foe'][0]
  u_foe = np.divide(x_cal*foe[2] - foe[0], scene['depths'])
  v_foe = np.divide(y_cal*foe[2] - foe[1], scene['depths'])
  # Omega component of the flow
  x_y = np.multiply(x_cal,y_cal)
  x2_1 = np.square(x_cal)+1
  y2_1 = np.square(y_cal)+1
  omega = scene['omega'][0]
  u_omega = omega[0]*x_y - omega[1]*x2_1 + omega[2]*y_cal
  v_omega = omega[0]*y2_1 - omega[1]*x_y - omega[2]*x_cal
  # Combine the parts of the flow
  u = u_foe + u_omega
  v = v_foe + v_omega
  if params['use_xy']:
    scene['flow'] = np.stack((u, v, x_cal, y_cal), -1)
  else:
    scene['flow'] = np.stack((u, v), -1)
  return scene

def generate_dataset(dataset_dir, params, mode='train', start=0, stop=22000):
  if not os.path.exists(os.path.join(dataset_dir, mode)):
    os.makedirs(os.path.join(dataset_dir, mode))
  with open(os.path.join(dataset_dir, 'params.yaml'), 'w') as f:
    f.write(yaml.dump(params))
  for idx in tqdm(range(start,stop), ascii=True):
    scene = generate_flow(params, idx, mode)
    sio.savemat(os.path.join(dataset_dir, mode, "{:06d}.mat".format(idx)), scene)

def profile(params):
  times = []
  for i in range(10):
    start = time.time()
    generate_scene(params, 4129 + i)
    end = time.time()
    times.append(end - start)
  print(times)
  print(np.mean(times))
  print(np.std(times))

def test(params, seed=4023):
  scene = generate_scene(params, seed)
  # im = scene['images'][0][:,:,:3]
  nbins = 32
  inf_thresh = 900.
  # im = scene['depths'].astype(np.float32)
  im = scene['invdepths'][:,:,0].astype(np.float32)
  print(len(scene['images']))
  print(stats(im))
  print(im.shape)
  p = 1.0
  mylevels = np.linspace(np.power(im.min(), p), np.power(im.max(), p), nbins+1)
  mylevels = np.power(mylevels, 1./p)
  plt.imshow(im, cmap='inferno')
  plt.contour(im, cmap='binary', levels=mylevels)
  plt.show()
  n, b, _ = plt.hist(im.reshape(-1),bins=mylevels)
  print(n)
  plt.show()
  xx, yy = np.meshgrid(np.arange(0,im.shape[1]), np.arange(0,im.shape[0]))
  im2 = np.copy(im) #.reshape(-1)
  for l in range(1, len(mylevels)):
    im[(im > mylevels[l-1]) & (im < mylevels[l])] = mylevels[l-1]
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(xx.reshape(-1), yy.reshape(-1), im.reshape(-1), s=4, c=im.reshape(-1))
  plt.show()

if __name__ == '__main__':
  opts = get_opts()
  with open(opts.params) as yaml_dict:
    params = yaml.load(yaml_dict)
  generate_dataset(opts.dataset_dir, params, opts.mode, opts.start, opts.stop)
