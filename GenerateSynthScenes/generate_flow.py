import os
import sys
import OpenEXR
import Imath
import numpy as np
import scipy.io
import scipy.misc
import yaml
import matplotlib.colors as colors

class GenerateFlow(object):
  """GenerateFlow takes in the depth map from blender and
  generates an optic flow"""
  def __init__(self, params_file, mode):
    super(GenerateFlow, self).__init__()
    # Load params file
    with open(params_file,'r') as prms:
        params = yaml.load(prms)
    self.mode = mode
    # Compute derived parameters
    # Image size
    self.resolution_scale = params['resolution_percentage']/100.0
    self.width = int(params['image_width']*self.resolution_scale)
    self.height = int(params['image_height']*self.resolution_scale)
    # # Motion parameters
    # self.foe_sigma = params['foe_sigma']
    # The rest of the parameters we leave in the dictionary
    self.params = params

  def loadDepth(self,depth_file_name):
    """Open the OpenEXR file that """
    # Load
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depths_exr = OpenEXR.InputFile(depth_file_name)
    # Meta-data about the image
    dw = depths_exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    # Load the image data from EXR file
    depthsstr = depths_exr.channel('R', pt) # Only need 1 channel
    # Set up numpy array
    depths = np.fromstring(depthsstr, dtype = np.float32)
    depths.shape = (size[1], size[0]) # (row, col)
    return depths

  def generate(self,imNum):
    """Generate flow number imNum, based on the parameters given"""
    # Moding is to avoid overflow
    np.random.seed(pow(self.params[self.mode + '_seed_base_flow'],imNum,4294967291))
    dpth = "{}/{}/TrueDepth{:06d}_0001.exr"
    depth_file_name = dpth.format(self.params['base_dir'],self.mode,imNum)
    depths = self.loadDepth(depth_file_name)
    # Motion parameters
    foe = np.random.randn(3)*self.params['foe_sigma']
    omega = np.random.randn(3)*self.params['omega_sigma']

    # Calibrated coordinates
    (x_pixels, y_pixels) = np.meshgrid(np.arange(self.width), np.arange(self.height))
    # Intrinsics
    intrinsics = { "lens_scale":1, "height_trans":0, "width_trans":0 }
    if self.params['intrinsics_random']:
      intrinsics_path = "{}/{}/intrinsics{:06d}.mat".format(\
                          self.params['base_dir'],self.mode,imNum)
      intrinsics = scipy.io.loadmat(intrinsics_path)
    scaled_lens = intrinsics['lens_scale']*self.params['lens']
    self.focal_x = scaled_lens*(self.width/self.params['sensor_width'])
    self.focal_y = scaled_lens*(self.height/self.params['sensor_height'])
    self.center_x = self.width*(1+intrinsics['width_trans'])/2.0
    self.center_y = self.height*(1+intrinsics['height_trans'])/2.0
    x_cal = (x_pixels - self.center_x)/self.focal_x
    y_cal = (y_pixels - self.center_y)/self.focal_y
    # FOE component of flow
    if self.params['use_foe'] != 0:
      u_foe = np.divide(x_cal*foe[2] - foe[0], depths)
      v_foe = np.divide(y_cal*foe[2] - foe[1], depths)
    else:
      u_foe = 0*x_cal
      v_foe = 0*y_cal

    # Omega component of the flow
    # Coordinate polynomials
    self.x_y = np.multiply(x_cal,y_cal)
    self.x2_1 = np.square(x_cal)+1
    self.y2_1 = np.square(y_cal)+1

    if self.params['use_omega']:
      # Using precomputed polynomials of coordinates
      u_omega = omega[0]*self.x_y - omega[1]*self.x2_1 + omega[2]*y_cal
      v_omega = omega[0]*self.y2_1 - omega[1]*self.x_y - omega[2]*x_cal
    else:
      u_omega = 0
      v_omega = 0
    # Combine the parts of the flow
    u = u_foe + u_omega
    v = v_foe + v_omega
    uv = 0
    if self.params['tensorflow_format']:
      uv = np.concatenate((np.reshape(u,(1,self.height,self.width,1)), \
                           np.reshape(v,(1,self.height,self.width,1)), \
                           np.reshape(x_cal,(1,self.height,self.width,1)), \
                           np.reshape(y_cal,(1,self.height,self.width,1))), 3)
    else:
      uv = np.concatenate((np.reshape(u,(1,self.height,self.width)), \
                           np.reshape(v,(1,self.height,self.width))), 0)
    # Save out the flow
    path = "{}/{}/{:06d}.mat".format( \
                self.params['base_dir'],self.mode,imNum)
    tosave = {'flow':uv,
              'foe':foe,
              'omega':omega,
              'depths':depths,
              'focal_x':self.focal_x,
              'focal_y':self.focal_y,
              'center_x':self.center_x,
              'center_y':self.center_y,};
    scipy.io.savemat(path,tosave)
    if not self.params['austerity']:
      angle = (np.arctan2(v,u)+np.pi)/(2*np.pi)
      mag = np.sqrt(np.square(u) + np.square(v))
      mag = mag / np.amax(mag)
      rgb = colors.hsv_to_rgb(np.dstack([angle,mag,np.ones(mag.shape)]))
      path = "{}/{}/flow{:06d}.png".format( \
                  self.params['base_dir'],self.mode,imNum)
      scipy.misc.imsave(path,rgb)
    if self.params['cleanup']:
      os.remove(depth_file_name)
      if self.params['intrinsics_random']:
        os.remove(intrinsics_path)

if __name__ == '__main__':
  # Get Script Arguments
  argv = sys.argv
  argv = argv[argv.index("--") + 1:]  # get all args after "--"
  params_file = argv[0]
  mode = argv[1]
  imNum = int(argv[2])
  # Go crazy and generate flow
  # GenerateScene(imNum)
  g = GenerateFlow(params_file, mode)
  g.generate(imNum)



