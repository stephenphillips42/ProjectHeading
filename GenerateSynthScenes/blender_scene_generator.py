#!/usr/bin/blender --background --python
# Blender libraries
import bpy
from bpy import context
# Standard libraries
import math
from math import sin, cos, radians
import random as rand
import time
# System libraries
import sys
import os
# Save out libraries
import scipy.io as sio

# Load Yaml Parameters
sys.path.append(os.getcwd())
import yamllocal as yaml

# Implemented from transformations.py by Christoph Gohlke at UCI
# epsilon for testing whether a number is close to zero
_EPS = sys.float_info.epsilon * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

##################### Helper Functions ########################
def eprint(mystring):
  print(mystring,file=sys.stderr)

def debugprint(mystring):
  print(mystring,file=sys.stderr)
  # pass

def printmatrix(M):
  printstr = \
    "\n".join([
      "[ " + ", ".join([ "{:+.3e}".format(M[i][j]) for j in range(len(M[i])) ]) + " ]"
      for i in range(len(M))
    ])
  print(printstr)

def convert_to_base_256(x):
  x1 = x % 256
  x2 = (x // 256) % 256
  x3 = (x // 256**2) % 256
  return (x1, x2, x3)

def eye(n):
  return [ [ 1. if i == j else 0. for j in range(n) ] for i in range(n) ]

def transpose(A):
  return [ [ A[j][i] for j in range(len(A))  ] for i in range(len(A[0]))]
  
def matadd(A, B):
  M = [ [ 0. for j in range(len(A[0])) ] for i in range(len(A)) ]
  for i in range(len(A)):
    for j in range(len(A[0])):
      M[i][j] = A[i][j] + B[i][j]
  return M

def scale(a, A):
  return [ [ a * A[i][j] for j in range(len(A[0])) ] for i in range(len(A)) ]

def matmul(A, B):
  M = [ [ 0. for j in range(len(B[0])) ] for i in range(len(A)) ]
  for i in range(len(A)):
    for j in range(len(B[0])):
      M[i][j] = sum([ A[i][k] * B[k][j] for k in range(len(B)) ])
  return M

def sinc(x):
  return max(_EPS, math.sin(x)) / max(_EPS, x)

def skew(omega):
   return [ [       0., -omega[2],  omega[1] ],
            [ omega[2],        0., -omega[0] ],
            [-omega[1],  omega[0],        0. ] ]

# Implemented from transformations.py by Christoph Gohlke at UCI
def euler_matrix(ai, aj, ak, axes='sxyz'):
  firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]

  i = firstaxis
  j = _NEXT_AXIS[i+parity]
  k = _NEXT_AXIS[i-parity+1]

  if frame:
    ai, ak = ak, ai
  if parity:
    ai, aj, ak = -ai, -aj, -ak

  si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
  ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
  cc, cs = ci*ck, ci*sk
  sc, ss = si*ck, si*sk

  M = [ [0. for i in range (3) ]  for i in range (3) ]
  if repetition:
    M[i][i] = cj
    M[i][j] = sj*si
    M[i][k] = sj*ci
    M[j][i] = sj*sk
    M[j][j] = -cj*ss+cc
    M[j][k] = -cj*cs-sc
    M[k][i] = -sj*ck
    M[k][j] = cj*sc+cs
    M[k][k] = cj*cc-ss
  else:
    M[i][i] = cj*ck
    M[i][j] = sj*sc-cs
    M[i][k] = sj*cc+ss
    M[j][i] = cj*sk
    M[j][j] = sj*ss+cc
    M[j][k] = sj*cs-sc
    M[k][i] = -sj
    M[k][j] = cj*si
    M[k][k] = cj*ci
  return M

# Implemented from transformations.py by Christoph Gohlke at UCI
def euler_from_matrix(matrix, axes='sxyz'):
  firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]

  i = firstaxis
  j = _NEXT_AXIS[i+parity]
  k = _NEXT_AXIS[i-parity+1]

  M = matrix
  if repetition:
    sy = math.sqrt(M[i][j]*M[i][j] + M[i][k]*M[i][k])
    if sy > _EPS:
      ax = math.atan2( M[i][j],  M[i][k])
      ay = math.atan2( sy,       M[i][i])
      az = math.atan2( M[j][i], -M[k][i])
    else:
      ax = math.atan2(-M[j][k],  M[j][j])
      ay = math.atan2( sy,       M[i][i])
      az = 0.0
  else:
    cy = math.sqrt(M[i][i]*M[i][i] + M[j][i]*M[j][i])
    if cy > _EPS:
      ax = math.atan2( M[k][j],  M[k][k])
      ay = math.atan2(-M[k][i],  cy)
      az = math.atan2( M[j][i],  M[i][i])
    else:
      ax = math.atan2(-M[j][k],  M[j][j])
      ay = math.atan2(-M[k][i],  cy)
      az = 0.0

  if parity:
      ax, ay, az = -ax, -ay, -az
  if frame:
      ax, az = az, ax
  return ax, ay, az


################### Material Functions ########################
def CubeName(cubeNum):
  if cubeNum <= 0:
    return "Cube"
  else:
    return "Cube.{:03d}".format(cubeNum)

def MaterialName(cubeNum):
  return "Material.{:03d}".format(cubeNum+1)

def BlockSize():
  return min([max([(rand.random()+1)*0.1, rand.expovariate(1.0/1.5)]),8])

def CubeLocation(width):
  return (2*rand.random()-1)*width

################# Scene generator functions #################
def MemoryCleanUp():
  # Clear data from previous scenes
  for material in bpy.data.materials:
    material.user_clear()
    bpy.data.materials.remove(material)

  for texture in bpy.data.textures:
    texture.user_clear()
    bpy.data.textures.remove(texture)

  # Get the tree
  bpy.context.scene.use_nodes = True
  tree = bpy.context.scene.node_tree
  links = tree.links

  # Clear default nodes
  for n in tree.nodes:
    tree.nodes.remove(n)
  
  # Remove objects from previsous scenes
  for item in bpy.data.objects:
    if item.type == "MESH":
      bpy.data.objects[item.name].select = True
      debugprint("bpy.ops.object.delete()")
      bpy.ops.object.delete()

  for item in bpy.data.meshes:
    bpy.data.meshes.remove(item)

def LightingSetup(light_energy):
  # Setup lighting
  light = bpy.data.objects['Lamp']
  light.data.energy = light_energy
  light.select = False

def CameraIntrinsics(params):
  # Setup intrinsics scaling
  lens_scale = 1
  width_trans = 0
  height_trans = 0
  if params['intrinsics_random']:
    lens_scale_range = params['lens_max_scale']-params['lens_min_scale']
    lens_scale = (lens_scale_range)*rand.random() + params['lens_min_scale']
    width_trans_range = params['width_max_trans']-params['width_min_trans']
    width_trans = width_trans_range*(rand.random()**2) + params['width_min_trans']
    height_trans_range = params['height_max_trans']-params['height_min_trans']
    height_trans = height_trans_range*(rand.random()**2) + params['height_min_trans']

  # Setup camera
  camera = bpy.data.objects['Camera']
  camera.select = True
  # Camera intrinsics information
  camera.data.lens = lens_scale*params['lens'] # for the focal length
  camera.data.shift_x = width_trans
  camera.data.shift_y = height_trans
  # Sensor size
  camera.data.sensor_width = params['sensor_width']
  camera.data.sensor_height = params['sensor_height']
  # Sensor size fit (fits ratio)
  camera.data.sensor_fit = params['sensor_fit']
  camera.select = False

  return lens_scale, width_trans, height_trans

# Definitely hacky
def CameraPosition(params):
  ax = (80.0*math.pi/180.0)*rand.random()
  ay = (10.0*math.pi/180.0)*rand.gauss(0.0,1.0)
  az = math.pi*rand.random()

  camx = 0.75*rand.random()
  camy = 0.75*rand.random()
  camz = 0.5 + 10.0*rand.random()
  rot = euler_matrix(ax, ay, az)
  return (camx, camy, camz), euler_matrix(ax, ay, az)

def SetCameraPosition(cam_pos, cam_rot):
  # Setup camera rotation
  camera = bpy.data.objects['Camera']
  camera.select = True
  camera.rotation_mode = 'XYZ'
  camera.rotation_euler = euler_from_matrix(cam_rot)
  camera.location = cam_pos
  camera.select = False

def CameraPositionSecond(params):
  # Setup camera rotation
  # TODO: How to parameterize this better?
  camera.select = True
  camera.rotation_mode = 'XYZ'
  (angle1, angle2, angle3) = camera.rotation_euler
  (Cam_x,Cam_y,Cam_z) = camera.location
  camera.select = False

################ Scene Structure Functions ###################
def CreateGroundPlane(params, all_materials):
  bpy.ops.mesh.primitive_plane_add(location=(0,0,0))
  ground_width = params['ground_width']
  bpy.ops.transform.resize(value=(ground_width, ground_width, ground_width))
  bpy.ops.material.new()
  ground_mat = bpy.data.materials[params['ground_plane_mat_name']]
  all_materials.append(ground_mat)
  ground_mat.diffuse_color = (1.0, 1.0, 1.0)

  bpy.data.objects[params['ground_plane_name']].data.materials.append(ground_mat)

def CreateGroundCube(params, all_materials, cubeNum):
  # Create the cube itself
  ground_width = params['ground_width']
  bpy.ops.mesh.primitive_cube_add(
      location=(CubeLocation(ground_width), CubeLocation(ground_width), 0))
  bpy.ops.transform.resize(
      value=(BlockSize(), BlockSize(), 2*BlockSize()))
  bpy.ops.material.new()
  # Material
  cube_mat = bpy.data.materials[MaterialName(cubeNum)]
  all_materials.append(cube_mat)
  cube_mat.diffuse_color = (rand.random(), rand.random(), rand.random())

  bpy.data.objects[CubeName(cubeNum)].data.materials.append(cube_mat)

def GenerateSceneStucture(params):
  all_materials = []
  # Create objects
  CreateGroundPlane(params, all_materials)
  for i in range(params['n_cubes']):
    CreateGroundCube(params, all_materials, i)

  return all_materials

################ Generate Scene Function ###################
def GenerateScene(wdir, rngseed):
  rand.seed(rngseed)
  with open(os.path.join(wdir, 'params.yaml'), 'r') as prms:
    params = yaml.load(prms)
  MemoryCleanUp()
  LightingSetup(params['light_energy'])
  lens_scale, width_trans, height_trans = CameraIntrinsics(params)

  # Create new scene -- setup for depth map
  scene = bpy.context.scene
  tree = bpy.context.scene.node_tree
  links = tree.links
  # OK I was not familiar with this so here is the explaination
  # This is basically a graph with each node doing something
  # Render Layers (rl)
  rl = tree.nodes.new(type="CompositorNodeRLayers")
  # Compositor node - just basically
  composite = tree.nodes.new(type = "CompositorNodeComposite")
  # Relative Inverse Depth Nodes
  normalize = tree.nodes.new(type = "CompositorNodeNormalize")
  invert = tree.nodes.new(type = "CompositorNodeInvert")

  scene = bpy.context.scene

  # Creating the image outputs
  scene.render.resolution_x = params['image_width']
  scene.render.resolution_y = params['image_height']
  scene.render.resolution_percentage = params['resolution_percentage']

  # Create scene structure 
  all_materials = GenerateSceneStucture(params)

	# Generate regular images
  bpy.context.scene.render.image_settings.file_format = 'PNG'
  for i in range(len(all_materials)):
    all_materials[i].use_shadeless = False
  links.new(rl.outputs['Image'],composite.inputs['Image'])
  # Initial scene in image
  cam_pos, cam_rot = CameraPosition(params)
  SetCameraPosition(cam_pos, cam_rot)
  scene.render.filepath = os.path.join(wdir, 'Image01.png')
  bpy.ops.render.render( write_still=True )
  # Generate next image in sequence
  # Motion parameters
  foe = [ rand.gauss(0., 1.) * params['foe_sigma'] for i in range(3) ]
  omega = [ rand.gauss(0., 1.) * params['omega_sigma'] for i in range(3) ]
  # Build new position and orientation
  cam_pos2 = [ cam_pos[i] + foe[i] for i in range(len(foe)) ]
  norm_omega = math.sqrt(sum([ x**2 for x in omega ]))
  skew_omega = scale(sinc(norm_omega), skew(omega))
  skew2_omega = scale(0.5, matmul(skew_omega, skew_omega))
  rot_omega = matadd(eye(3),matadd(skew_omega, skew2_omega))
  # Render second image 
  cam_rot2 = matmul(rot_omega, cam_rot)
  SetCameraPosition(cam_pos2, cam_rot2)
  scene.render.filepath = os.path.join(wdir, 'Image02.png')
  bpy.ops.render.render( write_still=True )
  SetCameraPosition(cam_pos, cam_rot)

  # Create the labels
  for i in range(len(all_materials)):
    all_materials[i].use_shadeless = True
    (i1,i2,i3) = convert_to_base_256(i+1)
    all_materials[i].diffuse_color = (i1/255.0, i2/255.0, i3/255.0)
  links.new(rl.outputs['Image'],composite.inputs['Image'])
  scene.render.filepath = os.path.join(wdir, 'Labels.png')
  bpy.ops.render.render( write_still=True )

  # Normalized Depth
  links.new(rl.outputs['Z'], invert.inputs['Color'])
  links.new(invert.outputs['Color'], normalize.inputs['Value'])
  scene.render.filepath = os.path.join(wdir,'InvDepth.png')
  links.new(normalize.outputs['Value'], composite.inputs['Image'])
  bpy.ops.render.render( write_still=True )

  # Create Depth
  # True Depth
  bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
  scene.render.filepath = wdir
  viewer = tree.nodes.new('CompositorNodeOutputFile')
  viewer.file_slots[0].path = "TrueDepth"
  links.new(rl.outputs['Z'], viewer.inputs['Image'])
  bpy.ops.render.render( write_still=True )

  # Save out non-image related things
  scaled_lens = lens_scale*params['lens']
  sio.savemat(os.path.join(wdir, 'misc.mat'), {
      "focal_x": scaled_lens*(params['image_width']/params['sensor_width']),
      "focal_y": scaled_lens*(params['image_height']/params['sensor_height']),
      "center_x": params['image_width']*(1+width_trans)/2.0,
      "center_y": params['image_height']*(1+height_trans)/2.0,
      "lens_scale": lens_scale,
      "height_trans": height_trans,
      "width_trans": width_trans,
      "foe": foe,
      "omega": omega,
	})

if __name__ == '__main__':
  # Get Blender Script Arguments
  argv = sys.argv
  argv = argv[argv.index("--") + 1:]  # get all args after "--"
  wdir = argv[0]
  rngseed = int(argv[1])
  # Go crazy and generate scene
  GenerateScene(wdir, rngseed)



