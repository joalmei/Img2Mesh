# ================================================================================
# OBJ and OFF files handling
import os

def read_off(filepath):
  with open(filepath, 'r') as file:
    if 'OFF' != file.readline().strip():
        raise Exception ('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def writeObj(vertices, faces, objpath):
  with open(objpath, 'w+') as file:
    
    file.write("# VERTICES\n")
    for v in vertices:
      vert = "v %f %f %f" % (v[0], v[1] , v[2])
      file.write(vert + "\n")
    
    file.write("# FACES\n")
    for f in faces:
      face = "f %d %d %d" % (f[0], f[1] , f[2])
      file.write(face + "\n")

def off2obj(inpath, outpath):
  verts, faces = read_off(inpath)
  for i in range(len(faces)):
    for j in range(len(faces[i])):
      faces[i][j] = faces[i][j] + 1

  writeObj(verts, faces, outpath)
  return verts, faces


# ==================================================================================
# CAMERA
import numpy as np

def trans (dx, dy, dz):
  return np.array([
       [1, 0, 0, dx],
       [0, 1, 0, dy],
       [0, 0, 1, dz],
       [0, 0, 0, 1],
    ])

def rotX (dt):
  return np.array([
       [1, 0,          0,           0],
       [0, np.cos(dt), -np.sin(dt), 0],
       [0, np.sin(dt), np.cos(dt),  0],
       [0, 0,          0,           1],
    ])
    
def rotY (dt):
  return np.array([
       [np.cos(dt),  0, np.sin(dt), 0],
       [0,           1, 0,          0],
       [-np.sin(dt), 0, np.cos(dt), 0],
       [0,           0, 0,          1],
    ])


def rotZ (dt):
  return np.array([
       [np.cos(dt), -np.sin(dt), 0, 0],
       [np.sin(dt), np.cos(dt),  0, 0],
       [0,          0,           1, 0],
       [0,          0,           0, 1],
    ])

def applyTransfos(transfos, origin):
  out = origin
  for t in transfos:
    out = np.dot(t, out)
  return out


def topCamera():
  return applyTransfos([trans(0,0,5)],
                       np.eye(4))
def bottomCamera():
  return applyTransfos([rotX(np.pi)],
                       topCamera())
def frontCamera():
  return applyTransfos([rotZ(np.pi/2), rotY(np.pi/2)],
                       topCamera())
def backCamera():
  return applyTransfos([rotZ(-np.pi/2), rotY(-np.pi/2)],
                       topCamera())
def rightCamera():
  return applyTransfos([rotZ(np.pi/2)],
                       frontCamera())
def leftCamera():
  return applyTransfos([rotZ(-np.pi/2)],
                       frontCamera())


def scaleVertices (vertices, feature_range=(0, 1)):
  vertices = np.array(vertices, dtype=float)
  a = np.argmax(np.ptp(vertices, axis=0))

  min = np.amin(vertices, axis=0)
  max = np.amax(vertices, axis=0)
  scale = (max[a] - min[a])

  for i in range(len(vertices)):
    #ratio = (vertices[i] - min) / scale + (1 - (max - min)/scale)/2;
    ratio = (2 * vertices[i] - (max + min) + scale) / (2 * scale)
    vertices[i] = feature_range[1] * ratio + feature_range[0] * (1 - ratio)

  return vertices

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import trimesh
import pyrender
from pyrender.constants import RenderFlags

def render (obj, cam_pos):
  # create scene
  scene = pyrender.Scene()

  # MESHES
  obj_trimesh = trimesh.load(obj)
  obj_trimesh.vertices = scaleVertices(obj_trimesh.vertices, feature_range=(-1, 1))

  mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
  nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
  scene.add_node(nm)

  # CAMERA
  cam = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
  nc = pyrender.Node(camera=cam, matrix=cam_pos)
  scene.add_node(nc)

  # RENDER
  flags = RenderFlags.DEPTH_ONLY
  r = pyrender.OffscreenRenderer(400, 400)
  depth = r.render(scene, flags=flags)
  return (depth);

                       