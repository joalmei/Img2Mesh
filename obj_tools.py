import os
import numpy as np

def read_off(filepath):
  with open(filepath, 'r') as file:
    if 'OFF' != file.readline().strip():
        raise Exception ('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def write_obj(vertices, faces, objpath):
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
  
  write_obj(verts, faces, outpath)
  return verts, faces

def parse_vertex(data):
  return [float(data[1]), float(data[2]), float(data[3])]

def parse_face(data):
  return [int(data[1].split('/')[0]),
          int(data[2].split('/')[0]),
          int(data[3].split('/')[0])]

def parse_obj(obj, get_faces = False):
  vertices = []
  faces = []

  for line in obj.split('\n'):
    data = line.split()

    if (len(data) == 0):
      continue

    if (data[0] == 'v'):
      vertices.append(parse_vertex(data))
    elif (get_faces == True and data[0] == 'f'):
      faces.append(parse_face(data))

  vertices = np.array(vertices)
  faces = np.array(faces)
      
  if (get_faces == True):
    return vertices, faces
  
  return vertices

def read_obj(obj, get_faces=True):
  # 3D OBJ PARSING
  with open(obj, 'r') as file:
    content = file.read()
    return parse_obj(content, get_faces = get_faces) #read obj