import os
import numpy as np

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