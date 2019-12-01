from obj_tools import read_obj

def write_objs (list_vertices, list_faces, obj_path='/content/objtest.obj'):
  with open(obj_path, 'w+') as file:
    pad = 0
    for i in range(len(list_vertices)):
      vertices = list_vertices[i]
      faces = list_faces[i] + pad
      pad = pad + len(list_vertices[i])

      file.write("o test" + str(i) + "\n")
      file.write("# VERTICES\n")
      for v in vertices:
        vert = "v %f %f %f" % (v[0], v[1] , v[2])
        file.write(vert + "\n")
          
      file.write("# FACES\n")
      for f in faces:
        face = "f %d %d %d" % (f[0], f[1] , f[2])
        file.write(face + "\n")

def write_point_cloud (points, obj_path='/content/pointstest.obj'):
  ico_v, ico_f = read_obj('/content/Img2Mesh/models/ico.obj')
  verts, faces = [], []
  for p in points:
    verts.append(ico_v + p)
    faces.append(ico_f)
  write_objs(verts, faces, obj_path)


def write_point_cloud_with_reference (points,
                                      ref_verts, ref_faces,
                                      obj_path='/content/pointstest.obj'):
  ico_v, ico_f = read_obj('/content/Img2Mesh/models/ico.obj')
  verts, faces = [], []
  for p in points:
    verts.append(ico_v + p)
    faces.append(ico_f)
  verts.append(ref_verts)
  faces.append(ref_faces)
  write_objs(verts, faces, obj_path)