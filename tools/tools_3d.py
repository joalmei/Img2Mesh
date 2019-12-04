import numpy as np

def face_normal(face, pos):
  e1 = pos[face[1] - 1] - pos[face[0] - 1]
  e2 = pos[face[1] - 1] - pos[face[2] - 1]

  dir = np.cross(e1, e2)
  norm = np.linalg.norm(dir)
  if norm > 0:
    dir = dir / norm

  return dir

# positions : positions of the original obj
# faces : faces (list of vertices) of the original obj
# sampled_verts : sampled vertices
def obj_normals(pos, faces, sampled_verts):
  # prepare normals dict
  normals = {}
  sampled_verts = sampled_verts + 1       # faces indexing uses vertices in 1 ... n
  vertices = set(sampled_verts)           # set for efficiency

  for v in vertices:
    normals[v] = list()

  # compute vertex faces
  for f in faces:
    vs = vertices.intersection(f)
    if (len(vs) > 0):
      n = face_normal(f, pos)
      for v in vs:
        normals[v].append(n)

  
  # remove vertices without face
  out_verts = []
  out_normals = []
  for v in sampled_verts:
    if (len(normals[v]) > 0):
      out_normals.append(np.mean(normals[v], axis=0))
      out_verts.append(v)
  
  out_verts = np.array(out_verts) - 1
  out_normals = np.array(out_normals)
  return out_verts, out_normals

