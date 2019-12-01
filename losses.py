import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


# ==============================================================================
# compute vertex normals using keras backend primitives
# vertices : (out_verts, 3) tensor
# faces : tensor / ndarray of faces indexes (BASED IN 1 PLEASE!)
# mask : tensor / ndarray MASK (for each vertex index, the indeses of the faces it belongs)
def get_tensor_normals(vertices, faces, mask):
    v0 = tf.gather(vertices, faces[:,0] - 1)
    v1 = tf.gather(vertices, faces[:,1] - 1)
    v2 = tf.gather(vertices, faces[:,2] - 1)
    n_faces = tf.linalg.cross(v2 - v1, v0 - v1)
    n_vertex = tf.gather(n_faces, mask)             # list of face normals for each vertex
    return K.mean(n_vertex, axis=1)

# ==============================================================================
def normal_loss (ref_normals, targ_normals, closest_ref):
    # return mean(dot(ref_normals, targ_normals[closest]))
    return K.mean(K.sum(ref_normals * tf.gather(targ_normals, closest_ref), axis=1))

# ==============================================================================
# ref  : original
# targ : neural network output
def chamfer_loss(ref, targ, return_argmin=False):
    nr = ref.shape[0]
    nt = targ.shape[0]

    r = tf.tile(ref, [nt, 1])
    r = tf.reshape(r, [nt, nr, 3])

    t = tf.tile(targ, [1, nr])
    t = tf.reshape(t, [nt, nr, 3])

    dist = K.sum(K.square(r - t), axis=2)

    if (return_argmin == True):
        closest = K.argmin(dist, axis=0)
        loss = (K.mean(K.min(dist, axis=1)) + K.mean(K.min(dist, axis=0))) / 2
        return loss, closest
    else:
        return (K.mean(K.min(dist, axis=1)) + K.mean(K.min(dist, axis=0)))/2

# ==============================================================================
def complete_loss ( ref_verts, ref_normals,
                    targ_verts, targ_faces, targ_face_mask):
    ch_loss, closest_ref = chamfer_loss(ref_verts, targ_verts, True)
    targ_normals = get_tensor_normals(targ_verts, targ_faces, targ_face_mask)
    norm_loss = normal_loss(ref_normals, targ_normals, closest_ref)
    return 0.9 * ch_loss + 0.1 * norm_loss


# OBS : Info on losses
# 
# closeTarg = K.argmin(dist, axis=1)
# closeRef = K.argmin(dist, axis=0)
# !!! d(ref, targ) = sum_t(min_r (d(r, t))) + sum_r(min_t (d(r, t)))
# http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf