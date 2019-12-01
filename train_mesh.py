import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model import createNetwork, createDeconvNetwork, createLeanNetwork
from mesh_optimizer import Optimizer
from prepare_data import prepareData, createBatches, prepare_mask

import tensorflow as tf
import numpy      as np
import time

# ==============================================================================
def prepareTrainData(path = ['./data/content/objnet/airplane/test'], ratio=0):

    X, Y, faces = prepareData(path, lean=True, fetch_faces=True)
    
    if (ratio > 0 and ratio < 1):
        outX = []
        outY = []
        out_faces = []
        for i in np.random.choice(len(X), size=int(ratio*len(X)), replace=False):
            outX.append(X[i])
            outY.append(Y[i])
            out_faces.append(faces[i])
        X = outX
        Y = outY
        faces = out_faces

    X = tf.constant(X, shape=[len(X), 1, 400, 400])

    return X, Y, faces
    
# ==============================================================================
def prepareNN(hidden_size=1024, out_verts=162,
              learning_rate=0.001,
              targ_obj_path='./models/ico_162.obj',
              norm_weight=0.1):
    
    net = createLeanNetwork(hidden_size, out_verts)
    faces, mask = prepare_mask(targ_obj_path)
    optim = Optimizer(net, faces=faces, mask=mask,
                      learning_rate=learning_rate, norm_weight=0.1)

    return net, optim

# ==============================================================================
def runTraining(optim, X, Y, Y_normals,
                batch_size=16, min_error=1e-3, min_step=1e-3,
                checkpoint_callback=None,
                num_epochs=10,
                max_repets=10):
    batches = createBatches(len(X), batch_size=batch_size)
    losses = optim.train(X, Y, Y_normals, batches,
                        min_error=min_error, min_step=min_step, plot=True,
                        checkpoint_callback=checkpoint_callback,
                        num_epochs=num_epochs,
                        max_repets=max_repets)

    return losses

# ==============================================================================
def saveCheckpoint(path = './checkpoints/check', download_callback=None):
    return lambda model : { saveModel(model, path),
                            download_callback(path) }

def saveModel(model, path):
    model.save_weights(path)
    
# ==============================================================================
def updateModel(model, checkpoint_path):
    model.load_weights(checkpoint_path)