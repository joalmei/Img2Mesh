import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model import createNetwork, createDeconvNetwork, createLeanNetwork
from optimizer import Optimizer
from prepare_data import prepareData, createBatches

import tensorflow as tf
import numpy      as np
import time

# ==============================================================================
def prepareTrainData(path = ['./data/content/objnet/airplane/test'],
                        ratio=0, shape='3D', lean=True,
                        fetch_faces=False):

    if (fetch_faces == True):
        X, Y, faces = prepareData(path, lean=lean, fetch_faces=True)
    else:
        X, Y = prepareData(path, lean=lean, fetch_faces=False)

    if (ratio > 0 and ratio < 1):
        outX = []
        outY = []
        out_faces = []
        for i in np.random.choice(len(X), size=int(ratio*len(X)), replace=False):
            outX.append(X[i])
            outY.append(Y[i])
            if (fetch_faces == True):
                out_faces.append(faces[i])
        X = outX
        Y = outY
        faces = out_faces

    n_channels = 6
    if (lean == True):
        n_channels = 1

    if (shape == '3D'):
        X = tf.constant(X, shape=[len(X), n_channels, 400, 400, 1])
    else:
        X = tf.constant(X, shape=[len(X), n_channels, 400, 400])

    if (fetch_faces==True):
        return X, Y, faces
    else:
        return X, Y

# ==============================================================================
def prepareNN(model='classic', learning_rate=0.001, hidden_size=1024, out_verts=32):
    if (model == 'classic'):
        net = createNetwork(hidden_size, out_verts)
    elif (model == 'lean'):
        net = createLeanNetwork(hidden_size, out_verts)
    else:
        net = createDeconvNetwork()
    optim = Optimizer(net, learning_rate=learning_rate)

    return net, optim

# ==============================================================================
def runTraining(optim, X, Y,
                batch_size=16, min_error=1e-3, min_step=1e-3,
                checkpoint_callback=None,
                num_epochs=10,
                max_repets=10):
    batches = createBatches(len(X), batch_size=batch_size)
    losses = optim.train(X, Y, batches,
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