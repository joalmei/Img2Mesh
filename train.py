import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model import createNetwork, createDeconvNetwork
from optimizer import Optimizer
from prepare_data import prepareData, createBatches

import tensorflow as tf
import numpy      as np
import time

# ==============================================================================
def prepareTrain(path = './data/content/objnet/airplane/test',
                    model='classic',
                    learning_rate=0.001):

    if (model == 'classic'):
        net = createNetwork(100, 50)
    else:
        net = createDeconvNetwork(100)
    optim = Optimizer(net, learning_rate=learning_rate)

    X, Y = prepareData(path)
    X = tf.constant(X, shape=[len(X), 6, 400, 400, 1])

    return net, optim, X, Y

# ==============================================================================
def runTraining(optim, X, Y,
                batch_size=16, min_error=1e-3, min_step=1e-3,
                checkpoint_callback=None):
    batches = createBatches(len(X), batch_size=batch_size)
    losses = optim.train(X, Y, batches,
                        min_error=min_error, min_step=min_step, plot=True,
                        checkpoint_callback=checkpoint_callback)

    return losses

# ==============================================================================
def saveCheckpoint(path = './checkpoints/check', download_callback=None):
    return lambda model : { saveModel(model, path),
                            download_callback(path) }

def saveModel(model, path):
    start = time.time()
    print("Saving...")
    model.save_weights(path)
    print("Saved in ", time.time() - start, " secs!")

# def download(path):
#    from google.colab import files
#    zip_path = path+'/checkpoint.zip'
#    !zip -r {zip_path} {path+"/*"},
#    files.download(zip_path) }
    
# ==============================================================================
def updateModel(model, checkpoint_path):
    model.load_weights(checkpoint_path)