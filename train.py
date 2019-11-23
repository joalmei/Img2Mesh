import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model import createNetwork
from optimizer import Optimizer
from prepare_data import prepareData, createBatches

import tensorflow as tf
import numpy        as np

def prepareTrain(path = './data/content/objnet/airplane/test'):
    net = createNetwork(100, 50)
    optim = Optimizer(net)

    X, Y = prepareData(path)
    X = tf.constant(X, shape=[len(X), 6, 400, 400, 1])
    return net, optim, X, Y

def runTrainning(optim, X, Y, batch_size=16, min_error=1e-3):
    batches = createBatches(len(X), batch_size=batch_size)
    losses = optim.train(X, Y, batches, min_error=min_error, plot=True)

    return losses