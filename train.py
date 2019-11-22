import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from model import createNetwork
from optimizer import Optimizer
from prepare_data import X, Y

import tensorflow as tf
import numpy        as np

net = createNetwork(100, 50)
optim = Optimizer(net)

X = tf.constant(X, shape=[len(X), 6, 400, 400, 1])
