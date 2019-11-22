from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers


def chamfer_loss(ref, targ):
  nr = int(ref.shape[0] / 3)
  nt = int(targ.shape[0] / 3)

  ref = tf.reshape(ref, (nr, 3))
  targ = tf.reshape(targ, (nt, 3))

  r = tf.tile(ref, [nt, 1])
  r = tf.reshape(r, (nt, nr, 3))

  t = tf.tile(targ, [1, nr])
  t = tf.reshape(t, [nt, nr, 3])

  dist = K.sum(K.square(r - t), axis=2)
  #closeTarg = K.argmin(dist, axis=0)
  return K.mean(K.min(dist, axis=0))

class MLP:
  def __init__ (self, insize=1, hiddensize=10, outsize=8*3, learning_rate=0.001):
    K.set_floatx('float64')
    self.model = keras.Sequential([
                          layers.Dense(hiddensize, activation='relu', input_shape=[insize]),
                          layers.Dense(outsize)
                          ])
    
    self.optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
  
  def loss (self, xs, ys):
    out = []
    ys_ = self.model(xs)
    for y, y_ in zip(ys, ys_):
      out.append(chamfer_loss(y, y_))
    return tf.stack(out)

  def grad (self, xs, ys):
    with tf.GradientTape() as tape:
      loss_value = self.loss(xs, ys)
    return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
  
  def train_epochs (self, X, Y, num_epochs=2000, minError=1e-3, minStep=1e-9):
    train_loss_results = []
    
    for epoch in range(num_epochs):
      # Optimize the model
      loss_value, grads = self.grad(X, Y)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

      # End epoch
      loss_value = K.mean(loss_value)
      train_loss_results.append(loss_value)

      if epoch % 200 == 0:
        if (epoch > 1 and (train_loss_results[-200] - loss_value < minStep) or loss_value < minError):
          return train_loss_results, True
    
    return train_loss_results, False
  
  def train (self, X, Y, minError=1e-3):
    losses = []
    interrupt = False
    while (not interrupt):
      loss, interrupt = self.train_epochs(X, Y, num_epochs=2000, minError=minError, minStep=minError*minError)
      losses.extend(loss)
    return losses
  
  def predict (self, X):
    return self.model(X)
  
  def test (self, X, Y):
    return self.loss(X, Y)