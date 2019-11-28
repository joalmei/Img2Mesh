import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import time
import random
import numpy as np

import matplotlib.pyplot as plt

def chamfer_loss(ref, targ):
    nr = ref.shape[0]
    nt = targ.shape[0]

    #ref = tf.reshape(ref, (nr, 3))
    #targ = tf.reshape(targ, (nt, 3))

    r = tf.tile(ref, [nt, 1])
    r = tf.reshape(r, [nt, nr, 3])

    t = tf.tile(targ, [1, nr])
    t = tf.reshape(t, [nt, nr, 3])

    dist = K.sum(K.square(r - t), axis=2)

    #closeTarg = K.argmin(dist, axis=1)
    #closeRef = K.argmin(dist, axis=0)
    # !!! d(ref, targ) = sum_t(min_r (d(r, t))) + sum_r(min_t (d(r, t)))
    # http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf
    return (K.mean(K.min(dist, axis=1)) + K.mean(K.min(dist, axis=0)))/2

class Optimizer:
    def __init__ (self, model, learning_rate=0.001):
        self.model = model    
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
    
    def train_epochs (self, X, Y, in_batches,
                        num_epochs=2000, minError=1e-3, minStep=1e-9,
                        checkpoint_callback=None, check_step=1):
        train_loss_results = []
        batches = in_batches

        for epoch in range(num_epochs):
            start_time = time.time()

            loss_value = 0
            nbatch = 0
            prevbatch = 0
            print("batches: ", end='')
            random.shuffle(batches)
            for batch in batches:
                nbatch = nbatch + 1
                if (int(10*nbatch/len(batches)) > prevbatch):
                    prevbatch = int(10*nbatch/len(batches))
                    print(prevbatch * 10, end=', ')
                # Optimize the model
                lossv, grads = self.grad(X[batch], Y[batch])
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # End epoch
                loss_value = K.mean(lossv) + loss_value

            loss_value = loss_value / len(batches)
            train_loss_results.append(loss_value)

            print('')

            if (epoch % check_step == 0):
                print("epoch : ", epoch,
                        " ; loss = ", loss_value,
                        " (", time.time() - start_time ,"secs)")
                if (checkpoint_callback != None):
                    checkpoint_callback(self.model)
            
            if (epoch % check_step == 0 and 
                epoch > 1 and
                ((np.absolute(train_loss_results[-check_step-1] - loss_value) < minStep) or
                loss_value < minError)):
                print(train_loss_results[-check_step-1] , ' - ', loss_value, ' < ', minStep)
                return train_loss_results, True
            
        return train_loss_results, False
    
    def train (self, X, Y, batches,
                min_error=1e-3, min_step=1e-3, plot=False,
                checkpoint_callback=None):
        losses = []
        interrupt = False
        while (not interrupt):
            print("========================================================================")
            loss, interrupt = self.train_epochs(X, Y, batches,
                                                num_epochs=10,
                                                minError=min_error,
                                                minStep=min_step,
                                                checkpoint_callback=checkpoint_callback)
            losses.extend(loss)
            if (plot == True):
                plt.plot(loss)
                plt.show()
        return losses
    
    def predict (self, X):
        return self.model(X)
    
    def test (self, X, Y):
        return self.loss(X, Y)