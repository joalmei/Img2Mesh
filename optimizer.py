import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import time
import random
import matplotlib.pyplot as plt

from operator import itemgetter 
from losses import chamfer_loss
from tools.optim_tools import shuffle_tensors


# ==============================================================================
# custom optimizer for chamfer_loss
# using Adam optimization algorithm
class Optimizer:
    def __init__ (self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # ==========================================================================
    def loss (self, xs, ys):
        out = []
        ys_ = self.model(xs)
        for y, y_ in zip(ys, ys_):
            out.append(chamfer_loss(y, y_))
        return tf.stack(out)

    # =========================================================================
    def predict (self, X):
        return self.model(X)
    
    # =========================================================================
    def test (self, X, Y):
        return self.loss(X, Y)

    # =========================================================================
    def grad (self, xs, ys):
        with tf.GradientTape() as tape:
            loss_value = self.loss(xs, ys)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
    
    # =========================================================================
    # trains the network for num_epochs epochs or until min_step is achieved
    def train_epochs (self, X, Y, batches,
                        num_epochs=2000, min_error=1e-3, min_step=1e-9,
                        checkpoint_callback=None, check_step=1):
        train_loss_results = []

        for epoch in range(num_epochs):
            loss_value, nbatch, prevbatch, start_time = 0, 0, 0, time.time()
            X, Y = shuffle_tensors(X, Y)

            print("batches: ", end='')
            for batch in batches:
                nbatch = nbatch + 1
                if (int(10*nbatch/len(batches)) > prevbatch):
                    prevbatch = int(10*nbatch/len(batches))
                    print('.', end='')
                
                # optimize the model
                lossv, grads = self.grad(X[batch], Y[batch])
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # end epoch
                loss_value = K.mean(lossv) + loss_value

            loss_value = loss_value / len(batches)
            train_loss_results.append(loss_value)

            print(' ', end=' ')

            if (epoch % check_step == 0):
                print("epoch : ", epoch,
                        " ; loss = ", float(loss_value),
                        " (", time.time() - start_time ,"secs)")
                if (checkpoint_callback != None):
                    checkpoint_callback(self.model)
            
            if (epoch % check_step == 0 and epoch > 1 and loss_value < min_error):
                print('min_error achieved at ', float(loss_value))
                return train_loss_results, True
            

        return train_loss_results, False
    
    # =========================================================================
    # trains the network max_repet times for num_epochs
    def train (self, X, Y, batches,
                min_error=1e-3, min_step=1e-3, plot=False,
                checkpoint_callback=None,
                num_epochs=10,
                max_repets=10):
        losses = []
        interrupt = False
        repet = 0
        while (not interrupt):
            repet = repet + 1
            if (repet > max_repets):
                break
            print("========================================================================")
            loss, interrupt = self.train_epochs(X, Y, batches,
                                                num_epochs=num_epochs,
                                                min_error=min_error,
                                                min_step=min_step,
                                                checkpoint_callback=checkpoint_callback)
            losses.extend(loss)
            if (plot == True):
                plt.plot(losses)
                plt.plot(loss, label=str(repet))
                plt.show()
        print("Trainning finished!")
        if (plot == True):
            plt.plot(losses)
            plt.show()
        return losses