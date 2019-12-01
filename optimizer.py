import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import time
import random
import matplotlib.pyplot as plt

from operator import itemgetter 
from losses import chamfer_loss


class Optimizer:
    def __init__ (self, model, learning_rate=0.001):
        self.model = model    
        #self.optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

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
            #random.shuffle(batches)
            
            #SHUFFLES THE DATASET
            indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            X = tf.gather(X, shuffled_indices)
            Y = list(itemgetter(*shuffled_indices)(Y)) 

            for batch in batches:
                nbatch = nbatch + 1
                if (int(10*nbatch/len(batches)) > prevbatch):
                    prevbatch = int(10*nbatch/len(batches))
                    print('.', end='')
                    #print(prevbatch * 10, end=', ')
                # Optimize the model
                lossv, grads = self.grad(X[batch], Y[batch])
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # End epoch
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
            
            # if (epoch % check_step == 0 and 
            #     epoch > 1 and
            #     ((train_loss_results[-check_step-1] - loss_value < minStep) or
            #     loss_value < minError)):
            #     print(int(train_loss_results[-check_step-1]) , ' - ', int(loss_value), ' < ', minStep)
            #     return train_loss_results, True
            if (epoch % check_step == 0 and epoch > 1 and loss_value < minError):
                print('minError achieved at ', float(loss_value))
                return train_loss_results, True
            

        return train_loss_results, False
    
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
                                                minError=min_error,
                                                minStep=min_step,
                                                checkpoint_callback=checkpoint_callback)
            if (repet > 1):
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
    
    def predict (self, X):
        return self.model(X)
    
    def test (self, X, Y):
        return self.loss(X, Y)