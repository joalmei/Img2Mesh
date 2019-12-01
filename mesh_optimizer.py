import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import time
import matplotlib.pyplot as plt

from operator import itemgetter 
from losses import complete_loss
from optim_tools import shuffle_tensors


class Optimizer:
    # model output, faces and mask are SUPER TANGGLED!!!!!
    # TODO: untagle

    def __init__ (self, model, faces, mask, learning_rate=0.001):
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.faces = faces
        self.mask = mask

    def loss (self, xs, ys, ys_normals):
        out = []
        targ_verts = self.model(xs)

        for ref, ref_normals, targ in zip(ys, ys_normals, targ_verts):
            out.append(complete_loss(
                                        ref, ref_normals,
                                        targ, self.faces, self.mask
                                        ))
        return tf.stack(out)

    def grad (self, xs, ys, ys_normals):
        with tf.GradientTape() as tape:
            loss_value = self.loss(xs, ys, ys_normals)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
    
    def train_epochs (self, X, Y, Y_normals, in_batches,
                        num_epochs=2000, minError=1e-3, minStep=1e-9,
                        checkpoint_callback=None, check_step=1):
        train_loss_results = []
        batches = in_batches

        for epoch in range(num_epochs):
            loss_value, nbatch, prevbatch, start_time = 0, 0, 0, time.time()
            X, Y, Y_normals = shuffle_tensors(X, Y, Y_normals)

            print("batches: ", end='')
            for batch in batches:
                nbatch = nbatch + 1
                if (int(10*nbatch/len(batches)) > prevbatch):
                    prevbatch = int(10*nbatch/len(batches))
                    print('.', end='')
                    
                lossv, grads = self.grad(X[batch], Y[batch], Y_normals[batch])
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
    
    def train (self, X, Y, Y_normals, batches,
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
            loss, interrupt = self.train_epochs(X, Y, Y_normals, batches,
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
    
    def test (self, X, Y, Y_normals):
        return self.loss(X, Y, Y_normals)