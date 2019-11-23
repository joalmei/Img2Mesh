import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

def chamfer_loss(ref, targ):
    nr = ref.shape[0]
    nt = int(targ.shape[0] / 3)

    #ref = tf.reshape(ref, (nr, 3))
    targ = tf.reshape(targ, (nt, 3))

    r = tf.tile(ref, [nt, 1])
    r = tf.reshape(r, [nt, nr, 3])

    t = tf.tile(targ, [1, nr])
    t = tf.reshape(t, [nt, nr, 3])

    dist = K.sum(K.square(r - t), axis=2)

    #closeTarg = K.argmin(dist, axis=1)
    #closeRef = K.argmin(dist, axis=0)
    # !!! d(ref, targ) = sum_t(min_r (d(r, t))) + sum_r(min_t (d(r, t)))
    # http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf
    return K.sum(K.min(dist, axis=1)) + K.sum(K.min(dist, axis=0))

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
    
    def train_epochs (self, X, Y, batches, num_epochs=2000, minError=1e-3, minStep=1e-9):
        train_loss_results = []
        
        for epoch in range(num_epochs):
            loss_value = 0
            for batch in batches:
                # Optimize the model
                lossv, grads = self.grad(X[batch], Y[batch])
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # End epoch
                loss_value = K.mean(lossv) + loss_value

            loss_value = loss_value / len(batches)
            train_loss_results.append(loss_value)

            if (epoch % 5 == 0):
                print("-> epoch : ", epoch, " ; loss = ", loss_value)
            
            if (epoch % 5 == 0):
                if (epoch > 1 and
                    (train_loss_results[-200] - loss_value < minStep) or
                    loss_value < minError):
                    return train_loss_results, True
            
        return train_loss_results, False
    
    def train (self, X, Y, batches, min_error=1e-3, min_step=1e-3, plot=False):
        losses = []
        interrupt = False
        while (not interrupt):
            print("========================================================================")
            loss, interrupt = self.train_epochs(X, Y, batches,
                                                num_epochs=2000,
                                                minError=min_error,
                                                minStep=min_step)
            losses.extend(loss)
            if (plot == True):
                plt.plot(losses)
                plt.show()
        return losses
    
    def predict (self, X):
        return self.model(X)
    
    def test (self, X, Y):
        return self.loss(X, Y)