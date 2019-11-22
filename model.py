from tensorflow.keras import models, layers
import tensorflow.keras.backend as K

def createNetwork(hidden_size, out_vertices):
    #K.set_floatx('float32')

    model = models.Sequential()
    model.add(layers.Conv3D(64, (2,3,3), input_shape=(6, 400, 400, 1)))
    model.add(layers.Conv3D(64, (1,3,3)))
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.Conv3D(128, (2,3,3)))
    model.add(layers.Conv3D(128, (1,3,3)))
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.Conv3D(256, (2,3,3)))
    model.add(layers.Conv3D(256, (1,4,4)))
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.Conv3D(512, (2,3,3)))
    model.add(layers.Conv3D(512, (1,3,3)))
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.Conv3D(1024, (2,3,3)))
    model.add(layers.Conv3D(1024, (1,3,3)))

    model.add(layers.Flatten())
    model.add(layers.Dense(hidden_size, activation='relu'))
    model.add(layers.Dense(3 * out_vertices, activation='relu'))

    return model