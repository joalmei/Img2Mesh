from tensorflow.keras import models, layers
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.constraints import MaxNorm


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
    model.add(layers.Dense(3 * out_vertices))
    model.add(layers.Reshape((out_vertices, 3)))

    return model

def createLeanNetwork(hidden_size, out_vertices):
    #K.set_floatx('float32')

    # model = models.Sequential()
    # model.add(layers.MaxPooling2D((4, 4), input_shape=(6, 400, 400),
    #                                 data_format='channels_first'))

    # model.add(layers.Conv2D(64, (5,5),
    #                         data_format='channels_first'))

    # model.add(layers.MaxPooling2D((3, 3),
    #                                 data_format='channels_first'))

    # model.add(layers.Conv2D(128, (3,3),
    #                         data_format='channels_first'))

    # model.add(layers.MaxPooling2D((3, 3),
    #                                 data_format='channels_first'))

    # model.add(layers.Conv2D(256, (3,3),
    #                         data_format='channels_first'))

    # model.add(layers.Flatten())

    # model.add(layers.Dropout(0.1))

    # model.add(layers.Dense(hidden_size, activation='relu'))

    # model.add(layers.Dropout(0.1))
    # model.add(layers.Dense(3 * out_vertices, activation='tanh'))
    # model.add(layers.Reshape((out_vertices, 3)))
    
    model = models.Sequential()
    model.add(layers.MaxPooling2D((8, 8), input_shape=(6, 400, 400),
                                    data_format='channels_first'))

    model.add(layers.Conv2D(64, (3,3),
                            data_format='channels_first'))

    model.add(layers.MaxPooling2D((8, 8),
                                    data_format='channels_first'))

    model.add(layers.Conv2D(128, (3,3),
                            data_format='channels_first'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(hidden_size, activation='relu'))

    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(3 * out_vertices, activation='tanh'))
    model.add(layers.Reshape((out_vertices, 3)))

    return model


# OUTPUT ALWAYS (50, 3)
def createDeconvNetwork():
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
    model.add(layers.Conv3D(1024, (1,4,4)))
    model.add(layers.MaxPooling3D((1, 2, 2)))

    model.add(layers.Reshape((8, 8, 1024)))

    model.add(layers.Conv2D(1024, (3,3)))
    model.add(layers.Conv2DTranspose(3, (15,15)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(3, (3,3)))
    model.add(layers.Reshape((8 * 8, 3)))
    model.add(layers.Conv1D(3, (15)))

    return model
