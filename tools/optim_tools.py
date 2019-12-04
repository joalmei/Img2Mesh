import tensorflow as tf
from operator import itemgetter 


def shuffle_tensors (X, Y, Y_normals=None):
    indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    X = tf.gather(X, shuffled_indices)
    Y = list(itemgetter(*shuffled_indices)(Y))
    if (Y_normals == None):
        return X, Y

    Y_normals = list(itemgetter(*shuffled_indices)(Y_normals))
    return X, Y, Y_normals