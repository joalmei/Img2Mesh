import os
import glob
import numpy as np
from sklearn.decomposition import PCA

def fetchData (path):
    X = []
    Y = []

    files = glob.glob(os.path.join(path, '*.npy'))
    for file in files:
        data = np.load(file, allow_pickle=True).item()
        
        X.append([
            data['top'],    data['bottom'],
            data['front'],  data['back'],
            data['left'],   data['right']])

        Y.append(np.array(data['vertices'], dtype='float32'))

    return X, Y

def prepareData (path):
    X, Y = fetchData(path)
    for i in range(len(X)):
        for j in range(len(X[i])):
            idx = X[i][j] > 0
            X[i][j][idx] = 1
    return X, Y

#  {'vertices': verts, 'faces': faces,
#   'top': top, 'bottom': bottom,
#   'front': front, 'back': back,
#   'left': left, 'right': right};
