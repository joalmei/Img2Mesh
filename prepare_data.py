import os
import glob
import numpy as np
from random import sample

def fetchData (path, lean=True):
    X = []
    Y = []

    files = glob.glob(os.path.join(path, '*.npy'))
    for file in files:
        data = np.load(file, allow_pickle=True).item()
        
        if (lean == True):
            X.append([data['left']])
        else:
            X.append([
                data['top'],    data['bottom'],
                data['front'],  data['back'],
                data['left'],   data['right']])

        Y.append(np.array(data['vertices'], dtype='float32'))

    return X, Y

def prepareData (paths, lean=True):
    #X, Y = fetchData(path)
    #for i in range(len(X)):
    #    for j in range(len(X[i])):
    #        idx = X[i][j] > 0
    #        X[i][j][idx] = 1
    
    #for i in range(len(Y)):
    #    Y[i] = (Y[i] + 1)/2

    X = []
    Y = []
    for p in paths:
        x_p, y_p = fetchData(p, lean=lean)
        X.extend(x_p)
        Y.extend(y_p)

    return X, Y

def createBatches (length, batch_size=16):
    batches = []

    offset = 0
    while offset < length - batch_size:
        batches.append(slice(offset, offset+batch_size))
        offset = offset + batch_size
    
    batches.append(slice(offset, None))

    return batches

def downsample(data, k=50000):
    out = []
    for d in data:
        if len(d) > k:
            idx = np.random.choice(len(d), size=k, replace=False)
            out.append(d[idx])
        else:
            out.append(d)
    return out

#  {'vertices': verts, 'faces': faces,
#   'top': top, 'bottom': bottom,
#   'front': front, 'back': back,
#   'left': left, 'right': right};
