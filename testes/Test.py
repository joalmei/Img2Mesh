from MLPRegressor import MLP
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import TestData as td

tf.executing_eagerly()

print(td.tensor_X)
print(td.tensor_Y)

max_hidden = 25
losses = []
hidden = range(2, max_hidden)
for h in hidden:
  print("hidden: ", h)
  mlp = MLP(1, h, 8*3)
  l = mlp.train(td.tensor_X, td.tensor_Y)
  print(l)
  losses.append(l[-1])
print(hidden)
print(losses)

plt.plot(hidden, losses)
plt.show()