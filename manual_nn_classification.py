import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import manual_nn


data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
features = data[0]
plt.scatter(features[:, 0], features[:, 1])
plt.show()

labels = data[1]
plt.scatter(features[:,0],features[:,1],c=labels,cmap='coolwarm')

x = np.linspace(0,11,10)
y = -x + 5
plt.scatter(features[:,0],features[:,1],c=labels,cmap='coolwarm')
plt.plot(x,y)
plt.show()

np.array([1, 1]).dot(np.array([[8],[10]])) - 5