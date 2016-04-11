from openml.apiconnector import APIConnector
import pandas as pd
import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_moons


# plot moons
X,Y = make_moons(n_samples=200, noise=.05)
plt.scatter(X[:,0], X[:,1])
plt.show()

#train ensemble

#plt surface


