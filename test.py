# from openml.apiconnector import APIConnector
import pandas as pd
# import os
# import numpy as np
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
# from pandas.tools.plotting import parallel_coordinates
# from sklearn.tree import DecisionTreeClassifier
#
# home_dir = os.path.expanduser("~")
# openml_dir = os.path.join(home_dir, ".openml")
# cache_dir = os.path.join(openml_dir, "cache")
# with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
#     key = fh.readline().rstrip('\n')
# openml = APIConnector(cache_directory=cache_dir, apikey=key)
# dataset = openml.download_dataset(10)
# X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
# iris = pd.DataFrame(X, columns=attribute_names)
#
# print iris[:0]

from openml.apiconnector import APIConnector

apikey = 'ca2397ea8a2cdd9707ef39d76576e786'
connector = APIConnector(apikey=apikey)
#loading data
dataset = connector.download_dataset(44)
X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)

X = pd.DataFrame(X, columns=attribute_names)
print X[:2]