from openml.apiconnector import APIConnector
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#connect to openml api
apikey = 'ca2397ea8a2cdd9707ef39d76576e786'
connector = APIConnector(apikey=apikey)
dataset = connector.download_dataset(44)
X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
X = pd.DataFrame(X, columns=attribute_names)
# print X[:2]

#train the classifier
clf = RandomForestClassifier(n_estimators= 32, warm_start=True, oob_score=True,
                               max_features="auto",
                               random_state=0)


clf = clf.fit(X, y)