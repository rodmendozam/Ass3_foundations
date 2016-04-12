import openml
from openml.apiconnector import APIConnector
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import zero_one_loss

#getting the dataset
apikey = 'cdce7558c3301b97a65c88401775a6c4'
connector = APIConnector(apikey=apikey)
dataset = connector.download_dataset(554)
X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)

#structuring the dataFrame (784 columns named "pixel1",.."pixel784" then a column called "classes" with the digit)
data = pd.DataFrame(X, columns=attribute_names)
data['classes'] = y

#splitting the data in training and test sets
trainingData = data.head(60000)
X_train = trainingData.values[:,:784]
y_train = trainingData.values[:,784:]

testData = data.tail(10000)
X_test = testData.values[:,:784]
y_test = testData.values[:,784:]

#training the classifier - 1 hidden layer with 100 units... in the classifier there are all the parameters to modify (as described in the assignment text)
clf = MLPClassifier(hidden_layer_sizes=(100,),algorithm='adam',batch_size='auto',learning_rate='constant',max_iter=200,momentum=0.9,alpha=0.0001)
clf.fit(X_train, y_train)

#evaluation
classifError_percent = (zero_one_loss(y_test, clf.predict(X_test)))*100
print("{} {}".format(classifError_percent,"%"))