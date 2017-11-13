# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import sys
import os
import argparse
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

from azureml.dataprep.package import run
from azureml.logging import get_azureml_logger

# initialize the logger
logger = get_azureml_logger()

# add experiment arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--arg', action='store_true', help='My Arg')
args = parser.parse_args()
print(args)

# This is how you log scalar metrics
# logger.log("MyMetric", value)

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)

print('Python version: {}'.format(sys.version))
print()

# load Iris dataset from a DataPrep package as a pandas DataFrame
titanic_dataset = run('titanic-clear.dprep', dataflow_idx=0, spark=False)
print ('Titanic dataset shape: {}'.format(titanic_dataset.shape))
print(titanic_dataset.dtypes)

# load features and labels
X = titanic_dataset[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].values
Y = titanic_dataset['survived'].values

# split data 65%-35% into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)

# train a logistic regression model on the training set
classification_model = LogisticRegression().fit(X_train, Y_train)
print(classification_model)

# evaluate the test set
accuracy = classification_model.score(X_test, Y_test)
print("Accuracy is {}".format(accuracy))

# log accuracy which is a single numerical value
logger.log("Accuracy", accuracy)

# calculate and log precesion, recall, and thresholds, which are list of numerical values
y_scores = classification_model.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(Y_test, y_scores[:,1])
logger.log("Precision", precision)
logger.log("Recall", recall)
logger.log("Thresholds", thresholds)

print("")
print("==========================================")
print("Serialize and deserialize using the outputs folder.")
print("")

# serialize the model on disk in the special 'outputs' folder
print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(classification_model, f)
f.close()