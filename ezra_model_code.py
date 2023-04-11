import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
import joblib

# Importing the dataset

df = pd.read_csv('dataset.csv')
df1 = pd.read_csv('Symptom-severity.csv')

# Cleaning the data

cols = df.columns
data = df[cols].values.flatten()
s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)
df = pd.DataFrame(s, columns=df.columns)
df = df.fillna(0)

# Note that earlier I mentioned that we have weighate against each symptom
# So we will simply perform an encoding operation here against each symptom

vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = \
    df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]

d = pd.DataFrame(vals, columns=cols)

# Weightage of these three aren't available in our dataset-2 hence as of now we are ignoring
d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination', 0)
df = d.replace('foul_smell_of urine', 0)

data = df.iloc[:,1:].values

# These are our Y in prediction (X,Y)
labels = df['Disease'].values

# Train Test split is done from the dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.85)

# We have chosen "SUPPORT_VECTOR_CLASSIFIER_MODEL" for this project
##########################################################
# clf = svm.SVC()
model = SVC()
# creating an instance of that model class
##########################################################

# Hyper-parameter tuning ::
#############################
# As of now kept blank
#############################

# Training the model ::
#############################
model.fit(x_train, y_train)

# Predicting using the test data ::

preds = model.predict(x_test)

# Model Metrics (Accuracy and others) ::
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)

filename = 'ezra_model.sav'
joblib.dump(preds, filename)