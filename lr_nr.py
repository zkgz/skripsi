import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels

featureNames= ["Rl", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Class"]
df = pd.read_csv(r'dataset/sample/glass.csv', header=None, names=featureNames)

X = df
y = df.pop("Class").values

#instantiate
ohe = OneHotEncoder(sparse=False)
#fit-transform
X = ohe.fit_transform(X)

clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X, y)
y_pred = clf.predict(X)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
print(tn, fp, fn, tp)
print(clf.score(X, y))
