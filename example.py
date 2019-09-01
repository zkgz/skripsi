import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels

# modified glass dataset, positiveClass=2, negativeClass=[0, 1, 6], samples=192, positives=17, negatives= 175
featureNames= ["Rl", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Class"]
df = pd.read_csv(r'dataset/sample/glass.csv', header=None, names=featureNames)

#color_wheel = {1: "#ff0000",
#               2: "#00ff00"}
#y = df["target"]
#colors = df["target"].map(lambda x: color_wheel.get(x + 1))

#pd.plotting.scatter_matrix(df, color=colors, alpha=0.9)

# create train and test set
#np.random.seed(1)
#mask = np.random.rand(len(df)) <= 0.8
#train = df[mask]
#test = df[~mask]

X = df
y = df.pop("Class").values

#instantiate
ohe = OneHotEncoder(sparse=False)
#fit-transform
X_transformed = ohe.fit_transform(X)

clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X_transformed, y)
y_pred = clf.predict(X_transformed)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
print(tn, fp, fn, tp)
print(clf.score(X_transformed, y))
