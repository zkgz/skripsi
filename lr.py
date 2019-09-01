import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn import datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels

# modified glass dataset, positiveClass=2, negativeClass=[0, 1, 6], samples=192, positives=17, negatives= 175
featureNames= ["Rl", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Class"]
df = pd.read_csv(r'dataset/sample/glass.csv', header=None, names=featureNames)

X = df
y = df.pop("Class").values
#instantiate
#ohe = OneHotEncoder(sparse=False, categories='auto')
#fit-transform
#X = ohe.fit_transform(X)

sm = SMOTE(random_state=1)
X_res, y_res = sm.fit_resample(X, y)

# LR with no resmpling
lr_nr = LogisticRegressionCV(cv=5, random_state=0, solver='liblinear', multi_class='ovr').fit(X, y)
y_pred_nr = lr_nr.predict(X)

# LR with SMOTE
lr_smote = LogisticRegressionCV(cv=5, random_state=0, solver='liblinear', multi_class='ovr').fit(X_res, y_res)
y_pred_smote = lr_smote.predict(X)

# confusion matrix with no resampling
tn_nr, fp_nr, fn_nr, tp_nr = confusion_matrix(y, y_pred_nr).ravel()
# confusion matrix with SMOTE
tn_smote, fp_smote, fn_smote, tp_smote = confusion_matrix(y, y_pred_smote).ravel()

print('No Resampling Accuracy:', lr_nr.score(X, y))
print('Smote Accuracy:', lr_smote.score(X, y))
print('\nTN, FP, FN, TP:')
print('No Resampling:', tn_nr, fp_nr, fn_nr, tp_nr)
print('Smote:', tn_smote, fp_smote, fn_smote, tp_smote)
