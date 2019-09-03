# Mandatory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter

# Oversampling
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
# Undersampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels

# Parameters
seed=1
cross_validations=5

# modified glass dataset, positiveClass=2, negativeClass=[0, 1, 6], samples=192, positives=17, negatives=175
featureNames= ["Rl", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Class"]
df = pd.read_csv(r'dataset/sample/glass.csv', header=None, names=featureNames)
X = df
y = df.pop("Class").values
print(Counter(y))

### RESAMPLING ###
## Oversampling
# SMOTE
smote = SMOTE(random_state=seed)
X_smote, y_smote = smote.fit_resample(X, y)
# BorderlineSMOTE
bsmote = BorderlineSMOTE(random_state=seed)
X_bsmote, y_bsmote = bsmote.fit_resample(X, y)
# ADASYN
adasyn = ADASYN(random_state=seed)
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
# RandomOverSampler
ros = RandomOverSampler(random_state=seed)
X_ros, y_ros = ros.fit_resample(X, y)

## Undersampling
# RandomUnderSampler
rus = RandomUnderSampler(random_state=seed)
X_rus, y_rus = rus.fit_resample(X, y)
# TomekLinks
tl = TomekLinks(sampling_strategy='majority', random_state=seed)
X_tl, y_tl = tl.fit_resample(X, y)
print(Counter(y_tl))
### CLASSIFICATION ###

# LR with no resmpling
lr_nr = LogisticRegressionCV(cv=cross_validations, random_state=seed, solver='liblinear', multi_class='ovr').fit(X, y)
y_pred_nr = lr_nr.predict(X)
## LR with oversampling techniques
# LR with SMOTE
lr_smote = LogisticRegressionCV(cv=cross_validations, random_state=seed, solver='liblinear', multi_class='ovr').fit(X_smote, y_smote)
y_pred_smote = lr_smote.predict(X)
# LR with BorderlineSMOTE
lr_bsmote = LogisticRegressionCV(cv=cross_validations, random_state=seed, solver='liblinear', multi_class='ovr').fit(X_bsmote, y_bsmote)
y_pred_bsmote = lr_bsmote.predict(X)
# LR with ADASYN
lr_adasyn = LogisticRegressionCV(cv=cross_validations, random_state=seed, solver='liblinear', multi_class='ovr').fit(X_adasyn, y_adasyn)
y_pred_adasyn = lr_adasyn.predict(X)
# LR with RandomOverSampler
lr_ros = LogisticRegressionCV(cv=cross_validations, random_state=seed, solver='liblinear', multi_class='ovr').fit(X_ros, y_ros)
y_pred_ros = lr_ros.predict(X)
## LR with undersampling techniques
# LR with RandomUnderSampler
lr_rus = LogisticRegressionCV(cv=cross_validations, random_state=seed, solver='liblinear', multi_class='ovr').fit(X_rus, y_rus)
y_pred_rus = lr_rus.predict(X)
# LR with BorderlineSMOTE
lr_tl = LogisticRegressionCV(cv=cross_validations, random_state=seed, solver='liblinear', multi_class='ovr').fit(X_tl, y_tl)
y_pred_tl = lr_tl.predict(X)

### EVALUATION ###

# confusion matrix with no resampling
tn_nr, fp_nr, fn_nr, tp_nr = confusion_matrix(y, y_pred_nr).ravel()
# confusion matrix with SMOTE
tn_smote, fp_smote, fn_smote, tp_smote = confusion_matrix(y, y_pred_smote).ravel()
# confusion matrix with BorderlineSMOTE
tn_bsmote, fp_bsmote, fn_bsmote, tp_bsmote = confusion_matrix(y, y_pred_bsmote).ravel()
# confusion matrix with ADASYN
tn_adasyn, fp_adasyn, fn_adasyn, tp_adasyn = confusion_matrix(y, y_pred_adasyn).ravel()
# confusion matrix with RandomOverSampler
tn_ros, fp_ros, fn_ros, tp_ros = confusion_matrix(y, y_pred_ros).ravel()
# confusion matrix with RandomUnderSampler
tn_rus, fp_rus, fn_rus, tp_rus = confusion_matrix(y, y_pred_rus).ravel()
# confusion matrix with TomekLinks
tn_tl, fp_tl, fn_tl, tp_tl = confusion_matrix(y, y_pred_tl).ravel()

print('\nNo Resampling Accuracy:', lr_nr.score(X, y))
print('Smote Accuracy:', lr_smote.score(X, y))
print('BorderlineSmote Accuracy:', lr_bsmote.score(X, y))
print('ADASYN Accuracy:', lr_adasyn.score(X, y))
print('RandomOverSampler Accuracy:', lr_ros.score(X, y))
print('RandomUnderSampler Accuracy:', lr_rus.score(X, y))
print('TomekLinks Accuracy:', lr_tl.score(X, y))

print('\nTN, FP, FN, TP:')
print('No Resampling:', tn_nr, fp_nr, fn_nr, tp_nr)
print('SMOTE:', tn_smote, fp_smote, fn_smote, tp_smote)
print('BorderlineSMOTE:', tn_bsmote, fp_bsmote, fn_bsmote, tp_bsmote)
print('ADASYN:', tn_adasyn, fp_adasyn, fn_adasyn, tp_adasyn)
print('RandomOverSampler:', tn_ros, fp_ros, fn_ros, tp_ros)
print('RandomUnderSampler:', tn_rus, fp_rus, fn_rus, tp_rus)
print('TomekLinks:', tn_tl, fp_tl, fn_tl, tp_tl)
