# Mandatory
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# ignore python warnings
warnings.filterwarnings("ignore")

#### Parameters ####
seed = 101
cross_validations = 5
####################

# Modified glass dataset, positiveClass=2, negativeClass=[0, 1, 6], samples=192, positives=17, negatives=175
featureNames = ["Rl", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Class"]
df = pd.read_csv(r'dataset/sample/glass.csv', header=None, names=featureNames)
X = df
y = df.pop("Class").values

# K-Fold Cross Validation #
kf = KFold(n_splits=cross_validations, random_state=seed, shuffle=True)

# 6 resampling strategies, 1 no-resampling #
res_name = ['smote', 'bsmote', 'adasyn', 'ros', 'rus', 'tl', 'no-resampling']
res = []
res.append(SMOTE(random_state=seed))
res.append(BorderlineSMOTE(random_state=seed))
res.append(ADASYN(random_state=seed))
res.append(RandomOverSampler(random_state=seed))
res.append(RandomUnderSampler(random_state=seed))
res.append(TomekLinks(sampling_strategy='majority', random_state=seed))
res.append(None)

# 6 classification models #
clf_name = ['lr', 'svm', 'mlp', 'dt', 'nb', 'knn']
clf = []
clf.append(LogisticRegression(solver='liblinear', multi_class='ovr', random_state=seed))
clf.append(SVC(kernel='rbf', gamma='auto', random_state=seed))
clf.append(MLPClassifier(activation='logistic', solver='lbfgs', hidden_layer_sizes=(5, 2), random_state=seed))
clf.append(DecisionTreeClassifier(criterion='entropy', random_state=seed))
clf.append(ComplementNB(alpha=0.5, norm=False))
clf.append(KNeighborsClassifier(n_neighbors=3, weights='distance'))

avg = np.zeros((7, 6, 4, 2)) # 6+1 resampling, 6 classifier, 4 metrics (acc, pre, rec, fs), 2 class

for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    for i in range(len(res)):
        if(i==6):
            X_res, y_res = X_train, y_train
        else:
            X_res, y_res = res[i].fit_resample(X_train, y_train)
        
        # fit with balanced (resampled) data
        for j in range(len(clf)):
            model = clf[j].fit(X_res, y_res)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            pre, rec, fs, sup = precision_recall_fscore_support(y_test, y_pred)

            # in pre, rec, fs: index 0 = negative (majority), 1 = positive (minority)
            avg[i, j, 0, 0] += acc
            avg[i, j, 1, 0] += pre[0]
            avg[i, j, 2, 0] += rec[0]
            avg[i, j, 3, 0] += fs[0]

            avg[i, j, 0, 1] += acc
            avg[i, j, 1, 1] += pre[1]
            avg[i, j, 2, 1] += rec[1]
            avg[i, j, 3, 1] += fs[1]

# print score
for i in range(len(res)):
    for j in range(len(clf)):
        avg[i, j, 0:4] /= cross_validations
        print(clf_name[j], ' with ', res_name[i], ': ', 
            round(avg[i, j, 0, 1], 2), 
            round(avg[i, j, 1, 1], 2), 
            round(avg[i, j, 2, 1], 2),
            round(avg[i, j, 3, 1], 2))
    print()

accs = np.zeros(6)
pres = np.zeros(6)
recs = np.zeros(6)
fss = np.zeros(6)
for i in range(len(clf)):
    for j in range(len(res) - 1):
        accs[i] += avg[j, i, 0, 0]
        pres[i] += avg[j, i, 1, 0]
        recs[i] += avg[j, i, 2, 0]
        fss[i] += avg[j, i, 3, 0]
    accs[i] /= 6
    pres[i] /= 6
    recs[i] /= 6
    fss[i] /= 6

labels = clf_name
x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()
rects_ = []
rects_.append(ax.bar(x - width, accs, width, label='acc'))
rects_.append(ax.bar(x - width/2, pres, width, label='pre'))
rects_.append(ax.bar(x + width/2, recs, width, label='rec'))
rects_.append(ax.bar(x + width, fss, width, label='fs'))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Average Scores by Classifiers on majority class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#####################
accs = np.zeros(6)
pres = np.zeros(6)
recs = np.zeros(6)
fss = np.zeros(6)
for i in range(len(clf)):
    for j in range(len(res) - 1) :
        accs[i] += avg[j, i, 0, 1]
        pres[i] += avg[j, i, 1, 1]
        recs[i] += avg[j, i, 2, 1]
        fss[i] += avg[j, i, 3, 1]
    accs[i] /= 6
    pres[i] /= 6
    recs[i] /= 6
    fss[i] /= 6

fig2, ax2 = plt.subplots()
rects_2 = []
rects_2.append(ax2.bar(x - width, accs, width, label='acc'))
rects_2.append(ax2.bar(x - width/2, pres, width, label='pre'))
rects_2.append(ax2.bar(x + width/2, recs, width, label='rec'))
rects_2.append(ax2.bar(x + width, fss, width, label='fs'))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax2.set_ylabel('Scores')
ax2.set_title('Average Scores by Classifiers on minority class')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()

########################
accs = np.zeros(6)
pres = np.zeros(6)
recs = np.zeros(6)
fss = np.zeros(6)
for i in range(len(res) - 1):
    for j in range(len(clf)) :
        accs[i] += avg[i, j, 0, 0]
        pres[i] += avg[i, j, 1, 0]
        recs[i] += avg[i, j, 2, 0]
        fss[i] += avg[i, j, 3, 0]
    accs[i] /= 6
    pres[i] /= 6
    recs[i] /= 6
    fss[i] /= 6

fig3, ax3 = plt.subplots()
rects_3 = []
rects_3.append(ax3.bar(x - width, accs, width, label='acc'))
rects_3.append(ax3.bar(x - width/2, pres, width, label='pre'))
rects_3.append(ax3.bar(x + width/2, recs, width, label='rec'))
rects_3.append(ax3.bar(x + width, fss, width, label='fs'))

# Add some text for labels, title and custom x-axis tick labels, etc.
labels = res_name

ax3.set_ylabel('Scores')
ax3.set_title('Average Scores by Resampling Strategy on majority class')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend()
########################
accs = np.zeros(6)
pres = np.zeros(6)
recs = np.zeros(6)
fss = np.zeros(6)
for i in range(len(res) - 1):
    for j in range(len(clf)) :
        accs[i] += avg[i, j, 0, 1]
        pres[i] += avg[i, j, 1, 1]
        recs[i] += avg[i, j, 2, 1]
        fss[i] += avg[i, j, 3, 1]
    accs[i] /= 6
    pres[i] /= 6
    recs[i] /= 6
    fss[i] /= 6

fig4, ax4 = plt.subplots()
rects_4 = []
rects_4.append(ax4.bar(x - width, accs, width, label='acc'))
rects_4.append(ax4.bar(x - width/2, pres, width, label='pre'))
rects_4.append(ax4.bar(x + width/2, recs, width, label='rec'))
rects_4.append(ax4.bar(x + width, fss, width, label='fs'))

# Add some text for labels, title and custom x-axis tick labels, etc.
labels = res_name

ax4.set_ylabel('Scores')
ax4.set_title('Average Scores by Resampling Strategy on minority class')
ax4.set_xticks(x)
ax4.set_xticklabels(labels)
ax4.legend()
########################

def autolabel(list, axes):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rects in list:
        for rect in rects:
            height = rect.get_height()
            axes.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

autolabel(rects_, ax)
autolabel(rects_2, ax2)
autolabel(rects_3, ax3)
autolabel(rects_4, ax4)
fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
plt.show()