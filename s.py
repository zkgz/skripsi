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
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
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
df = pd.read_csv(r'dataset/segment.csv', header=0)
print(df.describe())
X = df
y = df.pop("class").values

# Models #
kf = KFold(n_splits=cross_validations, random_state=seed, shuffle=True)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))

# 6 resampling strategies, 1 no-resampling #
res_name = ['smote', 'bsmote', 'adasyn', 'ros', 'rus', 'tl', 'noRes']
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

# Normalize data
print(X.head())
scaler.fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X)

# cross validations
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    

    for i in range(len(res)):
        #resample data
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
            round(avg[i, j, 3, 1], 2), )
    print()

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

###########################################
# classifier
accs = np.zeros((6, 2))
pres = np.zeros((6, 2))
recs = np.zeros((6, 2))
fss = np.zeros((6, 2))
for i in range(len(clf)):
    for j in range(len(res) - 1):
        for k in range(2):
            accs[i, k] += avg[j, i, 0, k] / 6
            pres[i, k] += avg[j, i, 1, k] / 6
            recs[i, k] += avg[j, i, 2, k] / 6
            fss[i, k] += avg[j, i, 3, k] / 6
    for j in range(2):
        accs[i, j] = round(accs[i, j], 2)
        pres[i, j] = round(pres[i, j], 2)
        recs[i, j] = round(recs[i, j], 2)
        fss[i, j] = round(fss[i, j], 2)

labels = clf_name
x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

for i in range(2):
    if(i==0):
        state = 'majority'
    else:
        state = 'minority'
    fig, ax = plt.subplots()
    rects_ = []
    rects_.append(ax.bar(x - width-(width/2), accs[:,i], width, label='acc'))
    rects_.append(ax.bar(x - width/2, pres[:,i], width, label='pre'))
    rects_.append(ax.bar(x + width/2, recs[:,i], width, label='rec'))
    rects_.append(ax.bar(x + width+(width/2), fss[:,i], width, label='fs'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')

    ax.set_title('Average Scores by Classifiers on ' + state + ' class - ' + str(cross_validations) + ' cross validataions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    autolabel(rects_, ax)
#################################
# resampling
accs = np.zeros((7, 2))
pres = np.zeros((7, 2))
recs = np.zeros((7, 2))
fss = np.zeros((7, 2))
for i in range(len(res)):
    for j in range(len(clf)):
        for k in range(2):
            accs[i, k] += avg[i, j, 0, k] / 7
            pres[i, k] += avg[i, j, 1, k] / 7
            recs[i, k] += avg[i, j, 2, k] / 7
            fss[i, k] += avg[i, j, 3, k] / 7
    for j in range(2):
        accs[i, j] = round(accs[i, j], 2)
        pres[i, j] = round(pres[i, j], 2)
        recs[i, j] = round(recs[i, j], 2)
        fss[i, j] = round(fss[i, j], 2)
        
labels = res_name
x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars
for i in range(2):
    if(i==0):
        state = 'majority'
    else:
        state = 'minority'
    fig, ax = plt.subplots()
    rects_ = []
    rects_.append(ax.bar(x - width-(width/2), accs[:,i], width, label='acc'))
    rects_.append(ax.bar(x - width/2, pres[:,i], width, label='pre'))
    rects_.append(ax.bar(x + width/2, recs[:,i], width, label='rec'))
    rects_.append(ax.bar(x + width+(width/2), fss[:,i], width, label='fs'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')

    ax.set_title('Average Scores by Resampling Strategy on ' + state + ' class - ' + str(cross_validations) + ' cross validataions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    autolabel(rects_, ax)

names = []
combinations = []
for i in range(len(clf)):
    for j in range(len(res) - 1):
        names.append(res_name[j]+"-"+clf_name[i])
        combinations.append(round(avg[i, j, 2, 1], 2))
fig, ax = plt.subplots()
x = np.arange(len(names))  # the label locations
width = 0.2  # the width of the bars
rects_ = []
labels = names
rects_.append(ax.bar(x, combinations, width, label='rec'))
ax.set_ylabel('Scores')
ax.set_title('Average Scores by Individual combination on minority class - ' + str(cross_validations) + ' cross validataions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
autolabel(rects_, ax)
plt.show()
