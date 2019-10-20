import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
from collections import Counter
from statsmodels.graphics.gofplots import qqplot

from sklearn.preprocessing import RobustScaler
# ignore python warnings
warnings.filterwarnings("ignore")

#### Parameters ####
seed = 101
####################


scaler = RobustScaler()

# Modified glass dataset, positiveClass=2, negativeClass=[0, 1, 6], samples=192, positives=17, negatives=175
featureNames = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our", 
"word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
"word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
"word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
"word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
"word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
"word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
"word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
"word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
"word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
"char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average",
"capital_run_length_longest", "capital_run_length_total", "class"]
df = pd.read_csv(r'dataset/spambase.csv', header=None, names=featureNames)


colors = ["#00FF00", "#FF0000"]
sns.countplot('class', data=df, palette=colors)
plt.title('Class Distributions\n(0: Ham || 1: Spam)', fontsize=14)


#df = df
#y = df.pop("class").values

print(df.describe())
df['scaled_crla'] = scaler.fit_transform(df['capital_run_length_average'].values.reshape(-1,1))
df['scaled_crll'] = scaler.fit_transform(df['capital_run_length_longest'].values.reshape(-1,1))
df['scaled_crlt'] = scaler.fit_transform(df['capital_run_length_total'].values.reshape(-1,1))
df.drop(['capital_run_length_average','capital_run_length_longest', 'capital_run_length_total'], axis=1, inplace=True)

def distplot(data, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None, name="an attribute"):
    sns.distplot(a=data, bins=bins, hist=hist, kde=kde, rug=rug, fit=fit, hist_kws=hist_kws, kde_kws=kde_kws, rug_kws=rug_kws, fit_kws=fit_kws, color=color, vertical=vertical, norm_hist=norm_hist, axlabel=axlabel, label=label, ax=ax)
    ax.set_title('Distribution of '+name)
    ax.set_xlim([min(data), max(data)])

fig, ax = plt.subplots(3, 1)

distplot(df['scaled_crla'], ax=ax[0], color='r', norm_hist=False, name="scaled_crla")
distplot(df['scaled_crll'], ax=ax[1], color='g', norm_hist=False, name="scaled_crll")
distplot(df['scaled_crlt'], ax=ax[2], color='b', norm_hist=False, name="scaled_crlt")

f, axes = plt.subplots(1, 3)

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="class", y="scaled_crla", data=df, ax=axes[0])
sns.boxplot(x="class", y="scaled_crll", data=df, ax=axes[1])
sns.boxplot(x="class", y="scaled_crlt", data=df, ax=axes[2])
axes[0].set_title('scaled_crla vs Class Negative Correlation')
axes[1].set_title('scaled_crll vs Class Negative Correlation')
axes[2].set_title('scaled_crlt vs Class Negative Correlation')

#qqplot(df, line='s')
plt.show()