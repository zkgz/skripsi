import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r'dataset/sample/glass.csv', header=None)
df["target"] = df[[9]]
df["target"] = df["target"].replace("negative", 0)
df["target"] = df["target"].replace("positive", 1)
#color_wheel = {1: "#ff0000",
#               2: "#00ff00"}
#y = df["target"]
#colors = df["target"].map(lambda x: color_wheel.get(x + 1))

#pd.plotting.scatter_matrix(df, color=colors, alpha=0.9)

y = df.pop("target").values
ohe = OneHotEncoder(sparse=False)
transformed = ohe.fit_transform(df)
print(transformed)
print(y)

