# Zinedine Kahlil Gibran Zidane

<h1 align="center">
    Documentations
</h1>

- [Pandas](https://pandas.pydata.org/pandas-docs/stable/)
- [Seaborn](https://seaborn.pydata.org/api.html)
- [Numpy](https://docs.scipy.org/doc/numpy/user/index.html)
- [Scikit-learn](https://scikit-learn.org/stable/index.html)
- [Imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/index.html)

<h1 align="center">
    Package Installations
</h1>

## pip:
- numpy: `pip install numpy`
- seaborn: `pip install seaborn`
- pandas: `pip install pandas`
- sklearn (Scikit-learn): `pip install scikit-learn`
- imblearn (Imbalanced-learn): `pip install imblearn`
- matplotlib: `pip install matplotlib`

## conda:
- numpy: `conda install -c conda-forge numpy`
- seaborn: `conda install -c conda-forge seaborn`
- pandas: `conda install -c conda-forge pandas`
- sklearn (Scikit-learn): `conda install scikit-learn`
- imblearn (Imbalanced-learn): `conda install -c conda-forge imbalanced-learn`
- matplotlib: `conda install -c conda-forge matplotlib`

<h1 align="center">
    Changelog
</h1>

## 03/09/2019 11:20
- Added Installations section to readme.md

## 02/09/2019 12:00
- Added RandomUnderSampler to lr.py
- Added TomekLinks to lr.py
- Figured that TomekLinks relies heavily on the distance between the positive and negative classes

## 02/09/2019 07:25
- Added BorderlineSMOTE to lr.py
- Added ADASYN to lr.py
- Added RandomOverSampler to lr.py
- Figured that ADASYN performs best in event detection rate

## 02/09/2019 07:00
- Removed lr_nr.py file (Logistic Regression with No-Resampling)
- Added lr.py file (Logistic Regression)
- Added Imbalanced-learn to documentations section
- Figured that SMOTE significantly increases event detection rate

## 01/09/2019 10:57s
- Added lr_nr.py file (Logistic Regression with No-Resampling)
- Figured that Logistic Regression performs poorly in imbalanced data

## 01/09/2019 03:46
- Added scikit-learn to documentations section

## 31/08/2019 00:12
- Added Documentations section

## 30/08/2019 23:45
- Added sample dataset

## 30/08/2019 23:40
- Initial release, empty repo
