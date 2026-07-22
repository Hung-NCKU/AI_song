import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns       
from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import sklearn
import warnings
warnings.filterwarnings('ignore')

train = pd.read_parquet("label_test_source.parquet")
print(train.shape)
corrmat = train.corr()
k = 10
cols = corrmat.nlargest(k, 'song_id')['song_id'].index
print(cols)
cm = train[cols].corr()
f, ax = plt.subplots(figsize=(14, 11))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,fmt='.2f')
plt.show()