# %%
import numpy as np
import pandas as pd

X_train = pd.read_pickle('./data/Race_x_train.pkl')
y_train = pd.read_pickle('./data/Race_y_train_so.pkl')
X_test = pd.read_pickle('./data/Race_x_test.pkl')
y_test = pd.read_pickle('./data/Race_y_test_so.pkl')

# %%
y_test
# %%
df = X_train.copy()

df['label'] = y_train.values
# %%
df
# %%
# white
n_features = 128
n_classes = 4

averages = np.zeros((n_classes,n_features))

for i in range(0,n_classes):
    df_ = df[df['label']==i+1]
    averages[i] = np.average(df_.values, axis=0)[:-1]
# %%

averages.shape

# %%

import pickle

with open('./data/average.pkl', 'wb') as f:
    pickle.dump(averages, f)
# %%
