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

# %% get some samples
n_classes = 4

embeddings = averages.copy()

for i in range(0,n_classes):
    kek = df[df['label']==i+1].head(10)
    print(kek.shape)
    embeddings = np.vstack((embeddings, kek.values[:,:-1]))

embeddings.shape

# %%
# import pickle

# with open('./data/average.pkl', 'wb') as f:
#     pickle.dump(averages, f)
# %%

# mintám:
n = 4  # sample number
x = X_test.values[n:n+1]
x
# %%
# Tweekelés
x_ = x.copy()

# Tree 4, Tree 17
x_[:,10] += -0.0268
# Tree 9
x_[:,77] += -0.0228
# Tree 13, 29, 31
x_[:,43] += -0.02737
# Tree 24
x_[:,8] += 0.02539
# Tree 35
x_[:,88] += -0.0314
# Tree 37
x_[:,105] += -0.02175
# Tree 41
x_[:,44] += 0.02302

x_
# %%
embeddings = np.vstack((x, x_, embeddings))
embeddings.shape
# %%

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_new = pca.fit_transform(X_train)
# %%
X_new.shape
# %%
from matplotlib import pyplot as plt 
plt.scatter(X_new[:,0], X_new[:,1])
# %%
