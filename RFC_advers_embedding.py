# %%
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn import tree as sktree
from sklearn.metrics import accuracy_score
import graphviz

# %%
X_train = pd.read_pickle('./data/Race_x_train.pkl')
y_train = pd.read_pickle('./data/Race_y_train_so.pkl')
X_test = pd.read_pickle('./data/Race_x_test.pkl')
y_test = pd.read_pickle('./data/Race_y_test_so.pkl')
model = pd.read_pickle('./data/Race_model_so.pkl')

# model= RandomForestClassifier(random_state = 100)
# model.fit(X_train, y_train)
accuracy_score(y_test, model.predict(X_test))
# %%
def tree_votes(x):
    votes = np.zeros(n_trees, dtype=int)
    for idx, tree in enumerate(model.estimators_):
        votes[idx] = tree.predict(x)
    return votes

def count_votes(x):
    votes = np.zeros(n_classes)
    for tree in model.estimators_:
        votes[int(tree.predict(x))] += 1
    return votes

# %%
# constants
n = 2  # sample number
x = X_test.values[n:n+1]
tree_idx = 0
n_classes = 4
n_trees = 100
n_features = 128

# 1 db embedding
print(x.shape)

# %%
# Próbálgatás eltolással (centroidok és pont távolsága)
average = pd.read_pickle('./data/average.pkl')

i = 3
av = np.expand_dims(average[i],0)

# count_votes(av)
# print(x-av)
diff = av - x

print(count_votes(x))
print(count_votes(x+diff))

print("range")
for k in range(1,11):
    x_ = x + diff*(k/10)
    print(k, count_votes(x_))

# %%
votes = tree_votes(x)
expected = model.predict(x)[0]

# egyes fákon milyen featureök mentén haladunk végig
def recurse(idx, tree, node, f_list):
    if tree.feature[node] != _tree.TREE_UNDEFINED:
        # if threshold is close to our value, add it to list
        feature = tree.feature[node]
        threshold = tree.threshold[node]
        input_val = x[0,feature]
        diff = threshold-input_val
        # only add close values
        if (abs(diff) <= input_val*0.1):
            f_list.append((feature, input_val, threshold, diff))
        if (input_val <= threshold):
            recurse(idx, tree, tree.children_left[node], f_list)
        else:
            recurse(idx, tree, tree.children_right[node], f_list)
    else:
        output = np.argmax(tree.value[node]) 
        info.append((idx, output, f_list.copy()))

info = []
feature_names = ['f'+str(x+1) for x in range(n_features)]
for idx, tree in enumerate(model.estimators_):
    feature_list = []
    t = tree.tree_
    recurse(idx, t, 0, feature_list)

# %%
feature_counts = np.zeros(n_features, dtype='int')
for tuple in info:
    idx, output, line = tuple
    for data in line:
        fidx, val, th, diff = data
        feature = feature_names[fidx]
        feature_counts[fidx] += 1
        print("Tree: {} out: {}  feature: {}  {:4.3} <= {:4.3}, diff: {:4.3}".format(idx, output, feature, val, th, diff))
        
np.argmax(feature_counts)
# leggyakrabban előjövő feature
# %%

# %%

