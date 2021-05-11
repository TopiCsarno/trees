# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn import tree as sktree
import graphviz
# %%
# load iris dataset, train RFC
iris = load_iris() 
X = iris['data']
y = iris['target']

model= RandomForestClassifier(random_state=0)
model.fit(X, y)
# %%
n = 52  # határeset
x = X[n:n+1]
tree_idx = 0
n_classes = 3
n_trees = 100
# %%
# adott mintára fa szavazatok
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
        if (abs(diff) <= 0.1):
            f_list.append((feature, input_val, threshold, diff))
        if (input_val <= threshold):
            recurse(idx, tree, tree.children_left[node], f_list)
        else:
            recurse(idx, tree, tree.children_right[node], f_list)
    else:
        output = np.argmax(tree.value[node]) 
        info.append((idx, output, f_list.copy()))

info = []
feature_names = ['SL', 'SW', 'PL', 'PW']
for idx, tree in enumerate(model.estimators_):
    feature_list = []
    t = tree.tree_
    recurse(idx, t, 0, feature_list)

# %%
for tuple in info:
    idx, output, line = tuple
    for data in line:
        fidx, val, th, diff = data
        feature = feature_names[fidx]
        print("Tree: {} out: {}  feature: {}  {} <= {:4.5}, diff: {:4.5}".format(idx, output, feature, val, th, diff))
# %%
# modding

print("Eredeti értékek:")

print(x[0])
x_ = x+[0,0,0.15,0] # 3. featurehöz +0.15 (3%os módosítás)

print("\nMódosított értékek:")
print(x_[0])

ratio = (1-(np.linalg.norm(x) / np.linalg.norm(x_)))*100
print("\nMódosítás mértéke: {:.2}%".format(ratio))
# diff

print("\nFa szavazatok:")
print(count_votes(x))
print(count_votes(x_))
# %%

# %%
