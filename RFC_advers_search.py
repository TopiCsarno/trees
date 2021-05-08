# %%
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn.datasets import load_iris
from sklearn import tree as sktree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import graphviz

# Iris data
# iris = load_iris() 
# X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.3, random_state=0)

# Face embeddings
X_train = pd.read_pickle('./data/Race_x_train.pkl').values
y_train = pd.read_pickle('./data/Race_y_train_so.pkl').values-1
X_test = pd.read_pickle('./data/Race_x_test.pkl').values
y_test = pd.read_pickle('./data/Race_y_test_so.pkl').values-1

# %%
# constants
n = 4  # sample number (iris: 10)
x = X_test[n:n+1]
y = y_test[n]
n_classes = 4
n_features = 128
n_trees = 50

print(n_classes, n_features)

# feature_names = ['SL', 'SW', 'PL', 'PW']  # iris
feature_names = ['f'+str(x) for x in range(n_features)]  # face embeddings

# Train model
model= RandomForestClassifier(random_state = 0, n_estimators=n_trees)
model.fit(X_train, y_train)
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
# sample kereső
for n in range(20):
    xs = X_test[n:n+1]
    ys = y_test[n]
    print(ys, count_votes(xs))
# %%
print(count_votes(x))
print(y)
# %%
# fa kirajzolás
# tree_idx = 2
# dot_data = sktree.export_graphviz(model.estimators_[tree_idx], out_file=None, 
#                      feature_names=feature_names,  
#                      class_names=iris['target_names'],
#                      filled=True, rounded=True,  
#                      node_ids=True,
#                      special_characters=True)  

# graph = graphviz.Source(dot_data)  
# graph

# %%
votes = tree_votes(x)

# egyes fákon milyen featureök mentén haladunk végig
def recurse(idx, tree, node, f_list):
    if tree.feature[node] != _tree.TREE_UNDEFINED:
        # if threshold is close to our value, add it to list
        feature = tree.feature[node]
        threshold = tree.threshold[node]
        input_val = x[0,feature]
        diff = threshold-input_val
        # if condition is close to input value, explore both paths
        if (abs(diff) <= input_val*0.2):
            if (input_val <= threshold):
                # True (left)
                f_list.append((node, feature, 0.0))
                recurse(idx, tree, tree.children_left[node], f_list)
                del f_list[-1]
                # False (right)
                f_list.append((node, feature, diff))
                recurse(idx, tree, tree.children_right[node], f_list)
            else:
                # False (right)
                f_list.append((node, feature, 0.0))
                recurse(idx, tree, tree.children_right[node], f_list)
                del f_list[-1]
                # True (left)
                f_list.append((node, feature, diff))
                recurse(idx, tree, tree.children_left[node], f_list)

        # else go only one path
        elif (input_val <= threshold):
            recurse(idx, tree, tree.children_left[node], f_list)
        else:
            recurse(idx, tree, tree.children_right[node], f_list)
    else:
        output = np.argmax(tree.value[node]) 
        # if output is wrong
        if (output != y):
            info.append((idx, node, output, f_list.copy()))

info = []
for idx, tree in enumerate(model.estimators_):
    #skip incorrect trees
    pred = tree.predict(x)[0]
    if pred != y:
        continue
    feature_list = []
    t = tree.tree_
    recurse(idx, t, 0, feature_list)

# feature_counts = np.zeros(n_features, dtype='int')
for tuple in info:
    idx, outnode, output, line = tuple
    print(" Tree: {}, output: {}".format(idx, output))
    for data in line:
        node, fidx, diff = data
        feature = feature_names[fidx]
        # feature_counts[fidx] += 1
        print("\t{:4} diff: {:2.4}".format(feature, diff))
        
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

print('tree votes:')
print('\ncounts:')
print(count_votes(x))
print(count_votes(x_))

# %%
print(model.predict(x))
print(model.predict(x_))
# %%

np.linalg.norm(x-x_)
# %%
