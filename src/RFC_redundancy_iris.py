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
# NxN-es feature mátrixot tölti fel
def fill_mx(matrix, feature_list, test_list):
    l = len(test_list)
    for i in range(l-1):
        for j in range(i+1,l):
            x = feature_list.index(test_list[i])
            y = feature_list.index(test_list[j])
            if (x==y): continue
            matrix[min(x,y)][max(x,y)] += 1


# %%
# calculate edge importance, create Adjacency matrix
def recurse(tree, node, list):
    if tree.feature[node] != _tree.TREE_UNDEFINED:
        list.append(feature_name[node])
        recurse(tree, tree.children_left[node], list)
        recurse(tree, tree.children_right[node], list)
        del list[-1]
    else:
        fill_mx(matrix, feature_names, list)
                
feature_names = ['SL', 'SW', 'PL', 'PW']
feature_list = []
matrix = np.array(np.zeros((len(feature_names),len(feature_names))))

#start recurse
for tree in model.estimators_:
    t = tree.tree_
    feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in t.feature
        ]
    recurse(t, 0, feature_list)
print(matrix)

# %%
# sum up node degrees -> node importance
n = 4 
top = np.zeros(n)

for i in range(n):
    sum = np.sum(np.hstack((matrix[0:i,i] , matrix[i,i:n])))
    top[i] = sum

# normalize
top /= np.linalg.norm(top)

# sort vector
order = np.sort(top)[::-1]
order_i = np.argsort(top)[::-1]

# create dictionary
f_w = dict(zip(feature_names, top))
f_w

# print
for v, i in zip(order,order_i):
    print("feature: {} \t {}".format(feature_names[i],v))

# %% [markdown]
# RFC analisys
# %%
# analize RFC paths

info = []

# calculate edge importance, create Adjacency matrix
def recurse_mod(tree, node, f_list):
    if tree.feature[node] != _tree.TREE_UNDEFINED:
        f_list.append(feature_name[node])
        recurse_mod(tree, tree.children_left[node], f_list)
        recurse_mod(tree, tree.children_right[node], f_list)
        del f_list[-1]
    else:
        path_weight = np.sum([f_w[x] for x in f_list])
        output = np.argmax(tree.value[node]) 
        info.append((output, path_weight, f_list.copy()))

#start recurse
for tree in model.estimators_:
    t = tree.tree_
    feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in t.feature
        ]
    recurse_mod(t, 0, feature_list)

# %%

print('Az első 10 path')
for line in info[0:10]:
    out, wsum, flist = line
    print("Output: {}, weight {:.3}, features: {}".format(out, wsum, flist))
    
# %%
import pandas as pd 
df = pd.DataFrame(info)
df.columns=['output', 'path_weight', 'feature list']
df
# %%
df[df['output'] == 2]
# %%
df.sort_values('path_weight', ascending=False)
# %%
