# Adopted from: https://github.com/boyangumn/DCN/blob/master/pre_rcv1.py
"""
This script preprocesses the Reuters dataset as described in
    Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis." International conference on machine learning. 2016.
and saves it into the dataset_dir defined in Config.py

"""

from sklearn.datasets import fetch_rcv1
import numpy as np
import os
from scripts.Config import dataset_dir
from pathlib import Path

data_home = Path(f"{dataset_dir}/Reuters")
topics_file = Path(data_home, 'rcv1.topics.hier.orig')

if not topics_file.exists():
    import requests

    data_home.mkdir(exist_ok=True, parents=True)
    response = requests.get(
        'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig')
    txt = str(response.text)
    print(topics_file)
    f = open(topics_file, 'w+')
    f.write(txt)
    f.close()
    # Available here:

cwd = os.getcwd()
data = fetch_rcv1(data_home=data_home, download_if_missing=True)
names = data.target_names

ind = np.full(len(names), False, dtype=bool)
f = open(topics_file, 'r')
count = 0
cat_names = []
for i in range(len(names) + 1):
    s = f.readline()
    if s[8:12] == 'Root':
        ind[i - 1] = True
        cat_names.append(s[50:-1])
        count = count + 1
f.close()

labels = data.target[:, ind].copy()
labels = labels.toarray()
t = labels.sum(axis=1, keepdims=False)
single_docs = np.where(t == 1)[0]

# keep only the documents with single label
labels = labels[single_docs]
docs = data.data[single_docs]
#
frequency = np.squeeze(np.asarray(docs.sum(axis=0)))

fre_ind = np.argsort(frequency)
fre_ind = fre_ind[::-1]
selected_features = fre_ind[0:2000]
#
train_x = docs[:, selected_features].todense().astype(np.float32)
np.save(f'{data_home}/reuters_preprocessed_data.npy', train_x)

target_names = cat_names
target = labels.argmax(axis=1).astype(np.int32)
np.save(f'{data_home}/reuters_preprocessed_target.npy', target)

f = open(topics_file, 'r')
ind2 = np.full(len(names), False, dtype=bool)
for i in range(len(names) + 1):
    s = f.readline()
    if s[9:12] == 'CAT':
        ind2[i - 1] = True
        cat_names.append(s[50:-1])
f.close()

lbls_lvl2 = data.target[:, ind2].copy()
lbls_lvl2 = lbls_lvl2.toarray()
lbls_lvl2 = lbls_lvl2[single_docs]
lbls_lvl2 = np.concatenate([labels, lbls_lvl2], axis=1).astype(np.int32)

np.save(f'{data_home}/reuters_preprocessed_cat_names_2lvls.npy', np.asarray(cat_names))
np.save(f'{data_home}/reuters_preprocessed_labels_2lvls.npy', lbls_lvl2)
