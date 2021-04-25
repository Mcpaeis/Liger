import torch
import numpy as np
import operator
import itertools
from sklearn.cluster import AgglomerativeClustering as agglo
from ect.utils.evaluation.dendrogram_purity import *


def dendrogram_purity_tree_from_clusters(dec_module, labels, linkage="single"):
    centers = dec_module.centers.detach().cpu().numpy()

    clustering = agglo(compute_full_tree=True, linkage=linkage).fit(centers)

    grouped_ids = {k: [x[0] for x in v] for k, v in
                   itertools.groupby(sorted(enumerate(labels), key=operator.itemgetter(1)), key=operator.itemgetter(1))}

    def map_tree_rec(node):
        if node.is_leaf:
            node.dp_ids = grouped_ids.get(node.dp_ids[0], [])
        else:
            map_tree_rec(node.left_child)
            map_tree_rec(node.right_child)

    tree = to_dendrogram_purity_tree(clustering.children_)
    map_tree_rec(tree)
    return tree