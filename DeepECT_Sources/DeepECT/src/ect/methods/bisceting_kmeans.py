import numpy as np
from sklearn.cluster.k_means_ import KMeans
import attr
from queue import PriorityQueue
from ect.utils.evaluation.dendrogram_purity import DpNode, DpLeaf

def sum_square_error(centroid, data):
    return np.sum(np.linalg.norm(data - centroid, 2, 1))


"""
This method builds a bisection tree. It splits the node with highest squared error sum.

It is not a very efficient implementation!
"""


@attr.s(cmp=False)
class tree_node(object):
    id = attr.ib()
    centroid = attr.ib()
    split_order = attr.ib(init=False)
    left_child = attr.ib(init=False, default=None)
    right_child = attr.ib(init=False, default=None)

    @property
    def is_leaf(self) -> bool:
        return self.left_child is None


def _n_leafs(root: tree_node) -> int:
    leaf_count = 0

    def recursive(node):
        nonlocal leaf_count
        if node.is_leaf:
            leaf_count += 1
        else:
            recursive(node.left_child)
            recursive(node.right_child)

    return leaf_count


def _assign_leaf_ids(root: tree_node):
    leaf_counter = 0

    def recursive(node):
        nonlocal leaf_counter
        if node.is_leaf:
            node.leaf_id = leaf_counter
            leaf_counter += 1
        else:
            recursive(node.left_child)
            recursive(node.right_child)

    recursive(root)


def bisection(max_k: int, data: np.ndarray) -> tree_node:
    current_k = 1
    data_centroid = np.mean(data, 0)
    root = tree_node(0, data_centroid)
    root_sse = sum_square_error(data_centroid, data)
    next_split_order = 1
    next_node_id = 1
    queue = PriorityQueue()
    queue.put((-1.0 * root_sse, root, data))

    # print(f"rootsse {root.sse}")
    while current_k < max_k:
        _, leaf_to_split, split_data = queue.get()
        # print(f"leaf_to_split sse {leaf_to_split.sse}")
        leaf_to_split.split_order = next_split_order
        next_split_order += 1
        k = KMeans(2)
        labels = np.array(k.fit_predict(split_data), dtype=np.float32)
        labels = labels.reshape([len(labels), 1])

        left_idx = np.asanyarray([i for i in range(split_data.shape[0]) if labels[i] == 0])
        left_data = split_data[left_idx, :]
        left_child = tree_node(next_node_id, np.mean(left_data, 0))
        next_node_id += 1
        leaf_to_split.left_child = left_child
        queue.put((-1.0 * sum_square_error(left_child.centroid, left_data), left_child, left_data))
        # print(f"left_child sse {left_child.sse}")

        right_idx = np.asanyarray([i for i in range(split_data.shape[0]) if labels[i] == 1])
        right_data = split_data[right_idx, :]
        right_child = tree_node(next_node_id, np.mean(right_data, 0))
        next_node_id += 1
        leaf_to_split.right_child = right_child
        queue.put((-1.0 * sum_square_error(right_child.centroid, right_data), right_child, right_data))
        # print(f"right_child sse {right_child.sse}")

        current_k += 1  # it is only one leaf node more

    _assign_leaf_ids(root)

    return root


def predict_by_tree(tree_root: tree_node, data: np.ndarray, n_clusters=-1):
    """

    :param tree_root:
    :param data:
    :param stop_at_split: the result has stop_at_split+1 different labels if -1 then the whole tree is being walked
    :return:
    """
    if n_clusters == -1:
        stop_at_split = -1
    else:
        stop_at_split = n_clusters - 1
    labels = np.zeros([data.shape[0]])

    def walk_tree(node: tree_node, dp: np.ndarray):
        if node.is_leaf or node.split_order > stop_at_split > -1:
            return node.id
        else:
            left_dist = np.sum((dp - node.left_child.centroid) ** 2)
            right_dist = np.sum((dp - node.right_child.centroid) ** 2)
            if left_dist < right_dist:
                return walk_tree(node.left_child, dp)
            else:
                return walk_tree(node.right_child, dp)

    for idx, row in enumerate(data):
        labels[idx] = walk_tree(tree_root, row)

    return labels.astype(np.int)


def predict_id_tree(tree_root: tree_node, data: np.array, ids: np.array = None):
    if ids is None:
        ids = np.array(range(data.shape[0]))

    def empty_recursive(node):
        if node.left_child is None:
            left_child = DpLeaf([], node.id * 1000)
        else:
            left_child = empty_recursive(node.left_child)
        if node.right_child is None:
            right_child = DpLeaf([], node.id * 1000 + 1) 
        else:
            right_child = empty_recursive(node.right_child)
        return DpNode(left_child, right_child)

    def recursive(node, node_data, data_ids):
        if node.is_leaf:
            return DpLeaf(data_ids, node.id)
        else:
            lc = node.left_child
            rc = node.right_child
            sq_diff_left = np.sum((node_data - lc.centroid) ** 2, 1)
            sq_diff_right = np.sum((node_data - rc.centroid) ** 2, 1)

            to_right = sq_diff_left > sq_diff_right
            to_left = np.logical_not(to_right)

            data_left = node_data[to_left, :]
            ids_left = data_ids[to_left]
            data_right = node_data[to_right, :]
            ids_right = data_ids[to_right]
            if data_left.shape[0] == 0:
                left_result = empty_recursive(lc)
            else:
                left_result = recursive(node.left_child, data_left, ids_left)
            if data_right.shape[0] == 0:
                right_result = empty_recursive(lc)
            else:
                right_result = recursive(node.right_child, data_right, ids_right)
            return DpNode(left_result, right_result, node.id)

    return recursive(tree_root, data, ids)
