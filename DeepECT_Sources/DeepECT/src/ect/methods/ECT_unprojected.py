import torch
import torch.utils.data
from ect.utils.pytorch import int_to_one_hot
from sklearn.cluster.k_means_ import k_means
import queue
import numpy as np
from ect.utils.evaluation.dendrogram_purity import DpLeaf, DpNode
import logging

logger = logging.getLogger(__name__)

_prune_threshold = 0.1

"""
This is the unprojected version of ECT. IT SHOULD NOT BE USED.
It only shows the deficiencies an unprojected version has and why we use the projection as explained in the paper!


"""

class ECTnodeUnprojected(torch.nn.Module):
    def __init__(self, node_id: int, tree: 'ECTreeUnprojected', center_value_np,
                 split_level,
                 left_child,
                 right_child):
        super().__init__()

        self.node_id = node_id
        self.tree = (lambda: tree)
        self.split_level = split_level
        self.left_child = left_child
        self.right_child = right_child
        if center_value_np is not None:
            self.center_param = torch.nn.Parameter(torch.tensor(center_value_np, device=tree.device),
                                                   requires_grad=True, )
            self.register_parameter("center_param", self.center_param)
        # self.child_counter = None  # Will be set once the leaf node will be split
        # self.projection_dir = None  # Will be set once the leaf node will be split

    @classmethod
    def new_leaf_node(cls, node_id: int, tree: 'ECTreeUnprojected', center_value_np):
        return cls(node_id, tree, center_value_np, None, None, None)

    @classmethod
    def new_split_node(cls, node_id: int, tree: 'ECTreeUnprojected', split_level: int,
                       left_child: 'ECTnodeUnprojected',
                       right_child: 'ECTnodeUnprojected'):
        c = cls(node_id, tree, None, split_level, left_child, right_child)
        c.split_node(split_level, left_child, right_child)
        return c

    def is_leaf(self):
        return self.center_param is not None

    def split_node(self, split_level: int, left_child: 'ECTreeUnprojected', right_child: 'ECTreeUnprojected'):
        self.split_level = split_level
        self.left_child = left_child
        self.right_child = right_child
        self.center_param = None
        self.register_buffer(f"child_counter", torch.tensor([1.0, 1.0], device=self.tree().device))
        self.register_buffer(f"projection_dir",
                             torch.zeros([left_child.center_param.shape[0], 1], device=self.tree().device),
                             )

    def _check_not_leaf(self):
        if self.is_leaf():
            raise RuntimeError("Cannot be called on a leaf node")

    def child_centers(self):
        self._check_not_leaf()

        if self.left_child.is_leaf():
            left = self.left_child.center_param
        else:
            left = self.tree()._split_node_center_learning[self.left_child.node_id]

        if self.right_child.is_leaf():
            right = self.right_child.center_param
        else:
            right = self.tree()._split_node_center_learning[self.right_child.node_id]

        return torch.cat((left.unsqueeze(0), right.unsqueeze(0)), 0)

    def _update_counters(self, assigned_to_left, assigned_to_right):
        self._check_not_leaf()
        alpha = 0.5

        child_counter_update = torch.stack([assigned_to_left.sum(), assigned_to_right.sum()])
        self.child_counter = alpha * child_counter_update + (1.0 - alpha) * self.child_counter

    def _update_center_buffer(self):
        self._check_not_leaf()

        # updating the center_buffer
        child_centers = self.child_centers()
        new_center = ((child_centers * (self.child_counter.unsqueeze(1) )).sum(
            0)) / (self.child_counter.sum())
        self.tree()._split_node_center_learning[self.node_id] = new_center

    def node_center_np(self):
        """
        :return: The center value as a numpy array
        """
        if self.is_leaf():
            return self.center_param.detach().cpu().numpy()
        else:
            c = self.child_counter.cpu().numpy + 0.0001
            sum = self.left_child.node_center_np() * c[0] + self.right_child.node_center_np() * c[1]
            return sum / np.sum(c)

    def _projected_centers_and_data(self, data_batch):
        self._check_not_leaf()

        projected_centers = self.child_centers()
        projected_data = data_batch
        return projected_centers, projected_data

    def node_prediction_hard(self, data_batch):
        self._check_not_leaf()

        return self.node_prediction_dec(data_batch).argmax(1)

    def loss_data_fixed_centers(self, data_batch, assigned_to_left, assigned_to_right):
        self._check_not_leaf()

        projected_centers, projected_data = self._projected_centers_and_data(data_batch)

        # Left child
        center_left_fixed = projected_centers[0, :].detach().unsqueeze(0).data
        abs_diffs_left = (projected_data - center_left_fixed).abs().sum(1)

        scaled_diff_left = assigned_to_left * abs_diffs_left

        # right child
        center_right_fixed = projected_centers[1, :].detach().unsqueeze(0).data
        abs_diffs_right = (projected_data - center_right_fixed).abs().sum(1)
        scaled_diff_right = assigned_to_right * abs_diffs_right

        loss = torch.mean(scaled_diff_left) + torch.mean(scaled_diff_right)
        return loss

    def loss_centers_fixed_data(self, data_batch, assigned_to_left, assigned_to_right):
        self._check_not_leaf()

        data_batch_fixed = data_batch.detach()

        n_dps_left = assigned_to_left.sum(0).detach()
        if self.left_child.is_leaf() and n_dps_left > 0.0:
            batch_mean_left = (assigned_to_left.unsqueeze(1) * data_batch_fixed).sum(0) / n_dps_left
            loss_left = (self.left_child.center_param.unsqueeze(0) - batch_mean_left).pow(2).sum().sqrt()
        else:
            loss_left = 0.0

        n_dps_right = assigned_to_right.sum(0).detach()
        if self.right_child.is_leaf() and n_dps_right > 0.0:
            batch_mean_right = (assigned_to_right.unsqueeze(1) * data_batch_fixed).sum(0) / n_dps_right
            loss_right = (self.right_child.center_param.unsqueeze(0) - batch_mean_right).pow(2).sum().sqrt()
        else:
            loss_right = 0.0

        return loss_left + loss_right


class ECTreeUnprojected(torch.nn.Module):

    def __init__(self, optimizer, split_dataset_loader_gen, root_centers=None, device=None):
        super().__init__()
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.optimizer = optimizer
        self.split_dataset_loader_gen = split_dataset_loader_gen

        self.pruned_nodes_ids = []  # Contains the ids that have been pruned away

        # The _split_node_center_learning  is None except when during training where is contains the
        # calculated center values with gradient information
        self._split_node_center_learning = None

        self._create_root_node(root_centers)

    def _create_root_node_centers(self):
        node_data = None
        for batch_data in self.split_dataset_loader_gen():
            if node_data is None:
                node_data = batch_data.detach().cpu().numpy()
            else:
                node_data = np.concatenate([node_data, batch_data.detach().cpu().numpy()], 0)
        return k_means(node_data, 2, n_init=20)[0]

    def _create_root_node(self, root_centers=None):

        if root_centers is None:
            root_centers = self._create_root_node_centers()
        self.root_node: ECTnodeUnprojected = ECTnodeUnprojected.new_split_node(0, self, 1,
                                                                         ECTnodeUnprojected.new_leaf_node(1, self,
                                                                                                       root_centers[0,
                                                                                                       :]),
                                                                         ECTnodeUnprojected.new_leaf_node(2, self,
                                                                                                       root_centers[1,
                                                                                                       :]))
        self.add_module(f"node_{self.root_node.left_child.node_id}", self.root_node.left_child)
        self.add_module(f"node_{self.root_node.right_child.node_id}", self.root_node.right_child)

        self.optimizer.add_param_group({'params': self.root_node.left_child.parameters()})
        self.optimizer.add_param_group({'params': self.root_node.right_child.parameters()})

        self.n_splits: int = 1  # current number of splits
        self.next_free_node_id: int = 3

        self.leaf_nodes = [self.root_node.left_child, self.root_node.right_child]
        self._update_leaf_node_mappings()
        self._split_node_center_learning = {}
        self.root_node._update_center_buffer()
        self._split_node_center_learning = None

    @property
    def n_leaf_nodes(self):
        return self.n_splits + 1

    def _update_leaf_node_mappings(self):
        self.leaf_node_id_to_col_idx_mapping = {child.node_id: idx for idx, child in enumerate(self.leaf_nodes)}
        self.col_idx_to_leaf_node_id_mapping = {idx: child.node_id for idx, child in enumerate(self.leaf_nodes)}

    def centers_for_clusters(self, n_clusters):
        if 2 < n_clusters > self.n_splits + 1:
            raise RuntimeError(f"level {n_clusters} is not within 2 and {self.n_splits + 1}")
        max_split_level = n_clusters - 1

        node_queue = queue.PriorityQueue()
        node_queue.put((1, self.root_node, 0))

        centers = queue.PriorityQueue()
        next_free_row = 1

        while True:
            try:
                next_item = node_queue.get(block=False)
                node = next_item[1]
                row_id = next_item[2]

                if node.split_level < max_split_level:
                    if node.left_child.is_leaf():
                        centers.put((row_id, node.left_child.node_center_np()))
                    else:  # If the left child is not a node in the split tree, the path prob for this node is the path prob * dec prob
                        if node.left_child.split_level <= max_split_level:
                            node_queue.put((node.left_child.split_level, node.left_child, row_id))
                        else:
                            centers.put((row_id, node.left_child.node_center_np()))
                    if node.right_child.is_leaf():
                        centers.put((next_free_row, node.right_child.node_center_np()))
                        next_free_row += 1
                    else:  # If the right child is not a node in the split tree, the path prob for this node is the path prob * dec prob
                        if node.right_child.split_level <= max_split_level:
                            node_queue.put((node.right_child.split_level, node.right_child, next_free_row))
                            next_free_row += 1
                        else:
                            centers.put((next_free_row, node.right_child.node_center_np()))
                            next_free_row += 1
                elif node.split_level == max_split_level:
                    centers.put((row_id, node.left_child.node_center_np()))
                    centers.put((next_free_row, node.right_child.node_center_np()))
                    next_free_row = None
                else:
                    raise RuntimeError(
                        f"Not possible node.split_level: {node.split_level} max_split_level: {max_split_level}")
            except queue.Empty:
                break

        centers_col = [np.expand_dims(centers.get()[1], 0) for i in range(centers.qsize())]

        return np.concatenate(centers_col, axis=0)

    def leaf_node_centers(self):
        return torch.cat([l.center_param.detach().unsqueeze(0) for l in self.leaf_nodes], 0)

    def leaf_squared_dists(self, batch_data):
        centers = self.leaf_node_centers()
        ta = centers.unsqueeze(0)
        tb = batch_data.detach().unsqueeze(1)
        return (ta - tb).pow(2).sum(2)

    def leaf_prediction(self, batch_data):
        labels = self.leaf_squared_dists(batch_data).argmin(1).int()
        node_id_label_map = self.leaf_node_id_to_col_idx_mapping
        labels_node_id_map = self.col_idx_to_leaf_node_id_mapping
        return labels, labels_node_id_map, node_id_label_map

    def leaf_prediction_np(self, batch_data):
        labels, labels_node_id_map, node_id_label_map = self.leaf_prediction(batch_data)
        labels = labels.int().cpu().numpy()
        return labels, labels_node_id_map, node_id_label_map

    def prediction(self, n_clusters, batch_data):
        # This one is slower but it has the advantage that going from n to n+1 clusters all nodes which
        # remain keep their label and the node splitted assignes its label to the left child and the right child gets the new label
        # This is quite good to compare visualisations of different hierarchy levels
        if 2 < n_clusters > self.n_splits + 1:
            raise RuntimeError(f"level {n_clusters} is not within 2 and {self.n_splits + 1}")

        max_split_level = n_clusters - 1

        leaf_square_dists = self.leaf_squared_dists(batch_data)

        def recursive_assignment(node):
            if node.is_leaf():
                return leaf_square_dists[:, self.leaf_node_id_to_col_idx_mapping[node.node_id]]
            else:
                return torch.min(recursive_assignment(node.left_child), recursive_assignment(node.right_child))

        node_queue = queue.PriorityQueue()
        node_queue.put((1, self.root_node, 0))

        node_sdist_queue = queue.PriorityQueue()
        next_free_column = 1

        while True:
            try:
                next_item = node_queue.get(block=False)
                node = next_item[1]
                column_id = next_item[2]
                if node.split_level < max_split_level:
                    if node.left_child.is_leaf():
                        node_sdist_queue.put(
                            (column_id, recursive_assignment(node.left_child), node.left_child.node_id))
                    else:
                        node_queue.put((node.left_child.split_level, node.left_child, column_id))

                    if node.right_child.is_leaf():
                        node_sdist_queue.put(
                            (next_free_column, recursive_assignment(node.right_child), node.right_child.node_id))
                    else:
                        node_queue.put(
                            (node.right_child.split_level, node.right_child, next_free_column))
                elif node.split_level == max_split_level:
                    node_sdist_queue.put((column_id, recursive_assignment(node.left_child), node.left_child.node_id))
                    node_sdist_queue.put(
                        (next_free_column, recursive_assignment(node.right_child), node.right_child.node_id))
                elif node.split_level > max_split_level:  # this node is still within the split_level node but not its children
                    node_sdist_queue.put((column_id, recursive_assignment(node), node.node_id))
                else:
                    raise RuntimeError(
                        f"Not possible node.split_level: {node.split_level} max_split_level: {max_split_level}")
                next_free_column += 1
            except queue.Empty:
                break

        sdist_node_ids = [node_sdist_queue.get() for i in range(node_sdist_queue.qsize())]
        node_colmuns_id = [v[2] for v in sdist_node_ids]
        node_sdists = [v[1].unsqueeze(1) for v in sdist_node_ids]
        labels = torch.cat(node_sdists, dim=1).argmin(1)
        labels_node_id_map = {idx: n_id for idx, n_id in enumerate(node_colmuns_id)}
        node_id_label_map = {n_id: idx for idx, n_id in enumerate(node_colmuns_id)}
        return labels, labels_node_id_map, node_id_label_map

    def prediction_labels_np(self, n_clusters, batch_data):
        labels, labels_to_node_id, node_id_to_label = self.prediction(n_clusters, batch_data)
        labels = labels.int().cpu().numpy()
        return labels, labels_to_node_id, node_id_to_label

    def predict_tree(self, batch_data, batch_ids):
        labels, _, node_id_label_map = self.leaf_prediction_np(batch_data)

        def recursive(node):
            if node.left_child.is_leaf():
                dp_ids = batch_ids[np.where(labels == node_id_label_map[node.left_child.node_id])[0]]
                left_child = DpLeaf(list(dp_ids), node.left_child.node_id)
            else:
                left_child = recursive(node.left_child)
            if node.right_child.is_leaf():
                dp_ids = batch_ids[np.where(labels == node_id_label_map[node.right_child.node_id])[0]]
                right_child = DpLeaf(list(dp_ids), node.right_child.node_id)
            else:
                right_child = recursive(node.right_child)
            return DpNode(left_child, right_child, node.node_id)

        return recursive(self.root_node)

    def _find_highest_sse_node(self, embedded_data_iterator):
        leaf_sse_accu = {k.node_id: 0.0 for k in self.leaf_nodes}

        for batch_data in embedded_data_iterator:
            labels, _, node_id_label_map = self.leaf_prediction(batch_data)
            one_hot_labels = int_to_one_hot(labels, self.n_leaf_nodes)

            for leaf in self.leaf_nodes:
                assigned_to_node = one_hot_labels[:, node_id_label_map[leaf.node_id]]
                nodes_sses = (assigned_to_node * ((batch_data - leaf.center_param.unsqueeze(0)) ** 2)
                              .sum(1)).sum(0)
                leaf_sse_accu[leaf.node_id] += nodes_sses.item()
        return self.leaf_nodes[self.leaf_node_id_to_col_idx_mapping[max(leaf_sse_accu, key=leaf_sse_accu.get)]]

    def split_highest_sse_node(self):
        highest_sse_node = self._find_highest_sse_node(self.split_dataset_loader_gen())

        leaf_id = highest_sse_node.node_id
        node_data = None
        for batch_data in self.split_dataset_loader_gen():
            labels_np, _, node_id_label_map = self.leaf_prediction_np(batch_data)
            node_label_id = node_id_label_map[leaf_id]
            node_data_batch = batch_data.data.cpu().numpy()[labels_np == node_label_id]
            if node_data is None:
                node_data = node_data_batch
            else:
                node_data = np.concatenate([node_data, node_data_batch], 0)
        init_centers = k_means(node_data, 2, n_init=20)[0]

        new_left_leaf = ECTnodeUnprojected.new_leaf_node(self.next_free_node_id, self, init_centers[0, :])
        new_right_leaf = ECTnodeUnprojected.new_leaf_node(self.next_free_node_id + 1, self, init_centers[1, :])
        highest_sse_node.split_node(self.n_splits + 1, new_left_leaf, new_right_leaf)

        self.next_free_node_id += 2
        self.n_splits += 1

        self.add_module(f"node_{new_left_leaf.node_id}", new_left_leaf)
        self.add_module(f"node_{new_right_leaf.node_id}", new_right_leaf)

        self.leaf_nodes.remove(highest_sse_node)
        self.leaf_nodes.append(new_left_leaf)
        self.leaf_nodes.append(new_right_leaf)
        self._update_leaf_node_mappings()

        self.optimizer.add_param_group({'params': new_left_leaf.parameters()})
        self.optimizer.add_param_group({'params': new_right_leaf.parameters()})

        logger.info(f"new plit now we have {self.n_leaf_nodes} leaves")

    def loss(self, batch_data, is_training):
        labels, _, node_id_label_map = self.leaf_prediction(batch_data)
        one_hot_labels = int_to_one_hot(labels.detach(), self.n_leaf_nodes).detach()
        assignments = {}
        pruning_necessary = False
        self._split_node_center_learning = {}

        def update_dp_counters(node):
            nonlocal pruning_necessary
            nonlocal assignments
            if not node.left_child.is_leaf():
                update_dp_counters(node.left_child)
                assigned_to_left = assignments[node.left_child.node_id]
            else:
                assigned_to_left = one_hot_labels[:, node_id_label_map[node.left_child.node_id]]
                assignments[node.left_child.node_id] = assigned_to_left

            if not node.right_child.is_leaf():
                update_dp_counters(node.right_child)
                assigned_to_right = assignments[node.right_child.node_id]
            else:
                assigned_to_right = one_hot_labels[:, node_id_label_map[node.right_child.node_id]]
                assignments[node.right_child.node_id] = assigned_to_right
            assigned_to_node = assigned_to_left + assigned_to_right
            assignments[node.node_id] = assigned_to_node

            if is_training:
                node._update_counters(assigned_to_left, assigned_to_right)
                if node.child_counter[1] < _prune_threshold or node.child_counter[0] < _prune_threshold:
                    logger.info(
                        f"Pruning necessary because of node {node.node_id} counter: {node.child_counter.cpu().numpy()}")
                    pruning_necessary = True
            node._update_center_buffer()

        update_dp_counters(self.root_node)

        if pruning_necessary:
            self._prune_tree()
            return torch.tensor([0.0], device=self.device), torch.tensor([0.0], device=self.device)
        else:
            center_losses = 0.0
            dp_losses = 0.0

            weights = 0.0

            def recursive(node):
                nonlocal dp_losses
                nonlocal weights
                nonlocal center_losses
                if not node.left_child.is_leaf():
                    recursive(node.left_child)
                if not node.right_child.is_leaf():
                    recursive(node.right_child)
                assigned_to_left = assignments[node.left_child.node_id]
                assigned_to_right = assignments[node.right_child.node_id]
                weights += assigned_to_left + assigned_to_right

                node_loss = node.loss_data_fixed_centers(batch_data, assigned_to_left, assigned_to_right)

                center_node_loss = node.loss_centers_fixed_data(batch_data, assigned_to_left, assigned_to_right)
                dp_losses += node_loss
                center_losses += center_node_loss

            recursive(self.root_node)

            # We also divide here by the number of leaf nodes to
            # dp_loss = torch.sum(dp_losses / (dp_losses.shape[0] * self.n_leaf_nodes))
            # dp_loss = torch.mean(dp_losses / (2 * self.n_leaf_nodes - 1))
            dp_loss = dp_losses / self.n_splits
            center_losses = center_losses / self.n_leaf_nodes
            self._split_node_center_learning = None

            return dp_loss, center_losses

    def _prune_tree(self):
        logger.info("Start pruning the tree")
        n_pruned_nodes = 0

        def prune_nodes(node, parent_node, right_left_indicator):
            nonlocal n_pruned_nodes
            if not node.left_child.is_leaf():
                prune_nodes(node.left_child, node, "left")
            if not node.right_child.is_leaf():
                prune_nodes(node.right_child, node, "right")

            # Root node special treatment
            if parent_node is None:
                if node.child_counter[0] < _prune_threshold:
                    logger.info(f"Left child {node.left_child.node_id} of the root node died")
                    if node.right_child.is_leaf():
                        logger.info("Restart the root node")
                        new_centers = self._create_root_node_centers()
                        rn = self.root_node
                        rn.child_counter.data = torch.tensor([10.0, 10.0], device=self.device)
                        rn.left_child.center_param.data = torch.tensor(new_centers[0, :], device=self.device)
                        rn.right_child.center_param.data = torch.tensor(new_centers[1, :], device=self.device)
                    else:
                        logger.info(f"We replace it with the nodes of the right child")
                        self.leaf_nodes.remove(node.left_child)
                        n_pruned_nodes += 1
                        old_right_child = node.right_child
                        self.pruned_nodes_ids.append(old_right_child.node_id)
                        node.child_counter.data = old_right_child.child_counter.data
                        node.left_child = old_right_child.left_child
                        node.right_child = old_right_child.right_child
                elif node.child_counter[1] < _prune_threshold:
                    logger.info(f"Right child {node.right_child.node_id} of the root node died")
                    if node.left_child.is_leaf():
                        logger.info("Restart the root node")
                        new_centers = self._create_root_node_centers()
                        rn = self.root_node
                        rn.child_counter.data = torch.tensor([10.0, 10.0], device=self.device)
                        rn.left_child.center_param.data = torch.tensor(new_centers[0, :], device=self.device)
                        rn.right_child.center_param.data = torch.tensor(new_centers[1, :], device=self.device)
                    else:
                        logger.info(f"We replace it with the nodes of the left child")
                        self.leaf_nodes.remove(node.right_child)
                        n_pruned_nodes += 1
                        old_left_child = node.left_child
                        self.pruned_nodes_ids.append(old_left_child.node_id)
                        node.child_counter.data = old_left_child.child_counter.data
                        node.left_child = old_left_child.left_child
                        node.right_child = old_left_child.right_child
            elif node.child_counter[0] < _prune_threshold:  # The left node died
                self.leaf_nodes.remove(node.left_child)
                n_pruned_nodes += 1
                logger.info(f"Pruning node {node.node_id} because {node.left_child.node_id} died")
                self.pruned_nodes_ids.append(node.node_id)
                # We replace this node with the right child node
                if right_left_indicator is "left":
                    parent_node.left_child = node.right_child
                elif right_left_indicator is "right":
                    parent_node.right_child = node.right_child
                else:  # this only happens if the node is the root node, we ignore it
                    raise RuntimeError("This cannot happen")
            elif node.child_counter[1] < _prune_threshold:  # The right node died
                self.leaf_nodes.remove(node.right_child)
                n_pruned_nodes += 1
                logger.info(f"Pruning node {node.node_id} because {node.right_child.node_id} died")
                self.pruned_nodes_ids.append(node.node_id)
                # We replace this node with the left child node
                if right_left_indicator is "left":
                    parent_node.left_child = node.left_child
                elif right_left_indicator is "right":
                    parent_node.right_child = node.left_child
                else:  # this only happens if the node is the root node, we ignore it
                    raise RuntimeError("This cannot happen")

        prune_nodes(self.root_node, None, None)

        # Renumber the split order
        split_nodes_pq = queue.PriorityQueue()

        def renumber_split_order(node):
            if not node.is_leaf():
                split_nodes_pq.put((node.split_level, node))
                renumber_split_order(node.left_child)
                renumber_split_order(node.right_child)

        renumber_split_order(self.root_node)
        self.n_splits = 0
        while True:
            try:
                next_split_node = split_nodes_pq.get(block=False)[1]
                self.n_splits += 1
                next_split_node.split_level = self.n_splits
            except queue.Empty:
                break
        # We update the leaf node column id maps
        self._update_leaf_node_mappings()

        # # We split the tree s.t. it has the original number of nodes
        # for _ in range(0, n_pruned_nodes):
        #     self.split_highest_sse_node()

    def cpu(self):
        super.cpu()
        self.device = "cpu"

    def gpu(self, device=None):
        super.gpu(device)
        self.device = device
