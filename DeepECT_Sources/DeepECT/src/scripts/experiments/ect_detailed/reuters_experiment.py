import torch
import torch.nn.functional as F
import torch.utils.data
from ect.methods.stacked_ae import stacked_ae
from ect.utils.logging_helper import *
from ect.utils.evaluation.dendrogram_purity import *
from scripts.Config import *
import time
from scripts.Datasets import *
from pathlib import Path
from ect.utils.ect_print_utils import *
import os
from scripts.otherideas.ECT import ECTree
import numpy as np
import logging
from ect.utils.tree2string import tree2string

lbl_lvl2, lbl_lvl2_names = load_reuters_lvl2_labels()
lvl1_names = lbl_lvl2_names[0:4]
lvl2_names = lbl_lvl2_names[4:]


def create_reuters1_tree_result(dir_path: str, tree: DpNode, pred_labels: np.ndarray, ground_truth: np.ndarray,
                                col_node_id_map):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    f = open(f"{dir_path}/results.txt", "wt")
    f.write(f"NMI: {nmi(pred_labels, ground_truth, average_method='arithmetic')}\n")
    acc, acc_confusion = cluster_acc(pred_labels, ground_truth)
    f.write(f"ACC: {acc} confusion (ground truth \ prediction):\n")
    f.write(f"{acc_confusion}\n")
    f.write("\n\n")
    f.write(f"Dendrogram Purity: {dendrogram_purity(tree, ground_truth)}\n")
    lp = leaf_purity(tree, ground_truth)
    f.write(f"Leaf Purity: Avg:{lp[0]:1.3} std:{lp[1]:1.3}\n")
    col_nod_str = "\t".join([f"{x[0]}:{x[1]}" for x in sorted(list(col_node_id_map.items()), key=lambda x: x[0])])
    f.write(f"{col_nod_str}\n")
    f.write("\n\n")

    tree_str = tree2string(tree, "children", "node_id")
    f.write(f"{tree_str} \n")
    f.write("\n\n")
    f.write(_label_distribution(tree))
    f.close()


def _label_distribution(tree: DpNode):
    node_dist = []

    def recursive(node):
        nonlocal node_dist

        if node.is_leaf:
            dp_ids = list(node.dp_ids)
        else:
            dp_ids_right = recursive(node.right_child)
            dp_ids_left = recursive(node.left_child)
            dp_ids = list(dp_ids_right) + list(dp_ids_left)

        n_dps = len(dp_ids)
        dp_labels_dist = lbl_lvl2[dp_ids, :].sum(axis=0).astype(np.float) / float(n_dps)
        dp_labels_dist_lvl1 = dp_labels_dist[0:4]
        dp_labels_dist_lvl2 = dp_labels_dist[4:]
        unsorted_counts_lvl1 = [(lvl1_names[i], dp_labels_dist_lvl1[i]) for i in range(len(lvl1_names))]
        sorted_counts_lvl2 = sorted([(lvl2_names[i], dp_labels_dist_lvl2[i]) for i in range(len(lvl2_names))],
                                    key=lambda x: -1 * x[1])
        count_lvl1 = "\t".join([f"{name}: {count:.3f}" for name, count in unsorted_counts_lvl1])
        count_lvl2 = "\t".join([f"{name}: {count:.3f}" for name, count in sorted_counts_lvl2])
        node_dist.append((node.node_id,
                          f"{node.node_id:2}:\tn_dps: {n_dps:5} --- Dist-Cat: {count_lvl1}   \t\t Dist-Sub-Cat: {count_lvl2}"))
        return dp_ids

    recursive(tree)
    return "\n".join(map(lambda x: x[1], sorted(node_dist, key=lambda x: x[0])))


dataset_name = "reuters"
data, gold_labels = load_reuters()
split_idx = reuters_split_idx()
n_clusters = 4
n_leaf_nodes_before = 10
n_leaf_nodes_final = 12

result_dir = Path(result_main_dir, os.path.basename(__file__)[:-3])
result_dir.mkdir(parents=True, exist_ok=True)
config_root_logger_file_handler(f'{result_dir}/results_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.log')
config_root_logger_stout()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(42)
torch.manual_seed(np.random.randint(10000))


pt_data = torch.from_numpy(data)
pt_split_data = torch.from_numpy(data[split_idx, :])

ae_model_path = Path(ae_dir, dataset_name, "ae_reuters_5.model")  # with partial splits

train = torch.utils.data.TensorDataset(pt_data)
train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)

n_features = data.shape[1]
ae_reconstruction_loss_fn = lambda x, y: torch.mean((x - y) ** 2)
ae_module = stacked_ae(n_features, [500, 500, 2000, 10],
                       weight_initalizer=torch.nn.init.xavier_normal_,
                       activation_fn=lambda x: F.relu(x),
                       loss_fn=None,
                       optimizer_fn=None)

model_data = torch.load(ae_model_path, map_location='cpu')
ae_module.load_state_dict(model_data)
ae_module = ae_module.cuda()

optimizer = torch.optim.Adam(list(ae_module.parameters()), lr=0.0001)

embedded_split_data_loader = lambda: map(lambda x: ae_module.forward(x.cuda())[0],
                                         torch.utils.data.DataLoader(pt_split_data, batch_size=256, shuffle=True))

cluster_module = ECTree(optimizer, embedded_split_data_loader).cuda()

while cluster_module.n_leaf_nodes < n_leaf_nodes_before:
    cluster_module.split_highest_sse_node()


def evaluate(train_round_idx, ae_module, cluster_module):
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pt_data, torch.tensor(range(pt_data.shape[0]))), batch_size=256)

    pred_labels = np.zeros(pt_data.shape[0], dtype=np.int)
    pred_tree = None
    index = 0
    n_batches = 0
    print("start evaluation")
    for batch_data_id in test_loader:
        batch_data, batch_ids = batch_data_id
        batch_data = batch_data.cuda()
        n_batches += 1
        batch_size = batch_data.shape[0]
        embeded_data, reconstructed_data = ae_module.forward(batch_data)
        labels = cluster_module.prediction_labels_np(min(cluster_module.n_leaf_nodes, n_clusters), embeded_data)[0]
        pred_labels[index: index + batch_size] = labels
        new_pred_tree = cluster_module.predict_tree(embeded_data, batch_ids)
        if pred_tree is None:
            pred_tree = new_pred_tree
        else:
            pred_tree = combine_to_trees(pred_tree, new_pred_tree)
        index = index + batch_size

    lp = leaf_purity(pred_tree, gold_labels)
    nmi_value = nmi(gold_labels, pred_labels, average_method='arithmetic')
    acc_value = cluster_acc(gold_labels, pred_labels)[0]
    dp_value = dendrogram_purity(pred_tree, gold_labels)
    leaf_purity_value = f"{lp[0]:1.3}\t({lp[1]:1.3})"
    logger.info(
        f"{train_round_idx}  leaf_purity: {leaf_purity_value}, purity: {dp_value}, NMI: {nmi_value} ACC: {acc_value}")
    return nmi_value, acc_value, dp_value, leaf_purity_value


evaluate("init", ae_module, cluster_module)

n_rounds = 50000
train_round_idx = 0
while True:  # each iteration is equal to an epoch
    for batch_data in train_loader:
        train_round_idx += 1
        if train_round_idx > n_rounds:
            break
        batch_data = batch_data[0].cuda()

        if train_round_idx % 500 == 0 and cluster_module.n_leaf_nodes < n_leaf_nodes_final:
            cluster_module.split_highest_sse_node()

        embedded_data, reconstruced_data = ae_module.forward(batch_data)
        ae_loss = ae_reconstruction_loss_fn(batch_data, reconstruced_data)

        dp_loss, center_losses = cluster_module.loss(embedded_data, is_training=True)

        total_loss = dp_loss + center_losses + ae_loss

        if train_round_idx <= 10 or train_round_idx % 100 == 0:
            logger.info(
                f"{train_round_idx} - loss in this batch: dp_loss:{dp_loss.item()} center_losses:{center_losses.item()} ae_loss:{ae_loss.item()} total_loss: {total_loss.item()}")

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if train_round_idx % 2000 == 0:
            evaluate(train_round_idx, ae_module, cluster_module)

    else:  # For else is being executed if break did not occur, we continue the while true loop otherwise we break it too
        continue
    break  # Break while loop here


def create_tree_data(dir_name, ae_module, cluster_module):
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pt_data, torch.tensor(range(pt_data.shape[0]))), batch_size=256)

    pred_labels = np.zeros(pt_data.shape[0], dtype=np.int)
    pred_tree = None
    index = 0
    n_batches = 0
    col_node_id_map = None
    print("start result printing")
    for batch_data_id in test_loader:
        batch_data, batch_ids = batch_data_id
        batch_data = batch_data.cuda()
        n_batches += 1
        batch_size = batch_data.shape[0]
        embeded_data, reconstructed_data = ae_module.forward(batch_data)
        labels, labels_to_node_id, _ = cluster_module.prediction_labels_np(
            min(cluster_module.n_leaf_nodes, 4), embeded_data)
        col_node_id_map = labels_to_node_id
        pred_labels[index: index + batch_size] = labels
        new_pred_tree = cluster_module.predict_tree(embeded_data, batch_ids)
        if pred_tree is None:
            pred_tree = new_pred_tree
        else:
            pred_tree = combine_to_trees(pred_tree, new_pred_tree)
        index = index + batch_size

    create_reuters1_tree_result(dir_name, pred_tree, pred_labels, gold_labels, col_node_id_map)


create_tree_data(result_dir, ae_module, cluster_module)
