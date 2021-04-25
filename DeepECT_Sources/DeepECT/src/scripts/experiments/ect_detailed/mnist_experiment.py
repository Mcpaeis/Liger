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
import pickle
from ect.methods.ECT_Augment import ECTreeAugment
import numpy as np
import logging
from torchvision.transforms import RandomAffine
import PIL
import random
from ect.utils.deterministic import set_random_seed

data, gold_labels = load_mnist()
split_idx = mnist_split_idx()
n_clusters = 10
n_leaf_nodes_final = 20
image_wh = 28
image_min_value = 0.0
augmentation_config = ((-15, 15), (0.08, 0.08))  # 0.08 is the translation by 2 pixels for mnist

result_dir = Path(result_main_dir, os.path.basename(__file__)[:-3])
result_dir.mkdir(parents=True, exist_ok=True)
config_root_logger_file_handler(f'{result_dir}/results_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.log')
config_root_logger_stout()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def augmentation_transformer_factory():
    degrees, translation = augmentation_config
    # For some reasons fillcolor does not work therefore we substract the min value before and add it after
    r = RandomAffine(degrees=degrees, shear=degrees, translate=translation, resample=PIL.Image.BILINEAR)

    def transformer(data):
        target = data.detach().cpu().numpy().copy() - image_min_value
        for i in range(0, data.shape[0]):
            image = target[i, :].reshape((image_wh, image_wh))
            transformed = r(PIL.Image.fromarray(image))
            transformed = np.array(transformed)
            target[i, :] = transformed.reshape((1, image_wh * image_wh))
        return torch.from_numpy(target + image_min_value).to(data.device)

    return transformer


augmentation_transformer = augmentation_transformer_factory()

np.random.seed(42)
torch.manual_seed(np.random.randint(10000))

pt_data = torch.from_numpy(data)
pt_split_data = torch.from_numpy(data[split_idx, :])

ae_model_path = Path(f"{ae_dir}/mnist/ae_mnist_9.model")

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

cluster_module = ECTreeAugment(optimizer, embedded_split_data_loader).cuda()


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
    nmi_best_prun_tree = as_flat_clustering_pruned_for_highest_measure(pred_tree, n_clusters, gold_labels,
                                                                       lambda x, y: nmi(x, y, 'arithmetic'))
    acc_best_prun_tree = as_flat_clustering_pruned_for_highest_measure(pred_tree, n_clusters, gold_labels,
                                                                       lambda x, y: cluster_acc(x, y)[0])
    nmi_value = nmi(gold_labels, pred_labels, average_method='arithmetic')
    acc_value = cluster_acc(gold_labels, pred_labels)[0]
    dp_value = dendrogram_purity(pred_tree, gold_labels)
    leaf_purity_value = f"{lp[0]:1.3}\t({lp[1]:1.3})"
    logger.info(
        f"{train_round_idx}  leaf_purity: {leaf_purity_value}, D-purity: {dp_value}, NMI: {nmi_value} ACC: {acc_value}  NMI(best p-tree): {nmi_best_prun_tree} ACC (best p-tree): {acc_best_prun_tree}")
    return nmi_value, acc_value, dp_value, leaf_purity_value, nmi_best_prun_tree, acc_best_prun_tree


evaluate("init", ae_module, cluster_module)

n_rounds = 40000
train_round_idx = 0
while True:  # each iteration is equal to an epoch
    for batch_data in train_loader:
        train_round_idx += 1
        if train_round_idx > n_rounds:
            break
        batch_data = batch_data[0]
        batch_data_augmented = augmentation_transformer(batch_data)
        batch_data = batch_data.cuda()
        batch_data_augmented = batch_data_augmented.cuda()

        if train_round_idx % 500 == 0 and cluster_module.n_leaf_nodes < n_leaf_nodes_final:
            cluster_module.split_highest_sse_node()

        embedded_data, reconstruced_data = ae_module.forward(batch_data)
        embedded_data_aug, reconstruced_data_aug = ae_module.forward(batch_data_augmented)

        ae_loss = ae_reconstruction_loss_fn(batch_data, reconstruced_data) + ae_reconstruction_loss_fn(
            batch_data_augmented, reconstruced_data_aug)

        dp_loss, center_losses, augmentation_loss = cluster_module.loss(embedded_data, embedded_data_aug,
                                                                        is_training=True)

        total_loss = dp_loss + center_losses + ae_loss + augmentation_loss

        if train_round_idx <= 10 or train_round_idx % 100 == 0:
            logger.info(
                f"{train_round_idx} - loss in this batch: dp_loss:{dp_loss.item()} "
                f"center_losses:{center_losses.item()} ae_loss:{ae_loss.item()} augmentation_loss: {augmentation_loss.item()} total_loss: {total_loss.item()}")

            # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if train_round_idx % 2000 == 0:
            evaluate(train_round_idx, ae_module, cluster_module)

    else:  # For else is being executed if break did not occur, we continue the while true loop otherwise we break it too
        continue
    break  # Break while loop here

# Write last evaluation
nmi_value, acc_value, dp_value, leaf_purity_value, nmi_best_prun_tree, acc_best_prun_tree = evaluate("", ae_module,
                                                                                                     cluster_module)
result_file = Path(result_dir, f"results.txt")
result_file_exists = result_file.exists()
f = open(result_file, "a+")
if not result_file_exists:
    f.write(
        "#\"ae_model_name\"\t\"NMI\"\t\"ACC\"\t\"Dendrogram_Purity\"\t\"Leaf_Purity\"\t\"(Std)\"\t\"NMI(best-p-tree)\"\t\"ACC(best-p-tree)\"\n")
f.write(
    f"{ae_model_path.name}\t{nmi_value}\t{acc_value}\t{dp_value}\t{leaf_purity_value}\t{nmi_best_prun_tree}\t{acc_best_prun_tree}\n")
f.close()

 # Write last evaluation
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(pt_data, torch.tensor(range(pt_data.shape[0]))), batch_size=256)

pred_labels = np.zeros(pt_data.shape[0], dtype=np.int)
pred_tree = None
index = 0
n_batches = 0
col_node_id_map = None
all_embedded_data_np = None
all_labels = None
print("start result printing")
for batch_data_id in test_loader:
    batch_data, batch_ids = batch_data_id
    batch_data = batch_data.cuda()
    n_batches += 1
    batch_size = batch_data.shape[0]
    embeded_data, reconstructed_data = ae_module.forward(batch_data)

    if all_embedded_data_np is None:
        all_labels = gold_labels[batch_ids.detach().cpu().numpy()]
        all_embedded_data_np = embeded_data.detach().cpu().numpy()
    else:
        all_labels = np.concatenate([all_labels, gold_labels[batch_ids.detach().cpu().numpy()]], axis=0)
        all_embedded_data_np = np.concatenate([all_embedded_data_np, embeded_data.detach().cpu().numpy()], axis=0)

    labels, labels_to_node_id, _ = cluster_module.prediction_labels_np(min(cluster_module.n_leaf_nodes, 10),
                                                                       embeded_data)
    col_node_id_map = labels_to_node_id
    pred_labels[index: index + batch_size] = labels
    new_pred_tree = cluster_module.predict_tree(embeded_data, batch_ids)
    if pred_tree is None:
        pred_tree = new_pred_tree
    else:
        pred_tree = combine_to_trees(pred_tree, new_pred_tree)
    index = index + batch_size

np.save(f'{result_dir}/embedded_data.npy', all_embedded_data_np)
np.save(f'{result_dir}/embedded_gt.npy', all_labels)
pickle.dump(pred_tree, open(f'{result_dir}/tree.pred', "wb"))
create_tree_result(result_dir, pred_tree, pred_labels, gold_labels, col_node_id_map)
create_images_for_leaf_nodes(result_dir, pred_tree, data, image_wh)
create_dot_png(pred_tree, gold_labels, result_dir)


