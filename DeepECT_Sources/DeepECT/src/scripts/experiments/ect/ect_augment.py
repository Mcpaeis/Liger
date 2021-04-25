import torch
import torch.nn.functional as F
import torch.utils.data
from ect.methods.stacked_ae import stacked_ae
from ect.utils.logging_helper import *
from ect.utils.evaluation.dendrogram_purity import *
from scripts.Config import *
import time
import pickle
from ect.utils.deterministic import set_random_seed
from ect.utils.ect_print_utils import *
from ect.methods.ECT_Augment import ECTreeAugment
import logging
from torchvision.transforms import RandomAffine
import PIL
import random
from ect.utils.deterministic import set_random_seed

# Data
# Import data configuration based on the parameters
from scripts.experiments.dataset_configs import *

result_dir = Path(result_main_dir, os.path.basename(__file__)[:-3], dataset_name)
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


def run_experiment(ae_model_path, augmentation_transformer, seed):
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"Working now on {ae_model_path.name}")
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"Seed value for this is: {seed}")
    set_random_seed(seed)

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
    result_file = Path(result_dir, f"results_{dataset_name}.txt")
    result_file_exists = result_file.exists()
    f = open(result_file, "a+")
    if not result_file_exists:
        f.write(
            "#\"ae_model_name\"\t\"NMI\"\t\"ACC\"\t\"Dendrogram_Purity\"\t\"Leaf_Purity\"\t\"(Std)\"\t\"NMI(best-p-tree)\"\t\"ACC(best-p-tree)\"\n")
    f.write(
        f"{ae_model_path.name}\t{nmi_value}\t{acc_value}\t{dp_value}\t{leaf_purity_value}\t{nmi_best_prun_tree}\t{acc_best_prun_tree}\n")
    f.close()


pt_data = torch.from_numpy(data)
pt_split_data = torch.from_numpy(data[split_idx, :])
ae_directory = Path(ae_dir, dataset_name)
check_if_aes_exist(dataset_name)

augmentation_transformer = augmentation_transformer_factory()

# We sort all pre-trained aes and use the same random seed for reproducibility
random.seed(42)
aes_files_seeds = [(x, random.randint(0, 1000)) for x in
                   sorted([x for x in ae_directory.iterdir() if x.name.endswith('.model')], key=lambda x: x.name)]

for ae_model_path, seed in aes_files_seeds:
    run_experiment(ae_model_path, augmentation_transformer, seed)
