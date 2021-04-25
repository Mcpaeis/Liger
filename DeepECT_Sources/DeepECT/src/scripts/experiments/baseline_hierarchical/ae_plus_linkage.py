import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.cluster.k_means_ import k_means
from ect.methods.stacked_ae import stacked_ae
from ect.utils.logging_helper import *
from ect.utils.evaluation.dendrogram_purity import *
from scripts.Config import *
import time
from scripts.Datasets import *
from ect.methods.DEC import DEC
from pathlib import Path
from ect.utils.evaluation import cluster_acc
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster.hierarchical import AgglomerativeClustering
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
random.seed(42)

if data.shape[0] > 80000:
    data, gold_labels = shuffle_dataset(data,gold_labels)
    agglo_idx = np.random.choice(data.shape[0], 80000, replace=False)
    data = data[agglo_idx, :]
    gold_labels = gold_labels[agglo_idx]

pt_data = torch.from_numpy(data)

ae_directory = Path(ae_dir, dataset_name)
check_if_aes_exist(dataset_name)

# We sort all pre-trained aes and use the same random seed for reproducibility
aes_files = sorted([x for x in ae_directory.iterdir() if x.name.endswith('.model')], key=lambda x: x.name)
del data


def run_experiment(ae_model_path):
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"Working now on {ae_model_path.name}")
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    new_seed = random.randint(0, 1000)
    logger.info(f"Seed value for this is: {new_seed}")
    set_random_seed(new_seed)

    ae_module = stacked_ae(pt_data.shape[1], [500, 500, 2000, 10],
                           weight_initalizer=torch.nn.init.xavier_normal_,
                           activation_fn=lambda x: F.relu(x),
                           loss_fn=None,
                           optimizer_fn=None)

    model_data = torch.load(ae_model_path, map_location='cpu')
    ae_module.load_state_dict(model_data)
    ae_module = ae_module.cuda()

    # Get embedded data
    embedded_data = None
    for batch_data in torch.utils.data.DataLoader(pt_data, batch_size=256, shuffle=False):
        embedded_batch_np = ae_module.forward(batch_data.cuda())[0].detach().cpu().numpy()
        if embedded_data is None:
            embedded_data = embedded_batch_np
        else:
            embedded_data = np.concatenate([embedded_data, embedded_batch_np], 0)
    del ae_module

    sl_cl = AgglomerativeClustering(compute_full_tree=True, n_clusters=n_clusters, linkage="single").fit(embedded_data)
    sl_labels = sl_cl.labels_
    sl_purity_tree = prune_dendrogram_purity_tree(to_dendrogram_purity_tree(sl_cl.children_), n_leaf_nodes_final)
    sl_nmi = nmi(gold_labels, sl_labels, average_method='arithmetic')
    sl_acc = cluster_acc(sl_labels, gold_labels)[0]
    sl_purity = dendrogram_purity(sl_purity_tree, gold_labels)
    sl_lp = leaf_purity(sl_purity_tree, gold_labels)
    sl_leaf_purity_value = f"{sl_lp[0]:1.3}\t({sl_lp[1]:1.3})"

    result_file_sl = Path(f"{result_dir}/results_ae_agglo_single_{dataset_name}.txt")
    result_file_sl_exists = result_file_sl.exists()
    f = open(result_file_sl, "a+")
    if not result_file_sl_exists:
        f.write("#\"ae_model_name\"\t\"NMI\"\t\"ACC\"\t\"Dendrogram_Purity\"\t\"Leaf_Purity\t(Std)\"\n")
    f.write(f"{ae_model_path.name}\t{sl_nmi}\t{sl_acc}\t{sl_purity}\t{sl_leaf_purity_value}\n")
    f.close()
    del sl_cl, sl_labels, sl_purity_tree

    cl_cl = AgglomerativeClustering(compute_full_tree=True, n_clusters=n_clusters, linkage="complete").fit(
        embedded_data)
    cl_labels = cl_cl.labels_
    cl_purity_tree = prune_dendrogram_purity_tree(to_dendrogram_purity_tree(cl_cl.children_), n_leaf_nodes_final)
    cl_nmi = nmi(gold_labels, cl_labels, average_method='arithmetic')
    cl_acc = cluster_acc(cl_labels, gold_labels)[0]
    cl_purity = dendrogram_purity(cl_purity_tree, gold_labels)
    cl_lp = leaf_purity(cl_purity_tree, gold_labels)
    cl_leaf_purity_value = f"{cl_lp[0]:1.3}\t({cl_lp[1]:1.3})"


    result_file_cl = Path(f"{result_dir}/results_ae_agglo_complete_{dataset_name}.txt",)
    result_file_cl_exists = result_file_cl.exists()
    f = open(result_file_cl, "a+")
    if not result_file_cl_exists:
        f.write("#\"ae_model_name\"\t\"NMI\"\t\"ACC\"\t\"Dendrogram_Purity\"\t\"Leaf_Purity\t(Std)\"\n")
    f.write(f"{ae_model_path.name}\t{cl_nmi}\t{cl_acc}\t{cl_purity}\t{cl_leaf_purity_value}\n")
    f.close()
    del cl_cl, cl_labels, cl_purity_tree

for ae_model_path in aes_files:
    run_experiment(ae_model_path)
