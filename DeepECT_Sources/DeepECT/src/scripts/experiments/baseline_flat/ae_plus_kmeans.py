import torch
import torch.nn.functional as F
import torch.utils.data
from ect.methods.stacked_ae import stacked_ae
from ect.utils.logging_helper import *
import logging
from ect.utils.evaluation.dendrogram_purity import *
from scripts.Config import *
import time
from pathlib import Path
import numpy as np
from ect.utils.evaluation import cluster_acc
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster.k_means_ import k_means
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

pt_data = torch.from_numpy(data)

ae_directory = Path(ae_dir, dataset_name)

#We sort all pre-trained aes for reproducability
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

    # Perform k-means
    k_means_labels = k_means(embedded_data, n_clusters, n_init=20)[1]

    k_means_nmi_value = nmi(gold_labels, k_means_labels, average_method='arithmetic')
    k_means_acc_value = cluster_acc(gold_labels, k_means_labels)[0]

    result_file = Path(f"{result_dir}/results_ae_kmeans_{dataset_name}.txt")
    result_file_exists = result_file.exists()
    f = open(result_file, "a+")
    if not result_file_exists:
        f.write("#\"ae_model_name\"\t\"NMI\"\t\"ACC\"\n")
    f.write(f"{ae_model_path.name}\t{k_means_nmi_value}\t{k_means_acc_value}\n")
    f.close()


for ae_model_path in aes_files:
    run_experiment(ae_model_path)
