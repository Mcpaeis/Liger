import torch.nn.functional as F
import torch
import torch.utils.data
from ect.methods.stacked_ae import stacked_ae
from scripts.Datasets import *
from scripts.Config import *
import logging
from ect.utils.logging_helper import *
import time
import os


# Data
# Import data configuration based on the parameters
from scripts.experiments.dataset_configs import *


result_dir = f"{result_main_dir}/{os.path.basename(__file__)[:-3]}/{dataset_name}"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
config_root_logger_file_handler(f'{result_dir}/pretrain_ae_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.log')
config_root_logger_stout()

ds_ae_dir = f"{ae_dir}/{dataset_name}"
if not os.path.exists(ds_ae_dir):
    os.makedirs(ds_ae_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

n_features = data.shape[1]
pt_data = torch.from_numpy(data)

# Autoencoder Layout:
ae_layout = [500, 500, 2000, 10]

# Reconstruction Error
loss_fn = lambda x, y: torch.mean((x - y) ** 2)

# Nr of Training steps per layer
steps_per_layer = 20000
# Nr of Finetuning
refine_training_steps = 50000

def get_total_loss():
    total_loss = 0.0
    for batch in train_loader:
        batch = batch[0].cuda()
        total_loss += loss_fn(batch, ae.forward(batch)[1]).item()
    return total_loss


for index in range(0, 10):
    logging.info(f"Start training ae {index}")

    train = torch.utils.data.TensorDataset(pt_data)
    train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, pin_memory=True)

    # Original DEC paper AE
    ae = stacked_ae(n_features, ae_layout,
                    weight_initalizer=torch.nn.init.xavier_normal_,
                    activation_fn=lambda x: F.relu(x),
                    loss_fn=loss_fn,
                    optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.0001)).cuda()


    def add_noise(batch):
        mask = torch.empty(batch.shape, device=batch.device).bernoulli_(0.8)
        return batch * mask


    ae.pretrain(train_loader, rounds_per_layer=steps_per_layer, dropout_rate=0.2, corruption_fn=add_noise)

    logging.info(f"Complete data loss after pretraining {get_total_loss()}")

    ae.refine_training(train_loader, refine_training_steps, corruption_fn=add_noise)

    total_loss = get_total_loss()
    logging.info(f"Complete data loss after fine tuning {total_loss}")

    total_loss_str = f"{total_loss}".replace('.', '_')
    torch.save(ae.state_dict(), f"{ds_ae_dir}/ae_{dataset_name}_{index}.model")
    logging.info("saved model")
    del ae
