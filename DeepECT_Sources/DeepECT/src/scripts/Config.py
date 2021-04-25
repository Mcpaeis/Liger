from pathlib import Path
import os

dataset_dir = None  # Path to a directory containing the datasets
result_main_dir = None  # Path to a directory for results
ae_dir = None  # Path to a directory in which we save the pre-trained autoencoders



# dataset_dir = Path(Path.home(),"tmp/ect_test/datasets")
# result_main_dir = Path(Path.home(),"tmp/ect_test/results")
# ae_dir = Path(Path.home(),"tmp/ect_test/pretrained_aes")





if dataset_dir is None or result_main_dir is None or ae_dir is None:
    raise RuntimeError("Configure the paths for dataset_dir, result_main_dir and ae_dir")


def check_if_aes_exist(dataset_name):
    ae_directory = Path(ae_dir, dataset_name)
    if not ae_directory.exists() or len(list(ae_directory.iterdir())) == 0:
        raise RuntimeError(
            f"No pre-trained autoencoders for {dataset_name}! Please run pretrain_ae.py for this dataset.")
