# Data
from scripts.Datasets import *
import sys

if len(sys.argv) < 2:
    raise RuntimeError("please set the dataset name")

dataset_name = sys.argv[1]  # Or set it directly

if dataset_name == "mnist":
    # MNIST
    data, gold_labels = load_mnist()
    split_idx = mnist_split_idx()
    n_clusters = 10
    n_leaf_nodes_final = 20
    image_wh = 28
    image_min_value = 0.0
    augmentation_config = ((-15, 15), (0.08, 0.08))  #0.08 is the translation by 2 pixels for mnist

elif dataset_name == "usps":
    # USPS
    dataset_name = "usps"
    data, gold_labels = load_usps()
    split_idx = usps_split_idx()
    n_clusters = 10
    n_leaf_nodes_final = 20
    image_wh = 16
    image_min_value = -0.9999999
    augmentation_config = ((-15, 15), (0.14, 0.14)) #0.14 is the translation by 2 pixels for mnist

elif dataset_name == "reuters":
    # Reuters
    data, gold_labels = load_reuters()
    split_idx = reuters_split_idx()
    n_clusters = 4
    n_leaf_nodes_final = 12

elif dataset_name == "fashion-mnist":
    # Fashion Mnist
    data, gold_labels = load_fashion_mnist()
    split_idx = mnist_split_idx()
    n_clusters = 10
    n_leaf_nodes_final = 20
    image_wh = 28
    image_min_value = 0.0
    augmentation_config = ((-15, 15), (0.08, 0.08)) #0.08 is the translation by 2 pixels for mnist

else:
    raise RuntimeError("Unkown dataset name")
