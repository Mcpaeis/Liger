import numpy as np
from scipy.io import loadmat
import os
from scripts.Config import dataset_dir
from scipy import io
from pathlib import Path
from imageio import imread
import torchvision


def load_usps():
    # Can be downloaded here: https://github.com/marionmari/Graph_stuff/tree/master/usps_digit_data
    file_path = Path(f"{dataset_dir}/usps_resampled.mat")

    if not file_path.exists():
        import requests
        response = requests.get(
            "https://github.com/marionmari/Graph_stuff/blob/master/usps_digit_data/usps_resampled.mat?raw=true")
        f = open(file_path, 'wb+')
        f.write(response.content)
        f.close()

    data_mat = loadmat(file_path)

    data = np.concatenate([data_mat['train_patterns'].T, data_mat['test_patterns'].T], 0)
    labels = np.argmax(np.concatenate([data_mat['train_labels'].T, data_mat['test_labels'].T], 0), 1)
    return data.astype(np.float32), labels.astype(np.int32)


def load_mnist():
    file_path = Path(f"{dataset_dir}/mnist-original.mat")
    if not file_path.exists():
        import requests
        response = requests.get(
            "https://github.com/amplab/datascience-sp14/blob/master/lab7/mldata/mnist-original.mat?raw=true")
        f = open(file_path, 'wb+')
        f.write(response.content)
        f.close()

    with open(file_path, 'rb') as matlab_file:
        matlab_dict = io.loadmat(matlab_file, struct_as_record=True)

    data = matlab_dict['data'].T.astype(np.float32)
    label = np.squeeze(matlab_dict['label'], 0).astype(np.int32)

    data = data * 0.02  # This preprocessing is done in the DEC implementation, as well
    return data, label


def load_reuters():
    if not Path(f'{dataset_dir}/Reuters/reuters_preprocessed_data.npy').exists():
        raise RuntimeError("Preprocessed Reuters data seems to be missing. Please run preprocess_reuters.py first.")
    data = np.load(f'{dataset_dir}/Reuters/reuters_preprocessed_data.npy')
    data = data * 100.0  # Similar to preprocessing done in DEC implementation
    label = np.load(f'{dataset_dir}/Reuters/reuters_preprocessed_target.npy')
    return data, label


def load_reuters_lvl2_labels():  # Contains level 1 and level 2 categories
    if not Path(f'{dataset_dir}/Reuters/reuters_preprocessed_data.npy').exists():
        raise RuntimeError("Preprocessed Reuters data seems to be missing. Please run preprocess_reuters.py first.")
    cat_names = np.load(f'{dataset_dir}/Reuters/reuters_preprocessed_cat_names_2lvls.npy')
    labels = np.load(f'{dataset_dir}/Reuters/reuters_preprocessed_labels_2lvls.npy')
    return labels, cat_names


def load_fashion_mnist():
    dataset_train = torchvision.datasets.FashionMNIST(dataset_dir, train=True, download=True)
    dataset_test = torchvision.datasets.FashionMNIST(dataset_dir, train=False, download=True)
    data_train = (dataset_train.data.numpy() / 255.0).astype(np.float32)
    data_test = (dataset_test.data.numpy() / 255.0).astype(np.float32)
    labels_train = dataset_train.targets.numpy()
    labels_test = dataset_test.targets.numpy()
    return np.concatenate((data_train, data_test), 0).reshape((70000, -1)), np.concatenate((labels_train, labels_test),
                                                                                           0)


def coil20_split_idx():
    return range(0, 1440)  # Equal to the training data


def reuters_split_idx():
    return range(0, 80000)


def mnist_split_idx():
    return range(0, 70000)  # Equal to the training data


def usps_split_idx():
    return range(0, 9298)  # Equal to the training data


def shuffle_dataset(data, labels):
    shuffled_indices = np.random.permutation(len(data))
    shuffled_x = data[shuffled_indices, :]
    if labels is not None:
        shuffled_y = labels[shuffled_indices]
        return shuffled_x, shuffled_y
    else:
        return shuffled_x
