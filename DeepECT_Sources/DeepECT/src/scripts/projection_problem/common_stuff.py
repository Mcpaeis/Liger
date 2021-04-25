import torch
import torch.nn.functional as F
from sklearn import preprocessing
from pathlib import Path
import torch.utils.data
import numpy as np
from sklearn.datasets import make_blobs
from ect.methods.stacked_ae import stacked_ae
from scripts.Config import *
from matplotlib import pyplot as plt

colors = ("red", "green", "blue")
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4",
           "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]


def plot_data(data, labels, title):
    n_dp = data.shape[0]
    for i in range(0, n_dp):
        color = colors[labels[i]]
        plt.scatter(data[i, 0], data[i, 1], s=20, c=color)
        plt.title(title)
    plt.show()


def plot_data_with_highlight_points(data, color_labels, shape_labels, highlight, title):
    n_dp = data.shape[0]
    for i in range(0, n_dp):
        color = colors[color_labels[i]]
        marker = markers[shape_labels[i] + 2]
        plt.scatter(data[i, 0], data[i, 1], s=20, marker=marker, c=color)
        plt.title(title)
    plt.scatter(highlight[:, 0], highlight[:, 1], marker='x', s=100, c='black')
    plt.show()


def init_data_and_ae():
    ae_path = Path(Path(__file__).parent, "ae.model")

    data, gold_labels = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)

    min_max_scaler = preprocessing.MinMaxScaler((0.01, 0.99))
    data = np.float32(min_max_scaler.fit_transform(data))
    n_features = data.shape[1]
    pt_data = torch.from_numpy(data).cuda()
    train_ds = torch.utils.data.TensorDataset(pt_data)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    plot_data(data, gold_labels, "Original data")

    ae_reconstruction_loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    ae_module = stacked_ae(n_features, [50, 50, 200, 2],
                           weight_initalizer=torch.nn.init.xavier_normal_,
                           activation_fn=lambda x: F.leaky_relu(x),
                           loss_fn=ae_reconstruction_loss_fn,
                           optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.001))

    if ae_path.exists():
        model_data = torch.load(ae_path, map_location='cpu')
        ae_module.load_state_dict(model_data)
        ae_module = ae_module.cuda()
    else:
        print(
            "Warning we train a new AE, because we did not find the preexisting one, this could generate a diffrent result to the paper")
        ae_module = ae_module.cuda()
        ae_module.pretrain(train_loader, 1000)
        ae_module.refine_training(train_loader, 5000)
        torch.save(ae_module.state_dict(), ae_path)

    embedded_data = ae_module.forward(pt_data)[0]
    embedded_data_np = embedded_data.data.cpu().numpy()
    plot_data(embedded_data_np, gold_labels, "Emedded data initial")

    return ae_module, pt_data, gold_labels, train_ds, train_loader, ae_reconstruction_loss_fn
