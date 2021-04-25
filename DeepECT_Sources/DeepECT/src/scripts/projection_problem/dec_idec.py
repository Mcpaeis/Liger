import torch.utils.data
from sklearn.cluster.k_means_ import k_means
from ect.methods.DEC import DEC
from scripts.Config import *
from scripts.projection_problem.common_stuff import *

ae_module, pt_data, gold_labels, _,train_loader, ae_reconstruction_loss_fn = init_data_and_ae()

embedded_data_np = ae_module.encode(pt_data).detach().cpu().numpy()

dec_module = DEC(k_means(embedded_data_np, 2)[0]).cuda()

optimizer = torch.optim.Adam(list(ae_module.parameters()) + list(dec_module.parameters()), lr=0.001)

gamma = 0.1  # Put 0.0 here for pure DEC

n_rounds = 2000
train_round_idx = 0
while True:  # each iteration is equal to an epoch
    for batch_data in train_loader:
        train_round_idx += 1
        if train_round_idx > n_rounds:
            break
        batch_data = batch_data[0]

        embedded_data, reconstruced_data = ae_module.forward(batch_data)
        ae_loss = ae_reconstruction_loss_fn(batch_data, reconstruced_data)

        dec_loss = dec_module.loss_dec_compression(embedded_data)
        loss = dec_loss + gamma * ae_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if train_round_idx == 1 or train_round_idx == 10 or train_round_idx % 50 == 0:
            plotting_data = pt_data
            embedded_data, reconstruced_data = ae_module.forward(plotting_data)
            ae_loss = ae_reconstruction_loss_fn(plotting_data, reconstruced_data)
            dec_loss = dec_module.loss_dec_compression(embedded_data)
            loss = dec_loss + gamma * ae_loss
            #
            print(train_round_idx,
                  f"total_loss: {loss.item()} ae_loss:{ae_loss.item()} dec_loss: {dec_loss.item()}")
            embedded_data_np = embedded_data.data.cpu().numpy()

            labels = dec_module.prediction_hard(embedded_data).detach().cpu().numpy()
            plot_data_with_highlight_points(embedded_data_np, labels,
                                            gold_labels, dec_module.centers.detach().cpu().numpy(),
                                            f"IDEC emedded data step {train_round_idx}")

    else:  # For else is being executed if break did not occur, we continue the while true loop otherwise we break it too
        continue
    break  # Break while loop here
