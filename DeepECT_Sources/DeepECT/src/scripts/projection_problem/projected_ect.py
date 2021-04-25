import torch.utils.data
from scripts.otherideas.ECT import ECTree
from scripts.projection_problem.common_stuff import *


"""
This script shows the projected version of ECT and what deficiency this version has!
"""



ae_module, pt_data, gold_labels, train_ds, train_loader, ae_reconstruction_loss_fn = init_data_and_ae()

optimizer = torch.optim.Adam(list(ae_module.parameters()), lr=0.001)
embedded_split_data_loader = lambda: map(lambda x: ae_module.forward(x[0])[0],
                                         torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True))
ect_module = ECTree(optimizer, embedded_split_data_loader).cuda()

n_clusters = 2  # nr of clusters we expect
while ect_module.n_leaf_nodes < n_clusters:
    new_variables = ect_module.split_highest_sse_node()
    optimizer.add_param_group({'params': new_variables})

n_rounds = 1000
train_round_idx = 0
while True:  # each iteration is equal to an epoch
    for batch_data in train_loader:
        train_round_idx += 1
        if train_round_idx > n_rounds:
            break
        batch_data = batch_data[0]

        embedded_data, reconstruced_data = ae_module.forward(batch_data)
        ae_loss = ae_reconstruction_loss_fn(batch_data, reconstruced_data)

        dp_loss, center_losses = ect_module.loss(embedded_data, is_training=True)
        loss = dp_loss + center_losses + 0.1 * ae_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if train_round_idx == 1 or train_round_idx == 10 or train_round_idx % 50 == 0 or train_round_idx == 1010:
            plotting_data = pt_data
            embedded_data, reconstruced_data = ae_module.forward(plotting_data)
            ae_loss = ae_reconstruction_loss_fn(plotting_data, reconstruced_data)
            dp_loss, center_losses = ect_module.loss(embedded_data, is_training=False)
            loss = dp_loss + center_losses + 0.1 * ae_loss
            #
            print(train_round_idx,
                  f"total_loss: {loss.item()} ae_loss:{ae_loss.item()} dct_loss: {dp_loss.item()}  {center_losses.item()}")
            embedded_data_np = embedded_data.data.cpu().numpy()

            labels = ect_module.prediction_labels_np(ect_module.n_leaf_nodes, embedded_data)[0]
            plot_data_with_highlight_points(embedded_data_np, labels,
                                            gold_labels, ect_module.centers_for_clusters(ect_module.n_leaf_nodes),
                                            f"Projected ECT emedded data step {train_round_idx}")

        if train_round_idx == 500:
            ect_module.split_highest_sse_node()
            labels = ect_module.prediction_labels_np(ect_module.n_leaf_nodes, embedded_data)[0]
            plot_data_with_highlight_points(embedded_data_np, labels,
                                            gold_labels, ect_module.centers_for_clusters(ect_module.n_leaf_nodes),
                                            f"Projected ECT emedded data step {train_round_idx}")


    else:  # For else is being executed if break did not occur, we continue the while true loop otherwise we break it too
        continue
    break  # Break while loop here
