import numpy as np
import math
import os
from scipy import io
import PIL
from sklearn.metrics import normalized_mutual_info_score as nmi
from ect.utils.evaluation import cluster_acc
from ect.utils.evaluation.dendrogram_purity import DpNode, dendrogram_purity, leaf_purity
from ect.utils.tree2string import tree2string



def _label_distribution(tree: DpNode, ground_truth: np.ndarray):
    node_dist = []

    def recursive(node):
        nonlocal node_dist

        if node.is_leaf:
            dp_ids = list(node.dp_ids)
        else:
            dp_ids_right = recursive(node.right_child)
            dp_ids_left = recursive(node.left_child)
            dp_ids = list(dp_ids_right) + list(dp_ids_left)
        dp_labels = [ground_truth[dpid] for dpid in dp_ids]
        n_dps = len(dp_ids)
        count = "   ".join([f"{i}: {count / n_dps:.3f}" for i, count in
                            sorted(enumerate(np.bincount(dp_labels)), key=lambda x: -1 * x[1])])
        node_dist.append((node.node_id, f"{node.node_id:2}:\tn_dps: {n_dps:5} --- Dist: {count} "))
        return dp_ids

    recursive(tree)
    return "\n".join(map(lambda x: x[1], sorted(node_dist, key=lambda x: x[0])))


def create_images_for_leaf_nodes(result_dir: str, tree: DpNode, data: np.ndarray, img_size):
    rnd = np.random.RandomState(42)
    def recursive(node):
        if node.is_leaf:
            dp_ids = list(node.dp_ids)
        else:
            dp_ids_right = recursive(node.right_child)
            dp_ids_left = recursive(node.left_child)
            dp_ids = dp_ids_right + dp_ids_left
        n_dps = len(dp_ids)

        if n_dps > 0:
            if n_dps < 1000:
                rows = 10
                cols = int(math.ceil(n_dps / 10))
                plot_ids = dp_ids
            elif n_dps <= 10000:
                rows = 100
                cols = int(math.ceil(n_dps / 100))
                plot_ids = dp_ids
            else:
                plot_ids = rnd.choice(dp_ids, 10000, replace=False)
                rows = 100
                cols = 100
            plot_ids = sorted(plot_ids)
            img = create_image_panel(data[plot_ids, :], rows, cols, img_size)
            img = img - np.min(img)
            img = img / np.max(img)
            img = img * 255
            img = PIL.Image.fromarray(img)
            img = img.convert("L")
            img.save(f"{result_dir}/samples_node_{node.node_id}.png")
        return dp_ids

    recursive(tree)


def create_tree_result(dir_path: str, tree: DpNode, pred_labels: np.ndarray, ground_truth: np.ndarray, col_node_id_map):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    f = open(f"{dir_path}/results.txt", "wt")
    f.write(f"NMI: {nmi(pred_labels, ground_truth, average_method='arithmetic')}\n")
    acc, acc_confusion = cluster_acc(pred_labels, ground_truth)
    f.write(f"ACC: {acc} confusion (ground truth \ prediction):\n")
    f.write(f"{acc_confusion}\n")
    f.write("\n\n")
    f.write(f"Dendrogram Purity: {dendrogram_purity(tree, ground_truth)}\n")
    lp = leaf_purity(tree, ground_truth)
    f.write(f"Leaf Purity: Avg:{lp[0]:1.3} std:{lp[1]:1.3}\n")
    col_nod_str = "\t".join([f"{x[0]}:{x[1]}" for x in sorted(list(col_node_id_map.items()), key=lambda x: x[0])])
    f.write(f"{col_nod_str}\n")
    f.write("\n\n")

    tree_str = tree2string(tree, "children", "node_id")
    f.write(f"{tree_str} \n")
    f.write("\n\n")
    f.write(_label_distribution(tree, ground_truth))
    f.close()


def create_dot_png(tree: DpNode, ground_truth: np.ndarray,dir_path: str):
    import pygraphviz as pgv
    G=pgv.AGraph()
    G.node_attr['shape'] = "box"
    def recursive(node, parent_id):
        nonlocal G
        G.add_node(node.node_id)
        n = G.get_node(node.node_id)
        if parent_id is not None:
            G.add_edge(parent_id,node.node_id)
        if node.is_leaf:
            dp_ids = list(node.dp_ids)
        else:
            dp_ids_right = recursive(node.right_child,node.node_id)
            dp_ids_left = recursive(node.left_child,node.node_id)
            dp_ids = dp_ids_right + dp_ids_left

        dp_labels = [ground_truth[dpid] for dpid in dp_ids]
        n_dps = len(dp_ids)

        counts = "".join([f"<tr><td>{i}</td><td>{count / n_dps:.3f}</td></tr>" for i, count in
                            sorted(enumerate(np.bincount(dp_labels)), key=lambda x: -1 * x[1])])
        n.attr["label"] = f'<<table><tr><td colspan="2"><font point-size="30">{node.node_id}</font></td></tr>{counts}</table>>'
        return dp_ids

    recursive(tree, None)
    G.write(f'{dir_path}/tree.dot')
    G.layout(prog='dot')
    G.draw(f'{dir_path}/tree.png', prog='dot')





def create_image_panel(image_data, rows, cols, img_size):
    # There may be fewer images in the data than rows*cols
    n_images = image_data.shape[0]

    n_max_images = rows * cols

    max_idx = np.min([n_images, n_max_images])

    panel = np.zeros((rows * img_size, cols * img_size))
    img_nr = 0
    for r in range(rows):
        for c in range(cols):
            panel[img_size * r:img_size * r + img_size, c * img_size:c * img_size + img_size] = image_data[img_nr,
                                                                                                :].reshape(
                (img_size, img_size))
            img_nr += 1
            if img_nr == max_idx:
                break
        else:  # For else is being executed if break did not occur, we continue the for loop otherwise we break it too
            continue
        break
    return panel