import os.path as osp
import random
import subprocess
from io import StringIO
from time import perf_counter as t
from typing import Dict
import hydra
import ogb.lsc
import torch_geometric
import wandb
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

torch.multiprocessing.set_sharing_strategy('file_system')
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull, Reddit2, PPI, Reddit, Amazon, Coauthor, WikiCS
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv, GATConv

from model import Encoder, Model, drop_feature, EncoderRecoverability, SupervisedModel
from eval import label_classification_grace, label_classification_dgi, label_classification_supervised


def train(model: Model, data, max_edges_for_r, optimizer, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2):
    model.train()
    total_nodes = 0
    total_loss = 0
    for curr_data in data:
        x, edge_index = curr_data.x.cuda(), curr_data.edge_index.cuda()
        optimizer.zero_grad()
        if isinstance(model, EncoderRecoverability):
            h = model(x, edge_index)
            loss = model.loss(x, h, edge_index, max_edges_for_r)
        elif isinstance(model, SupervisedModel):
            loss = model.loss(x=x, edge_index=edge_index, y=curr_data.y.cuda(), mask=curr_data.train_mask.cuda())
        else:
            edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
            x_1 = drop_feature(x, drop_feature_rate_1)
            x_2 = drop_feature(x, drop_feature_rate_2)
            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)

            loss = model.loss(z1, z2, batch_size=0)
        loss.backward()
        optimizer.step()
        total_nodes += x.size(0)
        total_loss += loss.item() * x.size(0)

    loss_avg = total_loss / total_nodes

    return loss_avg


def test(model: Model, data: torch_geometric.data.Data, eval_method: str, exp_type: str):
    with torch.no_grad():
        model.eval()
        y_train_agg = []
        y_test_agg = []
        z_train_agg = []
        z_test_agg = []

        for curr_data in data:
            x, edge_index, y, test_mask = curr_data.x.cuda(), curr_data.edge_index.cuda(), curr_data.y.cuda(), curr_data.test_mask.cuda()
            z = model(x, edge_index)
            if isinstance(model, EncoderRecoverability):
                z = z[-1]
            z_train_agg.append(z[~test_mask].cpu())
            z_test_agg.append(z[test_mask].cpu())
            y_train_agg.append(y[~test_mask].cpu())
            y_test_agg.append(y[test_mask].cpu())

        z_train = torch.cat(z_train_agg).numpy()
        z_test = torch.cat(z_test_agg).numpy()
        y_train = torch.cat(y_train_agg).numpy()
        y_test = torch.cat(y_test_agg).numpy()

    if exp_type == "supervised":
        label_classification_supervised(z_test, y_test)
    else:
        if eval_method == "DGI":
            label_classification_dgi(z_train, z_test, y_train, y_test)
        elif eval_method == "GRACE":
            label_classification_grace(z_train, z_test, y_train, y_test)
        else:
            raise RuntimeError("Invalid classification method")


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode()),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = gpu_df['memory.free'].idxmax()
    print('The most free is GPU={} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx

def cfg2dict(cfg: DictConfig) -> Dict:
    """
    Recursively convert OmegaConf to vanilla dict
    """
    cfg_dict = {}
    for k, v in cfg.items():
        if type(v) == DictConfig:
            cfg_dict[k] = cfg2dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict


def cluster_data(data: torch_geometric.data.Data, num_clusters: int):
    cluster_data = ClusterData(data, num_parts=num_clusters, recursive=False,
                               save_dir=None)
    loader = ClusterLoader(cluster_data,
                           batch_size=1,
                           shuffle=True,
                           num_workers=4)
    data = [d for d in loader]
    return data


@hydra.main(config_path="configs", config_name="default")  # Config name will be given via command line
def main(root_config: DictConfig):
    dataset_name = root_config.dataset
    config = root_config[dataset_name] # Load the relevant part
    assert config["eval_method"] in ("GRACE", "DGI")

    method = root_config.method
    exp_type = root_config.exp_type
    torch.cuda.set_device(get_free_gpu())

    if root_config.use_wandb:
        wandb.init(project=root_config.wandb_project)
        config_to_log = cfg2dict(config)
        config_to_log["dataset"] = dataset_name
        config_to_log["method"] = root_config.method
        config_to_log["exp_type"] = root_config.exp_type
        wandb.config.update(config_to_log)

    print(OmegaConf.to_yaml(config))

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv,
                   'GATConv': GATConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    def get_dataset(path, name):
        name = 'dblp' if name == 'DBLP' else name

        if name == "dblp":
            data = CitationFull(root=path, name=name, transform=T.NormalizeFeatures())[0]
            # There is no split, so we perform random split
            node_idx = np.arange(data.x.size(0))
            labels = data.y
            idx_train, idx_test, _, _ = train_test_split(node_idx, labels,
                                                                test_size=0.2)
            train_mask = torch.zeros_like(data.y, dtype=torch.bool)
            train_mask[idx_train] = True
            test_mask = torch.zeros_like(train_mask)
            test_mask[idx_test] = True
            data.train_mask = train_mask
            data.test_mask = test_mask
        elif name in ("Cora", "CiteSeer", "PubMed"):
            data = Planetoid(root=path, name=name, transform=T.NormalizeFeatures())[0]
        elif name == "Reddit2":
            data = Reddit2(root=path)[0]  # Remove the first 2 features because there are in different scale
            data.x = data.x[:, 2:]
            data = T.NormalizeFeatures()(data)
        elif name == "Reddit":
            data = Reddit(root=path)[0]  # Remove the first 2 features because there are in different scale
            data.x = data.x[:, 2:]
            data = T.NormalizeFeatures()(data)
        elif name in ("ogbn_arxiv", "ogbn_products"):
            dataset = PygNodePropPredDataset(name=name.replace("_", "-"),
                                             root=path.replace("_","-"),
                                             transform=T.NormalizeFeatures()) # TODO: run without undirected as well
            data = dataset[0] # TODO: write the dropout idea
            split_idx = dataset.get_idx_split()
            data.train_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
            data.train_mask[split_idx["train"]] = True
            data.val_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
            data.val_mask[split_idx["valid"]] = True
            data.test_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
            data.test_mask[split_idx["test"]] = True
            data.y = data.y.flatten()
            data.edge_index = torch_geometric.utils.to_undirected(data.edge_index, None, num_nodes=data.x.size(0))
        elif name == "PPI":
            train_ds = PPI(root=path, split="train")
            val_ds = PPI(root=path, split="val")
            test_ds = PPI(root=path, split="test")

            # Build masks
            data_map = {"train_mask": [],
                        "val_mask": [],
                        "test_mask": []}
            for curr_ds, relevant_mask in ((train_ds, "train_mask"), (val_ds, "val_mask"), (test_ds, "test_mask")):
                for data in curr_ds:
                    data.val_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
                    data.train_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
                    data.test_mask = torch.zeros((data.x.size(0),), dtype=torch.bool)
                    setattr(data, relevant_mask, torch.ones((data.x.size(0),), dtype=torch.bool))
                    data_map[relevant_mask].append(data)

            # Merge graphs
            data = data_map["train_mask"] + data_map["val_mask"] + data_map["test_mask"]
        elif name in ("amazon_photos", "amazon_computers"):
            ds_sub_name = {"amazon_photos": "photo",
                           "amazon_computers": "computers"}[name]
            data = Amazon(root=path, name=ds_sub_name, transform=T.NormalizeFeatures())[0]
            node_idx = np.arange(data.x.size(0))
            labels = data.y
            idx_train, idx_test, _, _ = train_test_split(node_idx, labels,
                                                         test_size=0.2)
            train_mask = torch.zeros_like(data.y, dtype=torch.bool)
            train_mask[idx_train] = True
            test_mask = torch.zeros_like(train_mask)
            test_mask[idx_test] = True
            data.train_mask = train_mask
            data.test_mask = test_mask
        elif name in ("coauthor_physics", "coauthor_cs"):
            ds_sub_name = {"coauthor_physics": "physics",
                           "coauthor_cs": "CS"}[name]
            data = Coauthor(root=path, name=ds_sub_name)[0]
            node_idx = np.arange(data.x.size(0))
            labels = data.y
            idx_train, idx_test, _, _ = train_test_split(node_idx, labels,
                                                         test_size=0.2)
            train_mask = torch.zeros_like(data.y, dtype=torch.bool)
            train_mask[idx_train] = True
            test_mask = torch.zeros_like(train_mask)
            test_mask[idx_test] = True
            data.train_mask = train_mask
            data.test_mask = test_mask
        elif name == "wiki_cs":
            data = WikiCS(root=path)[0]
        elif name == "mag_240m":
            data = ogb.lsc.MAG240MDataset(root=path)
            loli = 3
        else:
            raise ValueError(f"Invalid DS: {dataset_name}")

        if config.num_data_splits > 1:
            data = cluster_data(data, config.num_data_splits)
        elif isinstance(data, torch_geometric.data.Data):
            data = [data]

        return data

    path = osp.join(osp.expanduser('~'), 'datasets', dataset_name)
    data = get_dataset(path, dataset_name)

    if exp_type == "supervised":
        num_classes = torch.max(torch.stack([torch.max(d.y) for d in data])).item() + 1
        model = SupervisedModel(in_channels=data[0].x.size(-1),
                                hidden_channels=num_hidden,
                                activation=activation,
                                nb_classes=num_classes,
                                base_model=base_model,
                                k=num_layers).cuda()
    else:
        if method == "recoverability":
            model = EncoderRecoverability(data[0].x.size(-1), num_hidden, activation, base_model=base_model, k=num_layers, kernel_lmbda=float(config["kernel_lambda"])).cuda()
        else:
            encoder = Encoder(data[0].x.size(-1), num_hidden, activation,
                              base_model=base_model, k=num_layers).cuda()
            model = Model(encoder, num_hidden, num_proj_hidden, method, tau).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    if exp_type == "random":
        print("Using random experiment, no training for model")
    else:
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data, config["max_edges_for_r"], optimizer, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2)

            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now

        print("=== Final ===")
    test(model, data, config['eval_method'], exp_type)


if __name__ == '__main__':
    main()
