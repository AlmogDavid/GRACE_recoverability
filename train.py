import os
import os.path as osp
import random
import subprocess
import tempfile
from io import StringIO
import multiprocessing as mp
from time import perf_counter
from typing import Dict
import hydra
import ogb.lsc
import torch_geometric
import tqdm
import wandb
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from contextlib import nullcontext
from sklearn.model_selection import train_test_split

torch.multiprocessing.set_sharing_strategy('file_system')
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull, Reddit2, PPI, Reddit, Amazon, Coauthor, WikiCS
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv, GATConv, DataParallel

from model import Encoder, Model, drop_feature, EncoderRecoverability, SupervisedModel
from eval import label_classification_grace, label_classification_dgi, label_classification_supervised


def train_procedure(config, root_config, model, data, ans_q: mp.SimpleQueue):
    exp_type = root_config.exp_type
    learning_rate = config['learning_rate']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    if root_config.multi_gpu:
        model = DataParallel(model)

    model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    start = perf_counter()
    prev = start
    if exp_type == "random":
        print("Using random experiment, no training for model")
    else:
        data_to_feed_size = torch.cuda.device_count() if root_config.multi_gpu else 1
        if root_config.multi_gpu:
            data_for_train = []
            t = []
            for d in data:
                t.append(d)
                if len(t) == data_to_feed_size:
                    data_for_train.append(t)
                    t = []
            if len(t):
                data_for_train.append(t)
        else:
            data_for_train = data

        scaler = torch.cuda.amp.GradScaler() if config.use_half_precision else None

        for epoch in range(1, num_epochs + 1):
            loss = train(model, data_for_train, optimizer, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, scaler)

            now = perf_counter()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now

    model_state_dict = model.cpu().state_dict()
    ans_q.put(model_state_dict)
    return model_state_dict

def train(model: Model, data, optimizer, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2, scaler):
    model.train()
    total_nodes = 0
    total_loss = 0
    multi_gpu_train = True if isinstance(model, DataParallel) else False

    for curr_data in data:
        optimizer.zero_grad()
        effective_model = model.module if isinstance(model, DataParallel) else model
        with nullcontext() if scaler is None else torch.cuda.amp.autocast():
            if isinstance(effective_model, EncoderRecoverability):
                loss = model(curr_data)
            elif isinstance(effective_model, SupervisedModel):
                loss = model(curr_data)
            else:
                edge_index_1 = dropout_adj(curr_data.edge_index, p=drop_edge_rate_1)[0]
                edge_index_2 = dropout_adj(curr_data.edge_index, p=drop_edge_rate_2)[0]
                x_1 = drop_feature(curr_data.x, drop_feature_rate_1)
                x_2 = drop_feature(curr_data.x, drop_feature_rate_2)
                z1 = model(x_1, edge_index_1)
                z2 = model(x_2, edge_index_2)

                loss = model.loss(z1, z2, batch_size=0)
            loss = torch.mean(loss)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            loss = scaler.scale(loss)
            loss.backward()
            scaler.step(optimizer)
            scaler.update()

        if multi_gpu_train:
            loss_avg = loss.item()
        else:
            total_nodes += curr_data.x.size(0)
            total_loss += loss.item() * curr_data.x.size(0)
            loss_avg = total_loss / total_nodes

    return loss_avg


def save_test_emb(model: Model, data: torch_geometric.data.Data, train_data_dir: str, running_dtype):
    with torch.no_grad():
        model.eval()

        # y_train_agg = []
        # y_test_agg = []
        # z_train_agg = []
        # z_test_agg = []
        nb_classes = 0


        for i, curr_data in tqdm.tqdm(enumerate(data), "Generating embeddings for testing"):
            curr_data.x = curr_data.x.type(running_dtype)
            curr_data = curr_data.cuda()
            x, edge_index, y, test_mask = curr_data.x, curr_data.edge_index, curr_data.y, curr_data.test_mask
            z = model(curr_data)
            if isinstance(model, EncoderRecoverability):
                z = z[-1]

            has_test = torch.any(test_mask)
            has_train = not torch.all(test_mask)
            nb_classes = max(nb_classes, torch.max(y).item())

            if has_train:
                train_emb_np = z[~test_mask].cpu().numpy()
                file_for_train_emb = os.path.join(train_data_dir, f"train_emb_{i}_{len(train_emb_np)}.npy")
                np.save(file_for_train_emb, train_emb_np)

                file_for_train_label = os.path.join(train_data_dir, f"train_lbl_{i}_{len(train_emb_np)}.npy")
                np.save(file_for_train_label, y[~test_mask].cpu().numpy())

            if has_test:
                test_emb_np = z[test_mask].cpu().numpy()
                file_for_test_emb = os.path.join(train_data_dir, f"test_emb_{i}_{len(test_emb_np)}.npy")
                np.save(file_for_test_emb, test_emb_np)

                file_for_test_label = os.path.join(train_data_dir, f"test_lbl_{i}_{len(test_emb_np)}.npy")
                np.save(file_for_test_label, y[test_mask].cpu().numpy())


            #z_train_agg.append(z[~test_mask].cpu())
            #z_test_agg.append(z[test_mask].cpu())
            # y_train_agg.append(y[~test_mask].cpu())
            # y_test_agg.append(y[test_mask].cpu())
            # curr_data.cpu()

        # z_train = torch.cat(z_train_agg).numpy()
        # z_test = torch.cat(z_test_agg).numpy()
        # y_train = torch.cat(y_train_agg).numpy()
        # y_test = torch.cat(y_test_agg).numpy()

    nb_classes += 1 # It holds the highest index

    return train_data_dir, nb_classes


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


def cluster_data(data: torch_geometric.data.Data, num_clusters: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    cluster_data = ClusterData(data, num_parts=num_clusters, recursive=False,
                               save_dir=save_dir)
    loader = ClusterLoader(cluster_data,
                           batch_size=1,
                           shuffle=True,
                           num_workers=8)
    data = [d for d in loader]
    return data


@hydra.main(config_path="configs", config_name="default", version_base=None)  # Config name will be given via command line
def main(root_config: DictConfig):
    dataset_name = root_config.dataset
    config = root_config[dataset_name] # Load the relevant part
    assert config["eval_method"] in ("GRACE", "DGI")

    method = root_config.method
    exp_type = root_config.exp_type
    if not root_config.multi_gpu:
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
        elif name == "Reddit":
            data = Reddit(root=path)[0]  # Remove the first 2 features because there are in different scale
            data.x = data.x[:, 1:]
        elif name in ("ogbn_arxiv", "ogbn_products"):
            dataset = PygNodePropPredDataset(name=name.replace("_", "-"),
                                             root=path.replace("_","-"),
                                             transform=T.NormalizeFeatures())
            data = dataset[0]
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
            dataset = ogb.lsc.MAG240MDataset(root=path)
            x = torch.arange(dataset.num_papers) # We will load the actual data after the clustering is done
            y = torch.from_numpy(dataset.all_paper_label)
            edge_index = torch.from_numpy(dataset.edge_index('paper', 'paper'))
            train_mask = torch.zeros((x.size(0),), dtype=torch.bool)
            train_mask[dataset.get_idx_split("train")] = True
            val_mask = torch.zeros((x.size(0),), dtype=torch.bool)
            val_mask[dataset.get_idx_split("valid")] = True
            test_mask = torch.zeros((x.size(0),), dtype=torch.bool)
            test_mask[dataset.get_idx_split("test-dev")] = True # TODO: do we need test-dev or test-challenge? https://ogb.stanford.edu/docs/lsc/mag240m/
            # Remove all nodes without labels from the train/val/test splits
            nodes_with_labels = torch.logical_not(torch.logical_or(torch.isnan(y), y < 0))
            train_mask = torch.logical_and(train_mask, nodes_with_labels)
            val_mask = torch.logical_and(val_mask, nodes_with_labels)
            test_mask = torch.logical_and(test_mask, nodes_with_labels)

            data = torch_geometric.data.Data(x=x,
                                             edge_index=edge_index,
                                             y=y,
                                             train_mask=train_mask,
                                             val_mask=val_mask,
                                             test_mask=test_mask)
            data.edge_index = torch_geometric.utils.to_undirected(data.edge_index, None, num_nodes=data.x.size(0))
        else:
            raise ValueError(f"Invalid DS: {dataset_name}")

        if config.num_data_splits > 1:
            data = cluster_data(data, config.num_data_splits, path)

        if name == "mag_240m":
            print("Loading data to clusters for mag_240m")
            orig_x = torch.from_numpy(dataset.all_paper_feat)
            for d in tqdm.tqdm(data):
                d.x = orig_x[d.x]

        elif isinstance(data, torch_geometric.data.Data):
            data = [data]

        return data

    path = osp.join(osp.expanduser('~'), 'datasets', dataset_name)
    data = get_dataset(path, dataset_name)

    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv,
                   'GATConv': GATConv})[config['base_model']]
    num_layers = config['num_layers']
    tau = config['tau']

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
            model = EncoderRecoverability(data[0].x.size(-1), num_hidden, activation, base_model=base_model, k=num_layers, kernel_lmbda=float(config["kernel_lambda"]), max_edges_for_r=config["max_edges_for_r"])
        else:
            encoder = Encoder(data[0].x.size(-1), num_hidden, activation,
                              base_model=base_model, k=num_layers).cuda()
            model = Model(encoder, num_hidden, num_proj_hidden, method, tau)

    print("Starting training process")
    ans_q = mp.SimpleQueue()
    train_p = mp.Process(target=train_procedure, args=(config, root_config, model, data, ans_q), )
    train_p.start()
    print("Waiting for training to end")
    train_p.join()
    model_state_dict = ans_q.get()

    #model_state_dict = train_procedure(config=config, root_config=root_config, model=model, data=data)
    print("Loading state dict for evaluation")
    if root_config.multi_gpu:
        model_state_dict = {k.split("module.")[1]: v for k,v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)

    print("=== Final ===")

    eval_method = config['eval_method']
    with tempfile.TemporaryDirectory() as tmp_dir:
        running_dtype = torch.half if config.use_half_precision else torch.float32
        if config.use_half_precision:
            model = model.half()
        with torch.cuda.amp.autocast() if config.use_half_precision is None else nullcontext():
            train_data_dir, nb_classes = save_test_emb(model.cuda(), data, tmp_dir, running_dtype)

        del data # Release memory

        if exp_type == "supervised":
            print("Start testing using SUPERVISED method")
            label_classification_supervised(train_data_dir)
        else:
            if eval_method == "DGI":
                print("Start testing using DGI method")
                label_classification_dgi(train_data_dir, nb_classes)
            elif eval_method == "GRACE":
                print("Start testing using GRACE method")
                label_classification_grace(train_data_dir)
            else:
                raise RuntimeError("Invalid classification method")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
