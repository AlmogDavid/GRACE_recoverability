import glob
import os
from typing import List

import numpy as np
import functools

import torch
import tqdm
import wandb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from torch.utils.data.dataset import Dataset
from torch_geometric.data import DataLoader

from model import LogReg

class EvalDataset(Dataset):

    def __init__(self, data_dir: str, prefix: str, transform = None):
        all_emb = glob.glob(os.path.join(data_dir, f"{prefix}_emb*"))
        all_labels = glob.glob(os.path.join(data_dir, f"{prefix}_lbl*"))

        sort_func_file_map = lambda x: int(x.split(".npy")[0].split("_")[-2])
        self.all_emb = sorted(all_emb, key=sort_func_file_map)
        self.all_lbl = sorted(all_labels, key=sort_func_file_map)

        self.all_emb_arr = [np.load(f) for f in tqdm.tqdm(self.all_emb, f"Loading precomputed embeddings for prefix: {prefix}")]
        self.all_emb_arr = torch.from_numpy(np.concatenate(self.all_emb_arr))

        self.all_lbl_arr = [np.load(f) for f in tqdm.tqdm(self.all_lbl, f"Loading precomputed labels for prefix: {prefix}")]
        self.all_lbl_arr = torch.from_numpy(np.concatenate(self.all_lbl_arr))

        self.transform = transform

    def __len__(self):
        return len(self.all_lbl_arr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.all_emb_arr[idx], self.all_lbl_arr[idx])
        return sample


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        if wandb.run is not None:
            wandb.log({f"{key}_mean": mean,
                       f"{key}_std": std})
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(10)
def label_classification_supervised(data_dir):
    test_ds = EvalDataset(data_dir, "test")
    preds_test = []
    y_test = []

    for pred, gt in test_ds:
        preds_test.append(pred)
        y_test.append(gt)

    preds_test = np.concatenate(preds_test)
    y_test = np.concatenate(y_test)

    preds_test = np.argmax(preds_test, axis=1)
    micro = f1_score(y_test, preds_test, average="micro")
    macro = f1_score(y_test, preds_test, average="macro")
    accuracy = accuracy_score(y_test, preds_test)

    return {
        'F1Mi': micro,
        'F1Ma': macro,
        "Accuracy": accuracy,
    }

@repeat(10)
def label_classification_dgi(data_dir: str, nb_classes: int):
    train_ds = EvalDataset(data_dir, "train")
    test_ds = EvalDataset(data_dir, "train")
    dummy_s = train_ds[0]
    sample_emb, sample_gt = dummy_s[0].numpy(), dummy_s[1].numpy()
    if sample_emb.ndim > 1 and sample_gt.shape[1] > 1: # Multi label DS
        criterion = torch.nn.BCEWithLogitsLoss()
        nb_classes = sample_emb.shape[1]
        multi_label = True
    else:
        criterion = torch.nn.CrossEntropyLoss()
        multi_label = False

    log = LogReg(len(sample_emb), nb_classes).cuda()
    opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.0)

    train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=True)

    for _ in tqdm.tqdm(range(100), "Training classifier"):
        log.train()
        #perm = torch.randperm(y_train.size(0)).cuda().type(torch.long)
        #X_train = X_train[perm]
        #y_train = y_train[perm]
        for curr_x, curr_y in train_dl:
            opt.zero_grad()
            logits = log(curr_x.type(torch.float32).cuda())
            loss = criterion(logits, curr_y.cuda())

            loss.backward()
            opt.step()

    with torch.no_grad():
        log.eval()
        preds = []
        y_test = []
        for curr_x, curr_y in test_dl:
            logits = log(curr_x.type(torch.float32).cuda()).cpu()
            if multi_label:
                preds.append((logits > 0))
            else:
                preds.append(torch.argmax(logits, dim=1))
            y_test.append(curr_y)

        y_test = torch.cat(y_test)
        preds = torch.cat(preds)

        micro = f1_score(y_test.cpu().numpy(), preds.cpu().numpy(), average="micro")
        macro = f1_score(y_test.cpu().numpy(), preds.cpu().numpy(), average="macro")
        accuracy = accuracy_score(y_test.cpu().numpy(), preds.cpu().numpy())

    res_dict = {
        'F1Mi': micro,
        'F1Ma': macro,
        "Accuracy": accuracy,
    }
    print(res_dict)

    return res_dict


@repeat(10)
def label_classification_grace(X_train, X_test, y_train, y_test):
    if y_train.ndim > 1 and y_train.shape[1] > 1: # Multi label DS
        y_train = y_train.astype(np.bool)
        y_test = y_test.astype(np.bool)
    else:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        onehot_encoder = OneHotEncoder(categories='auto').fit(np.concatenate([y_train, y_test]))
        y_train = onehot_encoder.transform(y_train).toarray().astype(np.bool)
        y_test = onehot_encoder.transform(y_test).toarray().astype(np.bool)

    X_train = normalize(X_train, norm='l2')
    X_test = normalize(X_test, norm='l2')

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'F1Mi': micro,
        'F1Ma': macro,
        'Accuracy': accuracy
    }
