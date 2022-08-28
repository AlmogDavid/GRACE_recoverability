import numpy as np
import functools

import torch
import wandb
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


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


@repeat(5)
def label_classification_dgi(X_train, X_test, y_train, y_test):
    class LogReg(torch.nn.Module):
        def __init__(self, ft_in, nb_classes):
            super(LogReg, self).__init__()
            self.fc = torch.nn.Linear(ft_in, nb_classes)

            for m in self.modules():
                self.weights_init(m)

        def weights_init(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        def forward(self, seq):
            ret = self.fc(seq)
            return ret

    xent = torch.nn.CrossEntropyLoss()

    X_train = torch.from_numpy(X_train).cuda()
    X_test = torch.from_numpy(X_test).cuda()
    y_train = torch.from_numpy(y_train).cuda()
    y_test = torch.from_numpy(y_test).cuda()

    nb_classes = torch.max(torch.cat([y_train, y_test])).item() + 1
    log = LogReg(X_train.size(1), nb_classes).cuda()
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(X_train)
        loss = xent(logits, y_train)

        loss.backward()
        opt.step()

    with torch.no_grad():
        log.eval()
        logits = log(X_test)
        preds = torch.argmax(logits, dim=1)
        micro = f1_score(y_test.cpu().numpy(), preds.cpu().numpy(), average="micro")
        macro = f1_score(y_test.cpu().numpy(), preds.cpu().numpy(), average="macro")

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }


@repeat(5)
def label_classification_grace(X_train, X_test, y_train, y_test):
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
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }
