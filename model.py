from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from gaussian_kernel import GaussianKernel


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class EncoderRecoverability(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, kernel_lmbda: float, max_edges_for_r: int,
                 base_model=GCNConv, k: int = 2,):
        super(EncoderRecoverability, self).__init__()
        self.base_model = base_model
        self.max_edges_for_r = max_edges_for_r
        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation
        self.kernel = GaussianKernel(kernel_lambda=kernel_lmbda)

    def forward(self, data):
        if self.training:
            return self.loss(data, self.max_edges_for_r)
        else:
            return self._internal_forward(data)

    def _internal_forward(self, data):
        x, edge_index = data.x, data.edge_index
        dtype = x.dtype
        h = []
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index)).type(dtype)
            h.append(x)
        return h

    def loss(self, data, max_edges_for_loss: int):
        x, edge_index = data.x, data.edge_index
        h = self._internal_forward(data)
        h.insert(0, x)
        loss = 0
        lvl_loss = []
        for i in range(1, len(h)):
            relevant_edges = edge_index.T
            if relevant_edges.shape[0] > max_edges_for_loss:
                idx_to_take = torch.randperm(relevant_edges.shape[0])[:max_edges_for_loss]
                relevant_edges = relevant_edges[idx_to_take]

            neighbours_emb = h[i - 1]  # We want to be able to reproduce the neighbours from the agg nodes
            target_emb = h[i]
            source_nodes, target_nodes = torch.split(relevant_edges, 1, dim=1)
            source_nodes = source_nodes.flatten()
            target_nodes = target_nodes.flatten()

            # Need to detach the neighbours from the loss calculation
            neighbours_emb = neighbours_emb.clone()

            if neighbours_emb.requires_grad:
                neighbours_emb.register_hook(lambda grad: torch.zeros_like(grad))
            selected_neighbours = neighbours_emb[source_nodes]

            selected_targets = target_emb[target_nodes]

            curr_loss = self.kernel.compute_d(x=selected_targets, y=selected_neighbours)
            lvl_loss.append(curr_loss.item())
            loss += curr_loss
        print(lvl_loss)
        return loss


class LogReg(torch.nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc0 = torch.nn.Linear(ft_in, ft_in)
        self.fc1 = torch.nn.Linear(ft_in, nb_classes)

    def forward(self, seq):
        ret = F.leaky_relu(self.fc0(seq))
        ret = self.fc1(ret)
        return ret


class SupervisedModel(torch.nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, activation, nb_classes: int,
                 base_model=GCNConv, k: int = 2):
        super().__init__()
        self.fe = EncoderRecoverability(in_channels=in_channels,
                                        out_channels=hidden_channels,
                                        activation=activation,
                                        kernel_lmbda=0,
                                        base_model=base_model,
                                        k=k)

        self.classifier = LogReg(ft_in=hidden_channels,
                                 nb_classes=nb_classes)

    def forward(self, data):
        if self.training:
            return self.loss(data, data.train_mask)

    def _internal_forward(self, data):
        x, edge_index = data.x, data.edge_index
        embs = self.fe(x, edge_index)
        preds = self.classifier(embs[-1])
        return preds

    def loss(self, data, mask: Optional[str]):
        y = data.y
        preds = self._internal_forward(data)
        if mask is not None:
            mask = getattr(data, mask)
            preds = preds[mask]
            y = y[mask]
        loss = F.cross_entropy(input=preds, target=y)
        return loss


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, loss_type: str,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.kernel = GaussianKernel()
        self.loss_type = loss_type

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        if self.loss_type == "recoverability":
            ret = self.kernel.compute_d(x=z1, y=z2)
        elif self.loss_type == "GRACE":
            h1 = self.projection(z1)
            h2 = self.projection(z2)

            if batch_size == 0:
                l1 = self.semi_loss(h1, h2)
                l2 = self.semi_loss(h2, h1)
            else:
                l1 = self.batched_semi_loss(h1, h2, batch_size)
                l2 = self.batched_semi_loss(h2, h1, batch_size)

            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        else:
            raise RuntimeError("Invalid loss")
        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
