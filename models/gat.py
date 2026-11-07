"""Graph Attention Network for transaction node classification (licit / illicit)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Two-layer multi-head GAT.

    conv1: in_dim -> hidden, `heads` attention heads (concatenated)
    conv2: hidden*heads -> out_hidden, single head
    lin:   out_hidden -> num_classes

    `forward(..., return_embedding=True)` additionally returns the penultimate
    node embeddings, which the DQN explainer uses as its state/action features.
    """

    def __init__(
        self,
        in_dim: int = 165,
        hidden: int = 64,
        heads: int = 8,
        out_hidden: int = 100,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_skip: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_skip = use_skip
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_hidden, heads=1, concat=False, dropout=dropout)
        # feature-skip: classifier sees [GAT embedding | raw features] so that the
        # decision can lean on node features when graph neighborhoods drift
        self.lin = nn.Linear(out_hidden + (in_dim if use_skip else 0), num_classes)

    def forward(self, x, edge_index, return_embedding: bool = False):
        x_in = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.conv2(x, edge_index))
        out = self.lin(torch.cat([h, x_in], dim=1) if self.use_skip else h)
        if return_embedding:
            return out, h
        return out
