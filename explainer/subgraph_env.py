"""Subgraph selection environment for the DQN explainer.

Follows RC-Explainer (Wang et al., TPAMI 2022, arXiv:2204.11028):
the explanation task is framed as a sequential decision process in which the
agent successively adds one edge from the frontier of the current subgraph.

  state  : frozen GAT embeddings of the target node + readout of the current
           subgraph nodes (+ normalized step count)
  action : pick one candidate edge (from the k-hop ego graph) to add
  reward : causal effect term + fidelity term
             r_t = [ P(y=illicit | do(G = S_t)) - P(y=illicit | do(G = S_{t-1})) ]
                   + lambda * 1{ argmax P(. | do(G = S_t)) == illicit }

NOTE (approximation disclosure): P(y | do(G = S)) is the GAT's own prediction
on node v when *all* message passing is restricted to the selected subgraph S
(every other edge is removed). This interventional estimate treats the trained
GNN as the causal mechanism and measures each edge's counterfactual
contribution relative to the previously selected edges. It is NOT a formally
identified statistical causal effect; it is the standard approximation used by
RC-Explainer and is documented as such in the README.
"""
from collections import deque

import numpy as np
import torch


def build_undirected_adjacency(edge_index: torch.Tensor):
    """dict[int, set[int]] adjacency treating money-flow edges as undirected
    (used only to delimit the candidate pool; edges themselves stay directed)."""
    adj = {}
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


class SubgraphEnv:
    def __init__(
        self,
        model,
        x: torch.Tensor,                 # full feature matrix (CPU)
        edge_index: torch.Tensor,        # full directed edge list (2, E)
        adjacency: dict,                 # undirected adjacency for ego BFS
        embeddings: np.ndarray,          # frozen GAT embeddings (N, H)
        target: int,                     # global node index to explain
        k_hop: int = 2,
        max_steps: int = 6,
        fidelity_bonus: float = 0.5,
    ):
        self.model = model
        self.target = int(target)
        self.max_steps = max_steps
        self.fidelity_bonus = fidelity_bonus
        self.emb_dim = embeddings.shape[1]

        # ---- k-hop ego graph (candidate pool) ----
        ego = {self.target}
        frontier = {self.target}
        for _ in range(k_hop):
            nxt = set()
            for u in frontier:
                nxt |= adjacency.get(u, set())
            nxt -= ego
            ego |= nxt
            frontier = nxt
        self.ego_nodes = sorted(ego)
        self.g2l = {g: i for i, g in enumerate(self.ego_nodes)}
        self.target_local = self.g2l[self.target]

        # candidate directed edges with both endpoints inside the ego graph
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        in_ego = np.isin(src, self.ego_nodes) & np.isin(dst, self.ego_nodes)
        self.candidate_edges = [
            (int(u), int(v)) for u, v in zip(src[in_ego], dst[in_ego])
        ]

        self.x_ego = x[self.ego_nodes]
        self.emb_ego = embeddings[self.ego_nodes]           # (n_ego, H)
        self.emb_target = embeddings[self.target]           # (H,)
        self._edge_emb = np.concatenate(
            [embeddings[[u for u, _ in self.candidate_edges]],
             embeddings[[v for _, v in self.candidate_edges]]], axis=1
        ) if self.candidate_edges else np.zeros((0, 2 * self.emb_dim), dtype=np.float32)

        # episode state
        self.selected = []          # list of candidate-edge indices
        self.selected_nodes = set() # global ids covered by the subgraph
        self.prev_conf = 0.0

    # ---------------- RL interface ----------------
    @property
    def state_dim(self) -> int:
        return 2 * self.emb_dim + 1

    @property
    def action_dim(self) -> int:
        return 2 * self.emb_dim

    def reset(self):
        self.selected = []
        self.selected_nodes = {self.target}
        self.prev_conf = self._confidence()
        return self.state()

    def state(self) -> np.ndarray:
        if self.selected_nodes:
            idx = [self.g2l[g] for g in self.selected_nodes]
            readout = self.emb_ego[idx].mean(axis=0)
        else:
            readout = np.zeros(self.emb_dim, dtype=np.float32)
        return np.concatenate(
            [self.emb_target, readout, [len(self.selected) / self.max_steps]]
        ).astype(np.float32)

    def available_actions(self):
        """Frontier edges: not yet selected, touching the current subgraph."""
        return [
            i for i, (u, v) in enumerate(self.candidate_edges)
            if i not in self.selected and (u in self.selected_nodes or v in self.selected_nodes)
        ]

    def action_matrix(self, action_ids) -> np.ndarray:
        return self._edge_emb[action_ids].astype(np.float32)

    def step(self, cand_idx: int):
        u, v = self.candidate_edges[cand_idx]
        self.selected.append(cand_idx)
        self.selected_nodes.update((u, v))

        conf = self._confidence()
        causal_effect = conf - self.prev_conf
        keeps_prediction = conf >= 0.5
        reward = causal_effect + self.fidelity_bonus * float(keeps_prediction)
        self.prev_conf = conf

        done = len(self.selected) >= self.max_steps or not self.available_actions()
        return float(reward), bool(done), float(conf)

    # ---------------- internals ----------------
    def _confidence(self) -> float:
        """P(illicit) for the target when message passing uses ONLY the
        currently selected edges (the do(G = S) intervention)."""
        if self.selected:
            ei = torch.tensor(
                [[self.g2l[self.candidate_edges[i][0]] for i in self.selected],
                 [self.g2l[self.candidate_edges[i][1]] for i in self.selected]],
                dtype=torch.long,
            )
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
        with torch.no_grad():
            out = self.model(self.x_ego, ei)
            prob = torch.softmax(out[self.target_local], dim=-1)[1]
        return float(prob)
