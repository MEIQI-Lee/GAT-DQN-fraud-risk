"""Training and inference of the DQN (RC-Explainer style) explainer.

Pipeline:
  1. freeze the trained GAT, compute full-graph embeddings & confidences;
  2. train ONE shared DQN agent over illicit nodes from the *train* period;
  3. explain illicit nodes from the *test* period with the greedy policy
     (agent never saw them during explainer training -> generalization).
"""
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from explainer.dqn import DQNAgent
from explainer.subgraph_env import SubgraphEnv, build_undirected_adjacency
from src import config
from src.visualize import plot_explanation_subgraph


class RCExplainer:
    def __init__(self, model, data, seed: int = 42):
        self.model = model
        self.model.eval()
        self.data = data
        self.rng = random.Random(seed)

        with torch.no_grad():
            logits, emb = model(data.x, data.edge_index, return_embedding=True)
        self.embeddings = emb.numpy().astype(np.float32)
        self.full_conf = torch.softmax(logits, dim=1)[:, 1].numpy()
        self.adjacency = build_undirected_adjacency(data.edge_index)
        self._env_cache = {}

    # ---------------- environment factory ----------------
    def get_env(self, target: int) -> SubgraphEnv:
        if target not in self._env_cache:
            self._env_cache[target] = SubgraphEnv(
                model=self.model,
                x=self.data.x,
                edge_index=self.data.edge_index,
                adjacency=self.adjacency,
                embeddings=self.embeddings,
                target=target,
                k_hop=config.EXPL_K_HOP,
                max_steps=config.EXPL_MAX_STEPS,
                fidelity_bonus=config.EXPL_FIDELITY_BONUS,
            )
        return self._env_cache[target]

    def eligible_nodes(self, mask: torch.Tensor, min_conf: float, require_edges: int = 2):
        """Labeled-illicit nodes the GAT already flags with enough confidence
        and that have a non-trivial ego graph to explain."""
        idx = torch.nonzero(mask & (self.data.y == 1)).flatten().tolist()
        ok = []
        for i in idx:
            if self.full_conf[i] < min_conf:
                continue
            if len(self.adjacency.get(i, set())) >= require_edges:
                env = self.get_env(i)
                if len(env.candidate_edges) >= require_edges:
                    ok.append(i)
        return ok

    def graph_lift(self, node: int):
        """How much the *graph* matters for this node:
        lift = full-graph confidence - empty-subgraph (self-loop only) confidence.
        High-lift nodes are the interesting explainability cases: their own
        features do not convict them, their transaction neighborhood does."""
        env = self.get_env(node)
        env.reset()
        return float(self.full_conf[node] - env.prev_conf), float(env.prev_conf)

    def graph_dependent_nodes(self, mask, min_conf, max_baseline, min_edges=3):
        """Illicit nodes correctly caught (full_conf >= min_conf) whose
        feature-only baseline is weak (<= max_baseline), sorted by graph lift."""
        out = []
        for i in self.eligible_nodes(mask, min_conf, require_edges=min_edges):
            lift, base = self.graph_lift(i)
            if base <= max_baseline:
                out.append((i, lift, base))
        out.sort(key=lambda t: -t[1])
        return out

    # ---------------- training ----------------
    def train(self, train_pool, episodes: int, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng.seed(seed)

        probe = self.get_env(train_pool[0])
        agent = DQNAgent(
            state_dim=probe.state_dim,
            action_dim=probe.action_dim,
            lr=config.EXPL_LR,
            gamma=config.EXPL_GAMMA,
            buffer_size=config.EXPL_BUFFER,
            batch_size=config.EXPL_BATCH,
            target_sync=config.EXPL_TARGET_SYNC,
        )

        eps = config.EXPL_EPS_START
        history = {"episode": [], "reward": [], "eps": [], "loss": []}
        for ep in range(episodes):
            target = self.rng.choice(train_pool)
            env = self.get_env(target)
            state = env.reset()
            ep_reward = 0.0
            while True:
                avail = env.available_actions()
                if not avail:
                    break
                amat = env.action_matrix(avail)
                row = agent.select(state, amat, eps)
                reward, done, _ = env.step(avail[row])
                next_state = env.state()
                nxt = env.available_actions() if not done else []
                agent.push(
                    state, amat[row], reward, next_state, done,
                    env.action_matrix(nxt) if nxt else None,
                )
                agent.update()
                ep_reward += reward
                state = next_state
                if done:
                    break
            eps = max(config.EXPL_EPS_END, eps * config.EXPL_EPS_DECAY)
            history["episode"].append(ep)
            history["reward"].append(ep_reward)
            history["eps"].append(eps)
            history["loss"].append(agent.last_loss)
            if (ep + 1) % 50 == 0:
                avg = np.mean(history["reward"][-50:])
                print(f"  episode {ep + 1}/{episodes} | avg reward(last50) "
                      f"= {avg:.4f} | eps = {eps:.3f} | loss = {agent.last_loss}")
        return agent, history

    # ---------------- explanation generation ----------------
    @torch.no_grad()
    def explain(self, target: int, agent: DQNAgent):
        """Greedy rollout (eps=0) -> ordered list of edges + confidences."""
        env = self.get_env(target)
        state = env.reset()
        baseline_conf = env.prev_conf  # do(G = empty): self-loop only
        while True:
            avail = env.available_actions()
            if not avail:
                break
            amat = env.action_matrix(avail)
            row = agent.select(state, amat, eps=0.0)
            _, done, conf = env.step(avail[row])
            state = env.state()
            if done:
                break
        edges = [env.candidate_edges[i] for i in env.selected]
        return {
            "target": target,
            "edges": edges,
            "baseline_confidence": float(baseline_conf),
            "final_confidence": float(env.prev_conf),
            "full_graph_confidence": float(self.full_conf[target]),
            "keeps_prediction": bool(env.prev_conf >= 0.5),
        }


def save_explanations(expls, data, path_prefix, agent, history):
    """Persist figures + json summary + training curve."""
    tx_ids = data.tx_ids.numpy()
    labels = {i: int(data.y[i]) for i in range(data.num_nodes)}

    summary = []
    for rank, ex in enumerate(expls, start=1):
        tx = int(tx_ids[ex["target"]])
        fig_path = config.EXPLANATIONS / f"explanation_node{rank}_tx{tx}.png"
        plot_explanation_subgraph(
            ex["edges"], labels, ex["target"], ex["final_confidence"], fig_path
        )
        summary.append({
            "rank": rank,
            "tx_id": tx,
            "node_index": ex["target"],
            "edges_global_idx": [[int(u), int(v)] for u, v in ex["edges"]],
            "edges_tx_id": [[int(tx_ids[u]), int(tx_ids[v])] for u, v in ex["edges"]],
            "n_edges": len(ex["edges"]),
            "baseline_confidence_empty_subgraph": ex["baseline_confidence"],
            "subgraph_confidence": ex["final_confidence"],
            "full_graph_confidence": ex["full_graph_confidence"],
            "keeps_illicit_prediction": ex["keeps_prediction"],
            "figure": str(fig_path.relative_to(config.PROJECT_ROOT)),
        })

    fidelity = float(np.mean([s["keeps_illicit_prediction"] for s in summary]))
    report = {
        "n_explained": len(summary),
        "fidelity_subgraph_keeps_illicit": fidelity,
        "avg_edges": float(np.mean([s["n_edges"] for s in summary])),
        "avg_subgraph_confidence": float(np.mean([s["subgraph_confidence"] for s in summary])),
        "avg_full_graph_confidence": float(np.mean([s["full_graph_confidence"] for s in summary])),
        "explanations": summary,
    }
    with open(config.EXPLANATIONS / "explanations.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # training curve (moving average of episode reward)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    r = np.asarray(history["reward"])
    ma = np.convolve(r, np.ones(25) / 25, mode="valid")
    ax.plot(history["episode"], r, alpha=0.25, color="tab:blue")
    ax.plot(history["episode"][24:], ma, color="tab:blue", lw=2,
            label="episode reward (MA-25)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("DQN explainer training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "explainer_training_curve.png", dpi=150)
    plt.close(fig)

    torch.save(agent.q_net.state_dict(), config.CHECKPOINTS / "dqn_explainer.pt")
    return report
