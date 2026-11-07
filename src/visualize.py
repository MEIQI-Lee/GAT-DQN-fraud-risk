"""Plotting utilities: metric curves and explanation subgraph figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"GAT (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test, out-of-time)")
    ax.legend()
    _save(fig, path)


def plot_pr(y_true, y_score, path):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = auc(rec, prec)
    base = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"GAT (PR-AUC = {ap:.4f})")
    ax.axhline(base, ls="--", color="gray", label=f"Base rate = {base:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (test)")
    ax.legend()
    _save(fig, path)


def plot_ks(y_true, y_score, path):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    diff = tpr - fpr
    idx = int(np.argmax(diff))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thr, tpr, lw=2, label="TPR (recall)")
    ax.plot(thr, fpr, lw=2, label="FPR")
    ax.plot(thr, diff, lw=2, ls="--", label="KS = TPR - FPR")
    ax.axvline(thr[idx], color="red", ls=":", label=f"KS={diff[idx]:.4f} @ thr={thr[idx]:.4f}")
    ax.set_xlabel("Score threshold (P(illicit))")
    ax.set_ylabel("Rate")
    ax.set_title("KS Curve (test)")
    ax.legend()
    _save(fig, path)


def plot_confusion_matrix(cm: dict, path):
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                color="white" if v > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred licit", "pred illicit"])
    ax.set_yticks([0, 1], ["true licit", "true illicit"])
    ax.set_title("Confusion Matrix (F1-optimal threshold)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_score_distribution(y_true, y_score, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.7, label="licit", density=True)
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.7, label="illicit", density=True)
    ax.set_yscale("log")
    ax.set_xlabel("Model score P(illicit)")
    ax.set_ylabel("Density (log)")
    ax.set_title("Score Distribution by Class (test)")
    ax.legend()
    _save(fig, path)


def plot_training_curve(history: dict, path):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(history["epoch"], history["train_loss"], color="tab:blue", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_auc"], color="tab:red", label="val AUC")
    ax2.set_ylabel("Validation AUC", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("GAT Training Curve")
    _save(fig, path)


def plot_explanation_subgraph(
    edges,                # list[(u_global, v_global)] in selection order
    node_labels: dict,    # global node id -> -1/0/1
    target: int,
    final_conf: float,
    path,
):
    """Visualize an explanation subgraph.

    Node color: red=illicit, green=licit, gray=unknown, target=yellow star.
    Edge labels show the order in which the DQN agent selected each edge.
    """
    g = nx.DiGraph()
    g.add_edges_from(edges)
    g.add_node(target)

    def color(n):
        if n == target:
            return "#FFD700"  # gold for the target
        lab = node_labels.get(n, -1)
        return {1: "#d62728", 0: "#2ca02c"}.get(lab, "#9e9e9e")

    node_colors = [color(n) for n in g.nodes]
    node_sizes = [900 if n == target else 380 for n in g.nodes]

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(g, seed=42, k=1.6)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors="black", linewidths=0.8, ax=ax)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=16,
                           edge_color="#444444", width=1.6, ax=ax,
                           connectionstyle="arc3,rad=0.08")
    edge_labels = {(u, v): f"#{i + 1}" for i, (u, v) in enumerate(edges)}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=9,
                                 font_color="#8B0000", ax=ax)
    short = {n: str(n)[:8] for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=short, font_size=7, ax=ax)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFD700",
               markeredgecolor="k", markersize=12, label="target (explained node)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markeredgecolor="k", markersize=10, label="illicit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="k", markersize=10, label="licit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9e9e9e",
               markeredgecolor="k", markersize=10, label="unknown"),
    ]
    ax.legend(handles=legend, loc="best", fontsize=8)
    ax.set_title(f"DQN explanation subgraph for node {target}\n"
                 f"subgraph-only P(illicit) = {final_conf:.4f}, "
                 f"#edges = selection order")
    ax.axis("off")
    _save(fig, path)
