"""Step 4: train the DQN explainer and explain test-period fraud nodes.

Usage:  python scripts/explain.py [--episodes N] [--topk K]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explainer.rc_explainer import RCExplainer, save_explanations
from src import config
from src.data_utils import build_dataset
from src.utils import set_seed
from scripts.evaluate import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=config.EXPL_EPISODES)
    parser.add_argument("--topk", type=int, default=config.EXPL_TOPK_EXPLAIN)
    args = parser.parse_args()

    set_seed(config.SEED)
    data = build_dataset()
    model, _ = load_model()

    # clear stale explanation artifacts from previous runs
    for old in config.EXPLANATIONS.glob("explanation_node*.png"):
        old.unlink()

    explainer = RCExplainer(model, data, seed=config.SEED)

    # -- agent is trained on TRAIN-period fraud nodes only --
    # prefer "graph-dependent" nodes (feature-only baseline weak, graph lifts
    # them over the decision boundary): edges actually matter there
    gd_train = explainer.graph_dependent_nodes(
        data.train_mask, min_conf=config.EXPL_MIN_CONF, max_baseline=0.5)
    train_pool = [i for i, _, _ in gd_train[: config.EXPL_TRAIN_NODES]]
    if len(train_pool) < 8:  # fallback: any confidently-caught illicit nodes
        train_pool = explainer.eligible_nodes(data.train_mask, config.EXPL_MIN_CONF)
        train_pool = train_pool[: config.EXPL_TRAIN_NODES]
    print(f"explainer train pool: {len(train_pool)} illicit train-period nodes "
          f"(graph-dependent: {len(gd_train)})")

    t0 = time.time()
    agent, history = explainer.train(train_pool, episodes=args.episodes,
                                     seed=config.SEED)
    print(f"[explain] agent trained in {time.time() - t0:.0f}s")

    # -- explain TEST-period fraud nodes the agent never saw --
    gd_test = explainer.graph_dependent_nodes(
        data.test_mask, min_conf=config.EXPL_MIN_CONF, max_baseline=0.5)
    targets = [i for i, _, _ in gd_test[: args.topk]]
    if len(targets) < args.topk:  # top up with the most confident ones
        rest = explainer.eligible_nodes(data.test_mask, config.EXPL_MIN_CONF)
        rest = [i for i in rest if i not in targets]
        rest.sort(key=lambda i: -explainer.full_conf[i])
        targets += rest[: args.topk - len(targets)]
    print(f"explaining {len(targets)} illicit test nodes "
          f"(graph-dependent pool: {len(gd_test)})")

    expls = [explainer.explain(t, agent) for t in targets]
    report = save_explanations(expls, data, "explanation", agent, history)

    print(json.dumps({k: v for k, v in report.items() if k != "explanations"},
                     indent=2))
    for s in report["explanations"]:
        print(f"  node tx={s['tx_id']} | edges={s['n_edges']} | "
              f"conf: empty={s['baseline_confidence_empty_subgraph']:.3f} -> "
              f"subgraph={s['subgraph_confidence']:.3f} "
              f"(full-graph {s['full_graph_confidence']:.3f}) | "
              f"keeps illicit: {s['keeps_illicit_prediction']}")
    print(f"explanations -> {config.EXPLANATIONS}")


if __name__ == "__main__":
    main()
