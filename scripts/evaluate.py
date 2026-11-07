"""Step 3: risk-control evaluation on the out-of-time test period.

Outputs (results/):
  reports/metrics.json            - machine-readable metrics
  reports/evaluation_report.md    - Chinese risk-control report
  figures/{roc,pr,ks,confusion_matrix,score_distribution}.png

Usage:  python scripts/evaluate.py [--protocol temporal|random]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.gat import GAT
from src import config
from src.data_utils import build_dataset, describe
from src.metrics import compute_risk_metrics, format_report
from src.utils import set_seed
from src.visualize import (
    plot_confusion_matrix,
    plot_ks,
    plot_pr,
    plot_roc,
    plot_score_distribution,
)


def load_model(protocol: str = "temporal"):
    suffix = "" if protocol == "temporal" else "_random"
    ckpt = torch.load(config.CHECKPOINTS / f"gat_best{suffix}.pt", weights_only=True)
    hp = ckpt["hyperparams"]
    model = GAT(**hp)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["temporal", "random"],
                        default="temporal")
    args = parser.parse_args()
    suffix = "" if args.protocol == "temporal" else "_random"

    set_seed(config.SEED)
    data = build_dataset(protocol=args.protocol)
    print(describe(data))
    model, ckpt = load_model(args.protocol)
    print(f"loaded checkpoint (best val AUC {ckpt['best_val_auc']:.4f} "
          f"@ epoch {ckpt['best_epoch']})")

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        prob = torch.softmax(out, dim=1)[:, 1]

    y_test = data.y[data.test_mask].numpy()
    s_test = prob[data.test_mask].numpy()

    m = compute_risk_metrics(y_test, s_test)
    m["protocol"] = args.protocol
    with open(config.REPORTS / f"metrics{suffix}.json", "w") as f:
        json.dump(m, f, indent=2)

    title = "GAT 欺诈交易识别 · 测试集评估报告"
    if args.protocol == "random":
        title += "（随机划分参考协议）"
    report = format_report(m, title=title)
    (config.REPORTS / f"evaluation_report{suffix}.md").write_text(report, encoding="utf-8")
    print(report)

    plot_roc(y_test, s_test, config.FIGURES / f"roc_curve{suffix}.png")
    plot_pr(y_test, s_test, config.FIGURES / f"pr_curve{suffix}.png")
    plot_ks(y_test, s_test, config.FIGURES / f"ks_curve{suffix}.png")
    plot_confusion_matrix(m["confusion_matrix_at_f1"],
                          config.FIGURES / f"confusion_matrix{suffix}.png")
    plot_score_distribution(y_test, s_test, config.FIGURES / f"score_distribution{suffix}.png")
    print(f"figures -> {config.FIGURES}")


if __name__ == "__main__":
    main()
