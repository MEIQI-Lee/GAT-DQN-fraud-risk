"""Risk-control evaluation metrics.

All metrics treat class 1 (illicit / fraud) as the positive class.
Terminology follows the risk-control (风控) convention:
  - Recall (召回率) = TPR: share of fraud caught by the model
  - FPR   (误报率) = FP / (FP + TN): share of good transactions wrongly flagged
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y_true, y_score, max_fpr: float):
    """Best achievable recall subject to FPR <= max_fpr.

    Returns (recall, threshold, achieved_fpr). Thresholds come from the ROC
    curve; the operating point is the highest-recall threshold whose FPR does
    not exceed the constraint.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = fpr <= max_fpr
    if not ok.any():
        return 0.0, float("inf"), 0.0
    idx = int(np.argmax(tpr[ok]))
    return float(tpr[ok][idx]), float(thr[ok][idx]), float(fpr[ok][idx])


def ks_statistic(y_true, y_score):
    """KS = max(TPR - FPR); also returns the threshold achieving it."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    diff = tpr - fpr
    idx = int(np.argmax(diff))
    return float(diff[idx]), float(thr[idx]), float(tpr[idx]), float(fpr[idx])


def best_f1_threshold(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    idx = int(np.argmax(f1[:-1]))  # last point of PR curve has no threshold
    return float(thr[idx]), float(f1[idx]), float(prec[idx]), float(rec[idx])


def compute_risk_metrics(y_true, y_score) -> dict:
    """Full risk-control metric suite."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    ks, ks_thr, ks_tpr, ks_fpr = ks_statistic(y_true, y_score)
    r1, thr1, fpr1 = recall_at_fpr(y_true, y_score, 0.01)
    r5, thr5, fpr5 = recall_at_fpr(y_true, y_score, 0.05)
    f1_thr, f1, f1_prec, f1_rec = best_f1_threshold(y_true, y_score)

    y_pred = (y_score >= f1_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    return {
        "n_samples": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ks": ks,
        "ks_threshold": ks_thr,
        "ks_tpr": ks_tpr,
        "ks_fpr": ks_fpr,
        "recall_at_fpr_1pct": r1,
        "threshold_at_fpr_1pct": thr1,
        "achieved_fpr_1pct": fpr1,
        "recall_at_fpr_5pct": r5,
        "threshold_at_fpr_5pct": thr5,
        "achieved_fpr_5pct": fpr5,
        "f1_optimal_threshold": f1_thr,
        "f1": f1,
        "precision_at_f1": f1_prec,
        "recall_at_f1": f1_rec,
        "confusion_matrix_at_f1": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def format_report(m: dict, title: str = "GAT 欺诈交易识别 · 测试集评估报告") -> str:
    """Render metrics as a Chinese risk-control report (markdown)."""
    cm = m["confusion_matrix_at_f1"]
    lines = [
        f"# {title}",
        "",
        f"- 样本量：{m['n_samples']:,}（欺诈 {m['n_positive']:,}，占比 "
        f"{m['n_positive'] / m['n_samples']:.2%}）",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| ROC-AUC | {m['roc_auc']:.4f} |",
        f"| PR-AUC (Average Precision) | {m['pr_auc']:.4f} |",
        f"| KS（阈值 {m['ks_threshold']:.4f}） | {m['ks']:.4f} |",
        f"| Recall @ FPR≤1% | {m['recall_at_fpr_1pct']:.4f} "
        f"（阈值 {m['threshold_at_fpr_1pct']:.4f}，实际 FPR {m['achieved_fpr_1pct']:.4f}） |",
        f"| Recall @ FPR≤5% | {m['recall_at_fpr_5pct']:.4f} "
        f"（阈值 {m['threshold_at_fpr_5pct']:.4f}，实际 FPR {m['achieved_fpr_5pct']:.4f}） |",
        f"| F1（最优阈值 {m['f1_optimal_threshold']:.4f}） | {m['f1']:.4f} |",
        f"| Precision @ F1 阈值 | {m['precision_at_f1']:.4f} |",
        f"| Recall @ F1 阈值 | {m['recall_at_f1']:.4f} |",
        "",
        "## 混淆矩阵（F1 最优阈值）",
        "",
        "| | 预测正常 | 预测欺诈 |",
        "| --- | --- | --- |",
        f"| **实际正常** | TN={cm['tn']:,} | FP={cm['fp']:,} |",
        f"| **实际欺诈** | FN={cm['fn']:,} | TP={cm['tp']:,} |",
        "",
        "## 风控结论",
        "",
        f"- 模型整体区分度 AUC = {m['roc_auc']:.4f}，KS = {m['ks']:.4f}；",
        f"- 在**误报率不超过 1%** 的严格约束下，欺诈交易召回率达到 "
        f"**{m['recall_at_fpr_1pct']:.2%}**（阈值 {m['threshold_at_fpr_1pct']:.4f}）；",
        f"- 在**误报率不超过 5%** 的约束下，欺诈交易召回率达到 "
        f"**{m['recall_at_fpr_5pct']:.2%}**（阈值 {m['threshold_at_fpr_5pct']:.4f}）；",
        f"- 以 F1 最优阈值 {m['f1_optimal_threshold']:.4f} 部署时，精确率 "
        f"{m['precision_at_f1']:.2%}、召回率 {m['recall_at_f1']:.2%}。",
    ]
    return "\n".join(lines)
