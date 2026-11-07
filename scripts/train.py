"""Step 2: train the GAT fraud classifier.

Full-graph transductive training (standard Elliptic protocol):
message passing uses the whole graph; the loss is computed on labeled
train-period nodes only. Early stopping on validation-period AUC.

Usage:  python scripts/train.py [--protocol temporal|random]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from models.gat import GAT
from src import config
from src.data_utils import build_dataset, describe
from src.utils import set_seed
from src.visualize import plot_training_curve


@torch.no_grad()
def eval_auc(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    prob = torch.softmax(out, dim=1)[:, 1]
    return roc_auc_score(data.y[mask].numpy(), prob[mask].numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["temporal", "random"],
                        default="temporal")
    args = parser.parse_args()
    suffix = "" if args.protocol == "temporal" else "_random"

    set_seed(config.SEED)
    data = build_dataset(protocol=args.protocol)
    print(describe(data))

    model = GAT(
        in_dim=config.GAT_IN_DIM,
        hidden=config.GAT_HIDDEN,
        heads=config.GAT_HEADS,
        out_hidden=config.GAT_OUT_HIDDEN,
        dropout=config.GAT_DROPOUT,
        use_skip=config.GAT_USE_SKIP,
    )
    opt = torch.optim.Adam(model.parameters(), lr=config.GAT_LR,
                           weight_decay=config.GAT_WEIGHT_DECAY)

    y_train = data.y[data.train_mask].numpy()
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = torch.tensor(w, dtype=torch.float32)
    print(f"class weights (licit, illicit) = {w.round(3).tolist()}")

    history = {"epoch": [], "train_loss": [], "val_auc": []}
    best_auc, best_state, best_epoch, bad = 0.0, None, -1, 0

    t0 = time.time()
    for epoch in range(1, config.GAT_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],
                               weight=class_weight)
        loss.backward()
        opt.step()

        val_auc = eval_auc(model, data, data.val_mask)
        history["epoch"].append(epoch)
        history["train_loss"].append(loss.item())
        history["val_auc"].append(float(val_auc))

        if val_auc > best_auc:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= config.GAT_PATIENCE:
                print(f"early stop at epoch {epoch}")
                break
        if epoch % 10 == 0:
            print(f"epoch {epoch:>3} | loss {loss.item():.4f} | val AUC {val_auc:.4f}"
                  f" | best {best_auc:.4f} @ {best_epoch} | {time.time() - t0:.0f}s")

    model.load_state_dict(best_state)
    test_auc = eval_auc(model, data, data.test_mask)
    print(f"[train] best val AUC {best_auc:.4f} (epoch {best_epoch}) | "
          f"test AUC {test_auc:.4f} | total {time.time() - t0:.0f}s")

    ckpt = {
        "state_dict": model.state_dict(),
        "hyperparams": {
            "in_dim": config.GAT_IN_DIM, "hidden": config.GAT_HIDDEN,
            "heads": config.GAT_HEADS, "out_hidden": config.GAT_OUT_HIDDEN,
            "dropout": config.GAT_DROPOUT, "use_skip": config.GAT_USE_SKIP,
        },
        "protocol": args.protocol,
        "best_val_auc": best_auc, "best_epoch": best_epoch,
        "test_auc_at_best": float(test_auc), "seed": config.SEED,
    }
    torch.save(ckpt, config.CHECKPOINTS / f"gat_best{suffix}.pt")
    plot_training_curve(history, config.FIGURES / f"training_curve{suffix}.png")
    with open(config.REPORTS / f"training_log{suffix}.json", "w") as f:
        json.dump({k: v for k, v in ckpt.items() if k != "state_dict"}, f, indent=2)
    print(f"checkpoint -> {config.CHECKPOINTS / f'gat_best{suffix}.pt'}")


if __name__ == "__main__":
    main()
