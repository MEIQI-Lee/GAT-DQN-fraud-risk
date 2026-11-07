"""Central configuration: paths, data split, model & explainer hyperparameters."""
from pathlib import Path

# ---------------- paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
REPORTS = RESULTS / "reports"
CHECKPOINTS = RESULTS / "checkpoints"
EXPLANATIONS = RESULTS / "explanations"

for _d in (DATA_RAW, DATA_PROCESSED, FIGURES, REPORTS, CHECKPOINTS, EXPLANATIONS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- reproducibility ----------------
SEED = 42

# ---------------- temporal split (Elliptic, 49 timesteps) ----------------
# Reference protocol (Weber et al. 2019; clemwo's Kaggle notebook):
# train pool = timesteps 1..34, test = 35..49 (strictly out-of-time).
# The validation set used for early stopping is a stratified random 15%
# holdout from the train pool. (Note: a *temporal* val window 31..34 was
# tried first and led to a large train-period/test-period generalization
# gap under concept drift; the random holdout is the standard protocol.)
TRAIN_END_TS = 34       # timesteps 1..34 -> train pool
VAL_FRACTION = 0.15     # stratified random holdout within the train pool
                        # timesteps 35..49 -> test (out-of-time)

# ---------------- GAT ----------------
GAT_IN_DIM = 165        # Elliptic node features (93 local + 72 aggregated)
GAT_HIDDEN = 64
GAT_HEADS = 8
GAT_OUT_HIDDEN = 100    # penultimate embedding size (also used by explainer)
GAT_DROPOUT = 0.3
GAT_USE_SKIP = False    # feature-skip connection into the classifier head
GAT_LR = 1e-3
GAT_WEIGHT_DECAY = 5e-4
GAT_EPOCHS = 300
GAT_PATIENCE = 40       # early stopping on validation AUC

# ---------------- DQN explainer (RC-Explainer style) ----------------
EXPL_K_HOP = 2              # candidate pool: edges of the 2-hop ego graph
EXPL_MAX_STEPS = 6          # explanation budget: at most 6 edges
EXPL_FIDELITY_BONUS = 0.5   # lambda: reward term for keeping the prediction
EXPL_EPISODES = 600
EXPL_BUFFER = 5000
EXPL_BATCH = 64
EXPL_GAMMA = 0.95
EXPL_LR = 1e-3
EXPL_EPS_START = 1.0
EXPL_EPS_END = 0.05
EXPL_EPS_DECAY = 0.99       # multiplicative, per episode
EXPL_TARGET_SYNC = 100      # gradient steps between target-network syncs
EXPL_TRAIN_NODES = 40       # illicit train-period nodes used to train the agent
EXPL_MIN_CONF = 0.6         # min model confidence for a node to be explained
EXPL_TOPK_EXPLAIN = 5       # number of test-period illicit nodes explained
