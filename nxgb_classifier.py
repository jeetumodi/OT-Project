"""
Neural-XGBoost (N-XGB) Hybrid Disaster Prediction -- Full Model Comparison
===========================================================================
Based on: "Neural-XGBoost: A Hybrid Approach for Disaster Prediction
           and Management Using Machine Learning" (IEEE Access, 2025)

Pipeline:
  1. Data Preprocessing  (filter, impute, encode, scale, SMOTE)
  2. Feature Extraction  (PyTorch NN -- once per experiment)
  3. GridSearchCV        (LR, SVM, RF, XGBoost, N-XGB on extracted features)
  4. Evaluation          (5-digit metrics, combined ROC, confusion matrices)

Two experiments:
  A. Balanced   (SMOTE on training data)
  B. Imbalanced (raw training data, no SMOTE)

Anti-leakage safeguards:
  - StandardScaler fit ONLY on train split, transform test
  - SMOTE ONLY on train split
  - GridSearchCV with StratifiedKFold(5) -- test set never seen during tuning
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ## Configuration ########################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "public_emdat.csv")
TARGET_COL = "Disaster Type"
CLASSES    = ["Wildfire", "Flood", "Earthquake"]
N_CLASSES  = len(CLASSES)
PRECISION  = 5  # decimal places for metrics

DROP_COLS = [
    # --- Identifiers / metadata ---
    "DisNo.", "External IDs", "Event Name",
    "Entry Date", "Last Update",
    "Admin Units", "GADM Admin Units",
    "Classification Key", "Historic",
    # --- Taxonomy columns that trivially encode Disaster Type ---
    # Disaster Subgroup: Geophysical=Earthquake, Hydrological=Flood,
    #                    Climatological=Wildfire  (100% correlation)
    "Disaster Group", "Disaster Subgroup", "Disaster Subtype",
    # Magnitude Scale: Moment Magnitude=Earthquake, Km2=Flood/Wildfire
    "Magnitude Scale",
    # Origin: cause descriptions specific to each disaster type
    "Origin",
    # Associated Types: related disaster types (leaks target)
    "Associated Types",
    # River Basin: only populated for floods (acts as a flood flag)
    "River Basin",
]

# NN hyperparameters
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NN_LR      = 0.001
EPOCHS     = 50
HIDDEN_DIM = 32

# GridSearchCV settings
CV_FOLDS   = 5
CV_SCORING = "f1_weighted"


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def print_header(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def print_class_dist(y, target_le, label=""):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Class distribution {label}:")
    for idx, cnt in zip(unique, counts):
        name = target_le.inverse_transform([idx])[0]
        print(f"    {name:>15s} : {cnt}")


class FeatureExtractorNN(nn.Module):
    """Input(N) --> 128 --> 64 --> 32 (H) --> 3"""

    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.3),
        )
        self.hidden = nn.Sequential(
            nn.Linear(64, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(),
        )
        self.classifier = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        h = self.hidden(x)
        out = self.classifier(h)
        return out, h


def train_nn_and_extract(X_tr, y_tr, X_te, label=""):
    """Train PyTorch NN, return (train_features_32d, test_features_32d)."""
    print(f"\n  [NN] Training feature extractor ({label}) ...")
    print(f"  [NN] Device: {DEVICE} | Epochs: {EPOCHS} | LR: {NN_LR}")

    tr_x = torch.tensor(X_tr, dtype=torch.float32)
    tr_y = torch.tensor(y_tr, dtype=torch.long)
    te_x = torch.tensor(X_te, dtype=torch.float32)

    dataset = TensorDataset(tr_x, tr_y)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = FeatureExtractorNN(X_tr.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_LR)

    for epoch in tqdm(range(1, EPOCHS + 1), desc=f"  NN ({label})", unit="ep"):
        model.train()
        ep_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * bx.size(0)
        ep_loss /= len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(f"    Epoch {epoch:>3d}/{EPOCHS}  --  Loss: {ep_loss:.4f}")

    model.eval()
    with torch.no_grad():
        _, feat_tr = model(tr_x.to(DEVICE))
        _, feat_te = model(te_x.to(DEVICE))
    feat_tr = feat_tr.cpu().numpy()
    feat_te = feat_te.cpu().numpy()

    assert feat_tr.shape == (X_tr.shape[0], HIDDEN_DIM)
    assert feat_te.shape == (X_te.shape[0], HIDDEN_DIM)
    print(f"  [NN] Extracted: train {feat_tr.shape}, test {feat_te.shape}  [OK]")
    return feat_tr, feat_te


def get_models_and_grids():
    """Return dict of {name: (estimator, param_grid)}."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=500, random_state=42),
            {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"]},
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200, 300], "max_depth": [5, 7, 10],
             "min_samples_split": [2, 5]},
        ),
        "XGBoost": (
            XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                          random_state=42, verbosity=0),
            {"learning_rate": [0.01, 0.1, 0.2], "max_depth": [5, 7, 9],
             "n_estimators": [100, 200, 300]},
        ),
        "N-XGB (Hybrid)": (
            XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                          random_state=42, verbosity=0),
            {"learning_rate": [0.01, 0.1, 0.2], "max_depth": [5, 7, 9],
             "n_estimators": [100, 200, 300]},
        ),
    }


def run_gridsearch(name, estimator, param_grid, X_tr, y_tr):
    """Run GridSearchCV, return best estimator."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator, param_grid, cv=cv, scoring=CV_SCORING,
        n_jobs=-1, verbose=0, refit=True,
    )
    gs.fit(X_tr, y_tr)
    print(f"    Best params : {gs.best_params_}")
    print(f"    Best CV F1  : {gs.best_score_:.{PRECISION}f}")
    return gs.best_estimator_


def evaluate_model(model, X_te, y_te):
    """Return (y_pred, y_prob, metrics_dict)."""
    y_pred = model.predict(X_te)
    # Get probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)
    else:
        y_prob = None

    metrics = {
        "Accuracy":  accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y_te, y_pred, average="weighted", zero_division=0),
        "F1-Score":  f1_score(y_te, y_pred, average="weighted", zero_division=0),
    }

    # Train accuracy for overfitting check is done outside
    return y_pred, y_prob, metrics


def plot_confusion_matrices(results, experiment_tag, y_te):
    """Plot all confusion matrices in one figure."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, info) in zip(axes, results.items()):
        cm = confusion_matrix(y_te, info["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    linewidths=0.5, linecolor="gray", ax=ax)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(f"Confusion Matrices -- {experiment_tag}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, f"cm_{experiment_tag.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


def plot_combined_roc(results, experiment_tag, y_te):
    """Plot all models' ROC curves on one figure."""
    y_te_bin = label_binarize(y_te, classes=list(range(N_CLASSES)))
    model_colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, (name, info) in enumerate(results.items()):
        y_prob = info["y_prob"]
        if y_prob is None:
            continue
        # Compute macro-average ROC
        fpr_all, tpr_all = [], []
        auc_scores = []
        for c in range(N_CLASSES):
            fpr_c, tpr_c, _ = roc_curve(y_te_bin[:, c], y_prob[:, c])
            auc_scores.append(auc(fpr_c, tpr_c))
            fpr_all.append(fpr_c)
            tpr_all.append(tpr_c)

        # Macro-average by interpolation
        all_fpr = np.unique(np.concatenate(fpr_all))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(N_CLASSES):
            mean_tpr += np.interp(all_fpr, fpr_all[c], tpr_all[c])
        mean_tpr /= N_CLASSES
        macro_auc = np.mean(auc_scores)

        ax.plot(all_fpr, mean_tpr, color=model_colors[idx % len(model_colors)],
                lw=2, label=f"{name} (AUC = {macro_auc:.{PRECISION}f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.50000)")
    ax.set_title(f"Multiclass ROC-AUC -- {experiment_tag}", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, f"roc_{experiment_tag.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


def print_results_table(results, experiment_tag):
    """Print a formatted comparison table with 5-digit precision."""
    print(f"\n  {'Model':<25s} {'Accuracy':>12s} {'Precision':>12s} {'Recall':>12s} {'F1-Score':>12s}")
    print("  " + "-" * 73)
    for name, info in results.items():
        m = info["metrics"]
        print(f"  {name:<25s} "
              f"{m['Accuracy']:>12.{PRECISION}f} "
              f"{m['Precision']:>12.{PRECISION}f} "
              f"{m['Recall']:>12.{PRECISION}f} "
              f"{m['F1-Score']:>12.{PRECISION}f}")


def print_overfit_check(results, experiment_tag):
    """Print train vs test accuracy to flag potential overfitting."""
    print(f"\n  Overfitting Check ({experiment_tag}):")
    print(f"  {'Model':<25s} {'Train Acc':>12s} {'Test Acc':>12s} {'Gap':>10s}  {'Status'}")
    print("  " + "-" * 73)
    for name, info in results.items():
        tr = info.get("train_acc", 0.0)
        te = info["metrics"]["Accuracy"]
        gap = tr - te
        status = "OK" if gap < 0.05 else "WARNING: possible overfit"
        print(f"  {name:<25s} {tr:>12.{PRECISION}f} {te:>12.{PRECISION}f} {gap:>+10.{PRECISION}f}  {status}")


def run_experiment(experiment_tag, X_tr_raw, y_tr, X_te_raw, y_te, X_tr_nn, X_te_nn):
    """
    Run GridSearchCV + evaluation for all 5 models.
    - LR, SVM, RF, XGBoost: trained on raw (scaled) features
    - N-XGB: trained on NN-extracted features
    """
    print_header(f"EXPERIMENT: {experiment_tag}")
    models_grids = get_models_and_grids()
    results = {}

    for name, (estimator, param_grid) in models_grids.items():
        print(f"\n  --- {name} ---")

        # N-XGB uses NN-extracted features; others use raw features
        if name == "N-XGB (Hybrid)":
            X_tr_use, X_te_use = X_tr_nn, X_te_nn
        else:
            X_tr_use, X_te_use = X_tr_raw, X_te_raw

        t0 = time.time()
        best_model = run_gridsearch(name, estimator, param_grid, X_tr_use, y_tr)
        elapsed = time.time() - t0
        print(f"    GridSearch time: {elapsed:.1f}s")

        y_pred, y_prob, metrics = evaluate_model(best_model, X_te_use, y_te)

        # Train accuracy for overfitting check
        train_acc = accuracy_score(y_tr, best_model.predict(X_tr_use))

        results[name] = {
            "y_pred": y_pred, "y_prob": y_prob,
            "metrics": metrics, "train_acc": train_acc,
            "best_model": best_model,
        }
        print(f"    Test Accuracy: {metrics['Accuracy']:.{PRECISION}f}")

    return results


# =========================================================================
# MAIN PIPELINE
# =========================================================================

if __name__ == "__main__":
    total_start = time.time()

    # =====================================================================
    # STAGE 1: DATA PREPROCESSING
    # =====================================================================
    print_header("STAGE 1: Data Preprocessing")

    # 1a. Load
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # 1b. Filter
    df = df[df[TARGET_COL].isin(CLASSES)].reset_index(drop=True)
    print(f"  Filtered to {CLASSES}: {df.shape[0]} rows")

    # 1c. Drop non-predictive columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 1d. Separate target
    y_series = df.pop(TARGET_COL)

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"  Numerical: {len(num_cols)} | Categorical: {len(cat_cols)}")

    # 1e. Impute
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # 1f. Encode categorical
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    target_le.fit(CLASSES)
    y = target_le.transform(y_series)
    print(f"  Target encoding: {dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}")

    X = df.values.astype(np.float32)
    print(f"  Feature matrix: {X.shape}")

    # 1g. Train/Test split FIRST (70:30, stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )
    print(f"  Split: Train={X_train_raw.shape[0]} | Test={X_test_raw.shape[0]}")

    # 1h. Scale -- FIT ON TRAIN ONLY (prevents data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)
    print("  StandardScaler: fit on train, transform on test  [OK - no leakage]")

    # =====================================================================
    # STAGE 2: PREPARE BALANCED + IMBALANCED DATASETS
    # =====================================================================
    print_header("STAGE 2: Prepare Balanced & Imbalanced Datasets")

    # --- Imbalanced (raw) ---
    X_train_imb = X_train_scaled.copy()
    y_train_imb = y_train.copy()
    print_class_dist(y_train_imb, target_le, "(Imbalanced Train)")

    # --- Balanced (SMOTE on train only) ---
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print_class_dist(y_train_bal, target_le, "(Balanced Train - SMOTE)")

    print(f"\n  Imbalanced train: {X_train_imb.shape[0]} samples")
    print(f"  Balanced train:   {X_train_bal.shape[0]} samples")
    print(f"  Test set:         {X_test_scaled.shape[0]} samples (untouched)")

    # =====================================================================
    # STAGE 3: NEURAL NETWORK FEATURE EXTRACTION
    # =====================================================================
    print_header("STAGE 3: PyTorch NN Feature Extraction")

    # Extract features for BALANCED experiment
    nn_train_bal, nn_test_bal = train_nn_and_extract(
        X_train_bal, y_train_bal, X_test_scaled, label="Balanced"
    )

    # Extract features for IMBALANCED experiment
    nn_train_imb, nn_test_imb = train_nn_and_extract(
        X_train_imb, y_train_imb, X_test_scaled, label="Imbalanced"
    )

    # =====================================================================
    # STAGE 4: GRIDSEARCHCV -- ALL MODELS
    # =====================================================================
    print_header("STAGE 4: GridSearchCV -- Model Training & Tuning")

    # --- Experiment A: Balanced ---
    results_bal = run_experiment(
        "Balanced (SMOTE)",
        X_train_bal, y_train_bal, X_test_scaled, y_test,
        nn_train_bal, nn_test_bal,
    )

    # --- Experiment B: Imbalanced ---
    results_imb = run_experiment(
        "Imbalanced (Raw)",
        X_train_imb, y_train_imb, X_test_scaled, y_test,
        nn_train_imb, nn_test_imb,
    )

    # =====================================================================
    # STAGE 5: EVALUATION & VISUALIZATION
    # =====================================================================
    print_header("STAGE 5: Evaluation & Visualization")

    # --- Metrics tables ---
    print("\n  *** BALANCED (SMOTE) RESULTS ***")
    print_results_table(results_bal, "Balanced")
    print_overfit_check(results_bal, "Balanced")

    print("\n  *** IMBALANCED (RAW) RESULTS ***")
    print_results_table(results_imb, "Imbalanced")
    print_overfit_check(results_imb, "Imbalanced")

    # --- Per-model classification reports ---
    for tag, results in [("Balanced", results_bal), ("Imbalanced", results_imb)]:
        print(f"\n  --- Classification Reports ({tag}) ---")
        for name, info in results.items():
            print(f"\n  [{name}]")
            print(classification_report(
                y_test, info["y_pred"], target_names=CLASSES, digits=PRECISION
            ))

    # --- Confusion matrices (all models in one figure per experiment) ---
    print("\n  Generating confusion matrices ...")
    plot_confusion_matrices(results_bal, "Balanced", y_test)
    plot_confusion_matrices(results_imb, "Imbalanced", y_test)

    # --- Combined ROC curves (one figure per experiment) ---
    print("\n  Generating combined ROC-AUC curves ...")
    plot_combined_roc(results_bal, "Balanced", y_test)
    plot_combined_roc(results_imb, "Imbalanced", y_test)

    # --- Summary ---
    total_elapsed = time.time() - total_start
    print_header("PIPELINE COMPLETE")
    print(f"  Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Models evaluated: {len(results_bal)} x 2 experiments")
    print(f"  Test samples: {X_test_scaled.shape[0]} (never used in training/tuning)")
    print()
