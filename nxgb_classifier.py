"""
Neural-XGBoost (N-XGB) Hybrid Disaster Prediction -- Full Model Comparison
===========================================================================
Based on: "Neural-XGBoost: A Hybrid Approach for Disaster Prediction
           and Management Using Machine Learning" (IEEE Access, 2025)

Pipeline according to Paper (Algorithm 1 & Figure 1):
  1. Data Preprocessing:
     - Filter disaster types: Wildfire, Flood, Earthquake (3 classes)
     - Impute missing values: Median for numerical, Mode for categorical
     - Encode categorical variables: LabelEncoder
     - Normalize numerical features: StandardScaler (fit on train split)
     - Split: 70% Train, 30% Test (Stratified Holdout Validation)
  2. Feature Extraction (PyTorch Neural Network):
     - Architecture: Fully connected layers (Input -> 128 -> 64 -> 32 -> 3)
     - Activation: ReLU, Regularization: Dropout (0.3) + BatchNorm
     - Training: 50 epochs, batch size 64, Adam optimizer (lr=0.001)
     - Extracted Embeddings H: 32-dimensional latent feature vector (P < N)
  3. Class Balancing (SMOTE):
     - Applied after feature extraction to balance training embeddings H_train (Eq. 6-7)
     - Applied to raw training features X_train for baseline models
     - Test data is left untouched (no leakage)
  4. Model Training & Tuning (GridSearchCV with 5-fold Stratified CV):
     - Baseline Models: Logistic Regression, SVM, Random Forest, k-Nearest Neighbors (k-NN)
     - Proposed Model: N-XGB (XGBoost classifier trained on NN-extracted features H)
     - Hyperparameters tuned per Table 2:
       * Learning rate: [0.01, 0.05, 0.1]
       * Max tree depth: [5, 7, 9]
       * Number of estimators: [100, 200, 300]
  5. Comprehensive Metric Evaluation & Comparison:
     - Accuracy & Balanced Accuracy
     - Precision (Macro, Weighted, Per-Class)
     - Recall / Sensitivity (Macro, Weighted, Per-Class)
     - Specificity / True Negative Rate (Macro, Weighted, Per-Class)
     - F1-Score (Macro, Weighted, Per-Class)
     - ROC-AUC (Macro, Weighted, Per-Class)
     - Matthews Correlation Coefficient (MCC) & Cohen's Kappa
     - Multi-model Confusion Matrices & ROC Visualizations
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
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, log_loss, matthews_corrcoef, cohen_kappa_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

# NN hyperparameters (Paper Section IV.A / System Specifications)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NN_LR      = 0.001
EPOCHS     = 50
HIDDEN_DIM = 32  # P < N compressed representation

# GridSearchCV settings
CV_FOLDS   = 5
CV_SCORING = "f1_weighted"


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def print_header(title):
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


def print_class_dist(y, target_le, label=""):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Class distribution {label}:")
    for idx, cnt in zip(unique, counts):
        name = target_le.inverse_transform([idx])[0]
        print(f"    {name:>15s} : {cnt}")


def compute_multiclass_specificity(y_true, y_pred, n_classes=3):
    """Compute per-class Specificity (TNR = TN / (TN + FP))."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    specificities = []
    total = np.sum(cm)
    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        tn = total - tp - fp - fn
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    return np.array(specificities)


class FeatureExtractorNN(nn.Module):
    """
    Neural Network Feature Extractor (Section III.D.1 & IV.A):
    Input(N) --> Linear(128) + BN + ReLU + Dropout(0.3)
             --> Linear(64)  + BN + ReLU + Dropout(0.3)
             --> Linear(32)  + BN + ReLU [Extracted Latent Embeddings H]
             --> Linear(num_classes)
    """

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
    """
    Train PyTorch NN feature extractor on training data,
    then extract 32-dimensional latent feature representations H_train and H_test.
    """
    print(f"\n  [NN] Training feature extractor ({label}) ...")
    print(f"  [NN] Device: {DEVICE} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {NN_LR}")

    tr_x = torch.tensor(X_tr, dtype=torch.float32)
    tr_y = torch.tensor(y_tr, dtype=torch.long)
    te_x = torch.tensor(X_te, dtype=torch.float32)

    dataset = TensorDataset(tr_x, tr_y)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = FeatureExtractorNN(X_tr.shape[1], num_classes=N_CLASSES).to(DEVICE)
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
    print(f"  [NN] Extracted embeddings H: train {feat_tr.shape}, test {feat_te.shape}  [OK]")
    return feat_tr, feat_te


def get_models_and_grids():
    """
    Return dictionary of {name: (estimator, param_grid)} matching models from the paper:
    1. Logistic Regression
    2. Support Vector Machine (SVM)
    3. Random Forest
    4. k-Nearest Neighbors (k-NN)
    5. Proposed Model: Neural-XGBoost (N-XGB Hybrid)
    """
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
        "k-NN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]},
        ),
        "N-XGB (Hybrid)": (
            XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                          random_state=42, verbosity=0),
            {
                # Hyperparameters tested exactly per Table 2 in paper:
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [5, 7, 9],
                "n_estimators": [100, 200, 300],
            },
        ),
    }


def run_gridsearch(name, estimator, param_grid, X_tr, y_tr):
    """Run GridSearchCV with 5-fold Stratified CV, return best estimator."""
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
    """
    Comprehensive evaluation returning:
    - Overall metrics: Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, MCC, Kappa, Log Loss
    - Per-class metrics: Precision, Recall, Specificity, F1, ROC-AUC for each disaster class
    """
    y_pred = model.predict(X_te)
    y_te_bin = label_binarize(y_te, classes=list(range(N_CLASSES)))

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)
    else:
        y_prob = None

    # Per-class metrics
    p_per_class = precision_score(y_te, y_pred, average=None, zero_division=0)
    r_per_class = recall_score(y_te, y_pred, average=None, zero_division=0)
    f_per_class = f1_score(y_te, y_pred, average=None, zero_division=0)
    spec_per_class = compute_multiclass_specificity(y_te, y_pred, N_CLASSES)

    # Class support counts
    unique_classes, counts = np.unique(y_te, return_counts=True)
    support_dict = dict(zip(unique_classes, counts))
    weights = np.array([support_dict.get(c, 0) for c in range(N_CLASSES)], dtype=float)
    weights /= weights.sum()

    # Per-class ROC-AUC & Macro/Weighted ROC-AUC
    auc_per_class = []
    if y_prob is not None:
        for c in range(N_CLASSES):
            try:
                auc_c = roc_auc_score(y_te_bin[:, c], y_prob[:, c])
            except Exception:
                auc_c = np.nan
            auc_per_class.append(auc_c)
        try:
            macro_auc = roc_auc_score(y_te_bin, y_prob, average="macro", multi_class="ovr")
            weighted_auc = roc_auc_score(y_te_bin, y_prob, average="weighted", multi_class="ovr")
        except Exception:
            macro_auc = np.nanmean(auc_per_class)
            weighted_auc = np.nansum(np.array(auc_per_class) * weights)
        try:
            loss_val = log_loss(y_te, y_prob)
        except Exception:
            loss_val = np.nan
    else:
        auc_per_class = [np.nan] * N_CLASSES
        macro_auc = np.nan
        weighted_auc = np.nan
        loss_val = np.nan

    metrics = {
        # Standard Overall Metrics
        "Accuracy":          accuracy_score(y_te, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_te, y_pred),
        "MCC":               matthews_corrcoef(y_te, y_pred),
        "Kappa":             cohen_kappa_score(y_te, y_pred),
        "Log_Loss":          loss_val,

        # Macro Averages
        "Precision_Macro":   precision_score(y_te, y_pred, average="macro", zero_division=0),
        "Recall_Macro":      recall_score(y_te, y_pred, average="macro", zero_division=0),
        "Specificity_Macro": float(np.mean(spec_per_class)),
        "F1_Macro":          f1_score(y_te, y_pred, average="macro", zero_division=0),
        "ROC_AUC_Macro":     macro_auc,

        # Weighted Averages
        "Precision_Weighted":   precision_score(y_te, y_pred, average="weighted", zero_division=0),
        "Recall_Weighted":      recall_score(y_te, y_pred, average="weighted", zero_division=0),
        "Specificity_Weighted": float(np.sum(spec_per_class * weights)),
        "F1_Weighted":          f1_score(y_te, y_pred, average="weighted", zero_division=0),
        "ROC_AUC_Weighted":     weighted_auc,

        # Per-class breakdowns (list corresponding to CLASSES)
        "Per_Class_Precision":   p_per_class,
        "Per_Class_Recall":      r_per_class,
        "Per_Class_Specificity": spec_per_class,
        "Per_Class_F1":          f_per_class,
        "Per_Class_AUC":         auc_per_class,
    }

    return y_pred, y_prob, metrics


def print_paper_table6(results, experiment_tag):
    """Print Paper Table 6 formatted comparison: Model | Accuracy | F1-Score."""
    print(f"\n  =======================================================")
    print(f"  TABLE 6: Comparison of Proposed Model and Baseline Models")
    print(f"           ({experiment_tag})")
    print(f"  =======================================================")
    print(f"  {'Model':<25s} {'Accuracy':>12s} {'F1-Score':>12s}")
    print("  " + "-" * 53)
    for name, info in results.items():
        disp_name = "Proposed Model (N-XGB)" if name == "N-XGB (Hybrid)" else name
        m = info["metrics"]
        print(f"  {disp_name:<25s} {m['Accuracy']:>12.{PRECISION}f} {m['F1_Weighted']:>12.{PRECISION}f}")


def print_comprehensive_metrics_table(results, experiment_tag):
    """Print complete multi-metric comparison table across all models."""
    print(f"\n  ========================================================================================================================")
    print(f"  COMPREHENSIVE METRICS COMPARISON -- {experiment_tag.upper()}")
    print(f"  ========================================================================================================================")
    print(f"  {'Model':<24s} {'Accuracy':>9s} {'Bal_Acc':>9s} {'Prec(W)':>9s} {'Rec(W)':>9s} {'Spec(W)':>9s} {'F1(W)':>9s} {'F1(M)':>9s} {'ROC_AUC':>9s} {'MCC':>9s} {'Kappa':>9s}")
    print("  " + "-" * 120)
    for name, info in results.items():
        disp_name = "Proposed (N-XGB)" if name == "N-XGB (Hybrid)" else name
        m = info["metrics"]
        auc_str = f"{m['ROC_AUC_Weighted']:>9.{PRECISION}f}" if not np.isnan(m['ROC_AUC_Weighted']) else "      N/A"
        print(f"  {disp_name:<24s} "
              f"{m['Accuracy']:>9.{PRECISION}f} "
              f"{m['Balanced_Accuracy']:>9.{PRECISION}f} "
              f"{m['Precision_Weighted']:>9.{PRECISION}f} "
              f"{m['Recall_Weighted']:>9.{PRECISION}f} "
              f"{m['Specificity_Weighted']:>9.{PRECISION}f} "
              f"{m['F1_Weighted']:>9.{PRECISION}f} "
              f"{m['F1_Macro']:>9.{PRECISION}f} "
              f"{auc_str} "
              f"{m['MCC']:>9.{PRECISION}f} "
              f"{m['Kappa']:>9.{PRECISION}f}")


def print_per_class_metrics_table(results, experiment_tag, target_le):
    """Print per-class breakdown (Precision, Recall, Specificity, F1, AUC) for each model."""
    print(f"\n  ========================================================================================================================")
    print(f"  PER-CLASS METRICS BREAKDOWN -- {experiment_tag.upper()}")
    print(f"  ========================================================================================================================")

    for name, info in results.items():
        disp_name = "Proposed Model (N-XGB)" if name == "N-XGB (Hybrid)" else name
        m = info["metrics"]
        print(f"\n  >>> Model: {disp_name}")
        print(f"      {'Disaster Class':<18s} {'Precision':>12s} {'Recall (Sens)':>14s} {'Specificity':>13s} {'F1-Score':>12s} {'ROC-AUC':>12s}")
        print("      " + "-" * 75)
        for c in range(N_CLASSES):
            cls_name = target_le.inverse_transform([c])[0]
            p = m["Per_Class_Precision"][c]
            r = m["Per_Class_Recall"][c]
            s = m["Per_Class_Specificity"][c]
            f = m["Per_Class_F1"][c]
            a = m["Per_Class_AUC"][c]
            a_str = f"{a:>12.{PRECISION}f}" if not np.isnan(a) else "         N/A"
            print(f"      {cls_name:<18s} {p:>12.{PRECISION}f} {r:>14.{PRECISION}f} {s:>13.{PRECISION}f} {f:>12.{PRECISION}f} {a_str}")


def print_overfit_check(results, experiment_tag):
    """Print train vs test accuracy to flag potential overfitting."""
    print(f"\n  Overfitting Check ({experiment_tag}):")
    print(f"  {'Model':<25s} {'Train Acc':>12s} {'Test Acc':>12s} {'Gap':>10s}  {'Status'}")
    print("  " + "-" * 73)
    for name, info in results.items():
        disp_name = "Proposed Model (N-XGB)" if name == "N-XGB (Hybrid)" else name
        tr = info.get("train_acc", 0.0)
        te = info["metrics"]["Accuracy"]
        gap = tr - te
        status = "OK" if gap < 0.05 else "WARNING: possible overfit"
        print(f"  {disp_name:<25s} {tr:>12.{PRECISION}f} {te:>12.{PRECISION}f} {gap:>+10.{PRECISION}f}  {status}")


def plot_confusion_matrices(results, experiment_tag, y_te):
    """Plot confusion matrices for all models side by side."""
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
        ax.set_ylabel("Actual")
    fig.suptitle(f"Confusion Matrices -- {experiment_tag}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, f"cm_{experiment_tag.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


def plot_combined_roc(results, experiment_tag, y_te):
    """Plot multi-class ROC curves for all models matching Paper Figures 6 & 9."""
    y_te_bin = label_binarize(y_te, classes=list(range(N_CLASSES)))
    model_colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, (name, info) in enumerate(results.items()):
        y_prob = info["y_prob"]
        if y_prob is None:
            continue
        # Compute macro-average ROC across all 3 classes
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

        label_name = f"Proposed Model (N-XGB)" if name == "N-XGB (Hybrid)" else name
        ax.plot(all_fpr, mean_tpr, color=model_colors[idx % len(model_colors)],
                lw=2, label=f"{label_name} (AUC = {macro_auc:.{PRECISION}f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.50000)")
    ax.set_title(f"Multiclass ROC-AUC -- {experiment_tag}", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, f"roc_{experiment_tag.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


def plot_per_class_roc_nxgb(nxgb_info, experiment_tag, y_te):
    """Plot per-class ROC curves for N-XGB (matching Paper Figure 7)."""
    y_te_bin = label_binarize(y_te, classes=list(range(N_CLASSES)))
    y_prob = nxgb_info["y_prob"]
    if y_prob is None:
        return

    class_colors = ["#e74c3c", "#3498db", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for c in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_te_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=class_colors[c], lw=2,
                label=f"{CLASSES[c]} (AUC = {roc_auc:.{PRECISION}f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_title(f"ROC Curve for N-XGB Model on {experiment_tag}", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, f"roc_nxgb_{experiment_tag.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


def run_experiment(experiment_tag, X_tr_raw, y_tr_raw, X_te_raw, y_te,
                   X_tr_nn, y_tr_nn, X_te_nn):
    """
    Run GridSearchCV + comprehensive evaluation for all models:
    - Baseline models (LR, SVM, RF, k-NN): trained on scaled input features
    - N-XGB (Hybrid): trained on NN-extracted features H
    """
    print_header(f"EXPERIMENT: {experiment_tag}")
    models_grids = get_models_and_grids()
    results = {}

    for name, (estimator, param_grid) in models_grids.items():
        print(f"\n  --- {name} ---")

        # Proposed N-XGB uses NN-extracted embeddings H; baseline models use raw features
        if name == "N-XGB (Hybrid)":
            X_tr_use, y_tr_use = X_tr_nn, y_tr_nn
            X_te_use = X_te_nn
        else:
            X_tr_use, y_tr_use = X_tr_raw, y_tr_raw
            X_te_use = X_te_raw

        t0 = time.time()
        best_model = run_gridsearch(name, estimator, param_grid, X_tr_use, y_tr_use)
        elapsed = time.time() - t0
        print(f"    GridSearch time: {elapsed:.1f}s")

        y_pred, y_prob, metrics = evaluate_model(best_model, X_te_use, y_te)

        # Train accuracy for overfitting check
        train_acc = accuracy_score(y_tr_use, best_model.predict(X_tr_use))

        results[name] = {
            "y_pred": y_pred, "y_prob": y_prob,
            "metrics": metrics, "train_acc": train_acc,
            "best_model": best_model,
        }
        print(f"    Test Accuracy : {metrics['Accuracy']:.{PRECISION}f} | Balanced Acc: {metrics['Balanced_Accuracy']:.{PRECISION}f} | F1 (Weighted): {metrics['F1_Weighted']:.{PRECISION}f}")

    return results


# =========================================================================
# MAIN PIPELINE
# =========================================================================

if __name__ == "__main__":
    total_start = time.time()

    # =====================================================================
    # STAGE 1: DATA PREPROCESSING (Paper Section III.B)
    # =====================================================================
    print_header("STAGE 1: Data Preprocessing (Section III.B)")

    # 1a. Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # 1b. Filter to relevant disaster types: Wildfire, Flood, Earthquake (Eq. 1)
    df = df[df[TARGET_COL].isin(CLASSES)].reset_index(drop=True)
    print(f"  Filtered to {CLASSES}: {df.shape[0]} samples")

    # 1c. Drop non-predictive / target-leaking columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 1d. Separate target
    y_series = df.pop(TARGET_COL)

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"  Numerical features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

    # 1e. Handling missing values (Eq. 2 & Eq. 3)
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # 1f. Encoding categorical variables (Eq. 4)
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Encode target categories
    target_le = LabelEncoder()
    target_le.fit(CLASSES)
    y = target_le.transform(y_series)
    print(f"  Target encoding: {dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}")

    X = df.values.astype(np.float32)
    print(f"  Preprocessed feature matrix: {X.shape}")

    # 1g. Holdout Validation Partition (70:30, Stratified - Section IV.B)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )
    print(f"  Holdout Split (70:30): Train = {X_train_raw.shape[0]} | Test = {X_test_raw.shape[0]}")

    # 1h. Normalizing numerical features with StandardScaler (Eq. 5)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)
    print("  StandardScaler: fit on train, transform on test [OK - no leakage]")

    # =====================================================================
    # STAGE 2: NEURAL NETWORK FEATURE EXTRACTION (Section III.D.1 & IV.A)
    # =====================================================================
    print_header("STAGE 2: PyTorch NN Feature Extraction (Algorithm 1 & Figure 1)")

    # Train NN feature extractor on training partition and extract latent embeddings H
    H_train_raw, H_test_scaled = train_nn_and_extract(
        X_train_scaled, y_train, X_test_scaled, label="Train Split"
    )

    # =====================================================================
    # STAGE 3: DATA BALANCING WITH SMOTE (Section III.B.4 & Algorithm 1)
    # =====================================================================
    print_header("STAGE 3: Class Balancing with SMOTE (Eq. 6-7)")

    # --- Imbalanced Data Sets (Raw, no SMOTE) ---
    X_train_imb = X_train_scaled.copy()
    y_train_imb = y_train.copy()
    H_train_imb = H_train_raw.copy()
    print_class_dist(y_train_imb, target_le, "(Imbalanced Training Set)")

    # --- Balanced Data Sets (SMOTE applied on training data only) ---
    smote_raw = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote_raw.fit_resample(X_train_scaled, y_train)

    smote_nn = SMOTE(random_state=42)
    H_train_bal, y_train_nn_bal = smote_nn.fit_resample(H_train_raw, y_train)
    print_class_dist(y_train_bal, target_le, "(Balanced Training Set - SMOTE)")

    print(f"\n  Imbalanced train samples : {X_train_imb.shape[0]}")
    print(f"  Balanced train samples   : {X_train_bal.shape[0]}")
    print(f"  Test set samples         : {X_test_scaled.shape[0]} (untouched holdout)")

    # =====================================================================
    # STAGE 4: MODEL TRAINING & HYPERPARAMETER TUNING (GridSearchCV)
    # =====================================================================
    print_header("STAGE 4: Model Training & Tuning (GridSearchCV 5-Fold)")

    # --- Experiment 1: Performance on Balanced Dataset (SMOTE) ---
    results_bal = run_experiment(
        "Balanced (SMOTE)",
        X_train_bal, y_train_bal, X_test_scaled, y_test,
        H_train_bal, y_train_nn_bal, H_test_scaled,
    )

    # --- Experiment 2: Performance on Imbalanced Dataset (Raw) ---
    results_imb = run_experiment(
        "Imbalanced (Raw)",
        X_train_imb, y_train_imb, X_test_scaled, y_test,
        H_train_imb, y_train_imb, H_test_scaled,
    )

    # =====================================================================
    # STAGE 5: RESULTS & DISCUSSION (Comprehensive Evaluation)
    # =====================================================================
    print_header("STAGE 5: Results Evaluation & Comparison")

    # --- Paper Table 6 Comparison ---
    print_paper_table6(results_bal, "Balanced Dataset")
    print_paper_table6(results_imb, "Imbalanced Dataset")

    # --- Comprehensive Multi-Metric Comparison Tables ---
    print_comprehensive_metrics_table(results_bal, "Balanced (SMOTE)")
    print_comprehensive_metrics_table(results_imb, "Imbalanced (Raw)")

    # --- Per-Class Metrics Breakdowns ---
    print_per_class_metrics_table(results_bal, "Balanced (SMOTE)", target_le)
    print_per_class_metrics_table(results_imb, "Imbalanced (Raw)", target_le)

    # --- Overfitting Checks ---
    print_overfit_check(results_bal, "Balanced")
    print_overfit_check(results_imb, "Imbalanced")

    # --- Classification Reports (Table 5 style) ---
    for tag, results in [("Balanced", results_bal), ("Imbalanced", results_imb)]:
        print(f"\n  =======================================================")
        print(f"  TABLE 5 / Classification Reports ({tag})")
        print(f"  =======================================================")
        for name, info in results.items():
            disp_name = "Proposed Model (N-XGB)" if name == "N-XGB (Hybrid)" else name
            print(f"\n  [{disp_name}]")
            print(classification_report(
                y_test, info["y_pred"], target_names=CLASSES, digits=PRECISION
            ))

    # --- Confusion Matrices (Paper Figure 5 & Figure 8 + Comparisons) ---
    print("\n  Generating confusion matrix visualizations ...")
    plot_confusion_matrices(results_bal, "Balanced", y_test)
    plot_confusion_matrices(results_imb, "Imbalanced", y_test)

    # --- Combined ROC Curves across models (Paper Figure 6 & Figure 9) ---
    print("\n  Generating combined ROC-AUC curves ...")
    plot_combined_roc(results_bal, "Balanced", y_test)
    plot_combined_roc(results_imb, "Imbalanced", y_test)

    # --- Per-class ROC curves for N-XGB (Paper Figure 7) ---
    print("\n  Generating per-class ROC curves for N-XGB ...")
    plot_per_class_roc_nxgb(results_bal["N-XGB (Hybrid)"], "Balanced", y_test)
    plot_per_class_roc_nxgb(results_imb["N-XGB (Hybrid)"], "Imbalanced", y_test)

    # --- Summary ---
    total_elapsed = time.time() - total_start
    print_header("PIPELINE COMPLETE")
    print(f"  Total runtime     : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Models evaluated  : {len(results_bal)} models x 2 experiments")
    print(f"  Test samples      : {X_test_scaled.shape[0]} (70:30 holdout split)")
    print()
