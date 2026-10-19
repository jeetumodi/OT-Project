"""Training-only Mountain Gazelle Optimization (MGO) hyperparameter tuner for N-XGB.

The original nxgb_classifier.py is intentionally untouched. MGO is an outer-loop
optimizer: each gazelle decodes to NN and XGBoost hyperparameters, is scored with
stratified CV on the training partition only, then the best configuration is fitted
once and reported on the held-out test partition.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
TARGET, CLASSES = "Disaster Type", ["Wildfire", "Flood", "Earthquake"]
DROP_COLUMNS = ["DisNo.", "External IDs", "Event Name", "Entry Date", "Last Update", "Admin Units", "GADM Admin Units", "Classification Key", "Historic", "Disaster Group", "Disaster Subgroup", "Disaster Subtype", "Magnitude Scale", "Origin", "Associated Types", "River Basin"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def device_label() -> str:
    """Return the currently selected execution device for every progress line."""
    return f"cuda:{torch.cuda.get_device_name(0)}" if DEVICE.type == "cuda" else "cpu"

def log(message: str) -> None:
    print(f"[device={device_label()}] {message}", flush=True)

def verify_xgboost_cuda() -> None:
    """Fail fast when the installed XGBoost package cannot use the selected GPU."""
    if DEVICE.type != "cuda":
        log("XGBoost CUDA preflight skipped because CPU was selected")
        return
    probe_x = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.], [2., 2.], [2., 0.]], dtype=np.float32)
    probe_y = np.array([0, 1, 0, 1, 1, 0])
    probe = XGBClassifier(n_estimators=1, max_depth=2, tree_method="hist", device="cuda", n_jobs=1, verbosity=0)
    probe.fit(probe_x, probe_y)
    config = probe.get_booster().save_config().lower()
    if "cuda" not in config:
        raise RuntimeError("XGBoost CUDA preflight failed. Install the GPU-capable 'xgboost' package (not 'xgboost-cpu').")
    log("XGBoost CUDA preflight verified")

@dataclass(frozen=True)
class SearchConfig:
    population_size: int = 8
    max_iterations: int = 5
    cv_folds: int = 3
    nn_epochs: int = 10
    seed: int = 42

class TunableFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_1: int, hidden_2: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_1), nn.BatchNorm1d(hidden_1), nn.ReLU(), nn.Dropout(dropout))
        self.layer2 = nn.Sequential(nn.Linear(hidden_1, hidden_2), nn.BatchNorm1d(hidden_2), nn.ReLU(), nn.Dropout(dropout))
        self.hidden = nn.Sequential(nn.Linear(hidden_2, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU())
        self.classifier = nn.Linear(embedding_dim, len(CLASSES))
    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); h = self.hidden(x)
        return self.classifier(h), h

def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "public_emdat.csv", low_memory=False)
    df = df.loc[df[TARGET].isin(CLASSES)].copy()
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df], errors="ignore")

def make_preprocessor(X: pd.DataFrame) -> Pipeline:
    """Baseline-compatible numeric representation, fitted only on each training fold.

    The original project label-encodes categorical columns and then standardizes the
    entire matrix.  This uses fold-local ordinal encoding (unknown categories become
    -1) followed by scaling, avoiding validation/test leakage while preserving the
    compact tabular input size expected by the N-XGB architecture.
    """
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    columns = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                   ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical),
    ], verbose_feature_names_out=False)
    return Pipeline([("columns", columns), ("scale", StandardScaler())])

def _choice(value: float, choices: list[int | float]):
    return choices[min(int(np.floor(np.clip(value, 0, .999999) * len(choices))), len(choices) - 1)]

def decode_hyperparameters(position: np.ndarray, quick: bool = False) -> dict:
    """Map an MGO position to reproducible NN/XGBoost settings.

    The search is deliberately evaluated only by cross-validation on the training
    partition.  The held-out test partition is never used to select a position.
    """
    estimators = [25, 50] if quick else [100, 200, 300]
    return {
        "nn_learning_rate": float(10 ** (-3.5 + 1.3 * position[0])),
        "nn_dropout": float(.15 + .25 * position[1]),
        "nn_hidden_1": int(_choice(position[2], [128, 192, 256])),
        "nn_hidden_2": int(_choice(position[3], [64, 96, 128])),
        "embedding_dim": int(_choice(position[4], [32, 48, 64])),
        "batch_size": int(_choice(position[5], [32, 64])),
        "nn_weight_decay": float(_choice(position[6], [0., 1e-5, 1e-4, 1e-3])),
        "xgb_learning_rate": float(_choice(position[7], [.01, .03, .05, .1])),
        "xgb_max_depth": int(_choice(position[8], [4, 5, 7, 9])),
        "xgb_n_estimators": int(_choice(position[9], estimators)),
        "xgb_min_child_weight": int(_choice(position[10], [1, 3, 5])),
        "xgb_subsample": float(_choice(position[11], [.7, .85, 1.])),
        "xgb_colsample_bytree": float(_choice(position[12], [.7, .85, 1.])),
    }

def train_embeddings(X_train, y_train, X_eval, params: dict, epochs: int, seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TunableFeatureExtractor(X_train.shape[1], params["nn_hidden_1"], params["nn_hidden_2"], params["embedding_dim"], params["nn_dropout"]).to(DEVICE)
    if DEVICE.type == "cuda" and not next(model.parameters()).is_cuda:
        raise RuntimeError("CUDA was requested but the neural-network parameters are not on the GPU.")
    log("PyTorch neural network verified on CUDA" if DEVICE.type == "cuda" else "PyTorch neural network verified on CPU")
    data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(data, batch_size=params["batch_size"], shuffle=True, pin_memory=DEVICE.type == "cuda", generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["nn_learning_rate"], weight_decay=params["nn_weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(); logits, _ = model(xb); loss = loss_fn(logits, yb); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        train_tensor = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
        eval_tensor = torch.tensor(X_eval, dtype=torch.float32, device=DEVICE)
        return model(train_tensor)[1].cpu().numpy(), model(eval_tensor)[1].cpu().numpy()

def fit_predict(X_train, y_train, X_eval, params: dict, cfg: SearchConfig, seed: int):
    h_train, h_eval = train_embeddings(X_train, y_train, X_eval, params, cfg.nn_epochs, seed)
    h_train, y_balanced = SMOTE(random_state=seed).fit_resample(h_train, y_train)
    model = XGBClassifier(objective="multi:softprob", num_class=len(CLASSES), eval_metric="mlogloss", random_state=seed, verbosity=0, n_jobs=1, tree_method="hist", device="cuda" if DEVICE.type == "cuda" else "cpu",
                          learning_rate=params["xgb_learning_rate"], max_depth=params["xgb_max_depth"], n_estimators=params["xgb_n_estimators"],
                          min_child_weight=params["xgb_min_child_weight"], subsample=params["xgb_subsample"], colsample_bytree=params["xgb_colsample_bytree"])
    fitted_model = model.fit(h_train, y_balanced)
    if DEVICE.type == "cuda":
        booster_config = fitted_model.get_booster().save_config().lower()
        if "cuda" not in booster_config:
            raise RuntimeError("CUDA was requested but XGBoost did not report a CUDA device after fitting.")
        log("XGBoost booster verified on CUDA")
    else:
        log("XGBoost booster verified on CPU")
    return fitted_model.predict(h_eval)

def score_position(position, X, y, cfg, quick, cache) -> float:
    params = decode_hyperparameters(position, quick); key = json.dumps(params, sort_keys=True)
    if key in cache:
        log(f"cache hit {params}"); return cache[key]
    log(f"scoring {params}")
    cv, scores = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed), []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
        transformer = make_preprocessor(X.iloc[train_idx])
        X_fold_train = transformer.fit_transform(X.iloc[train_idx]).astype(np.float32)
        X_fold_valid = transformer.transform(X.iloc[valid_idx]).astype(np.float32)
        prediction = fit_predict(X_fold_train, y[train_idx], X_fold_valid, params, cfg, cfg.seed + fold)
        scores.append(accuracy_score(y[valid_idx], prediction))
    fitness = 1 - float(np.mean(scores)); cache[key] = fitness
    log(f"complete: CV accuracy={1-fitness:.5f}; fitness={fitness:.5f}")
    return fitness

def mgo_candidates(population, best, iteration, maximum, rng):
    """Original Mountain Gazelle MGO: MSF, TSM, MH, and BMH candidate mechanisms."""
    count, dimension = population.shape; proposals = []
    for current in population:
        young = population[rng.permutation(count)[:int(np.ceil(count / 3))]]
        male = population[rng.integers(int(np.ceil(count / 3)), count)] * np.floor(rng.normal()) + young.mean(axis=0) * np.ceil(rng.normal())
        a = -1 - iteration / maximum; n1, n2 = rng.standard_normal(dimension), rng.standard_normal(dimension)
        coefficients = np.array([rng.random(dimension), np.full(dimension, a + 1 + rng.random()), a * rng.standard_normal(dimension), n1 * n2**2 * np.cos(2 * rng.random() * n1)])
        famine = rng.standard_normal(dimension) * np.exp(2 - 2 * iteration / maximum)
        distance = (np.abs(current) + np.abs(best)) * (2 * rng.random() - 1)
        integer = lambda: rng.integers(1, 3)
        proposals.extend([rng.random(dimension),
            best - np.abs((integer() * male - integer() * current) * famine) * coefficients[rng.integers(4)],
            male + coefficients[rng.integers(4)] + (integer() * best - integer() * population[rng.integers(count)]) * coefficients[rng.integers(4)],
            current - distance + (integer() * best - integer() * male) * coefficients[rng.integers(4)]])
    return np.clip(np.asarray(proposals), 0, 1)

def run_mgo(X, y, cfg, quick):
    rng, cache = np.random.default_rng(cfg.seed), {}
    population = rng.random((cfg.population_size, 13)); best_position, best_fitness, history = None, np.inf, []
    for iteration in range(1, cfg.max_iterations + 1):
        log(f"MGO {iteration}/{cfg.max_iterations}: scoring {len(population)} parent configurations")
        parent_fitness = np.array([score_position(p, X, y, cfg, quick, cache) for p in population])
        candidate = int(parent_fitness.argmin())
        if parent_fitness[candidate] < best_fitness: best_fitness, best_position = parent_fitness[candidate], population[candidate].copy()
        expanded = np.vstack([population, mgo_candidates(population, best_position, iteration, cfg.max_iterations, rng)])
        log(f"MGO {iteration}/{cfg.max_iterations}: scoring {len(expanded)} parent/candidate configurations")
        fitness = np.array([score_position(p, X, y, cfg, quick, cache) for p in expanded])
        survivors = np.argsort(fitness)[:cfg.population_size]; population = expanded[survivors]
        if fitness[survivors[0]] < best_fitness: best_fitness, best_position = fitness[survivors[0]], population[0].copy()
        history.append({"iteration": iteration, "best_cv_accuracy": 1-best_fitness, "unique_configurations": len(cache), **decode_hyperparameters(best_position, quick)})
        log(f"MGO {iteration}/{cfg.max_iterations}: best CV accuracy={1-best_fitness:.5f}")
    return decode_hyperparameters(best_position, quick), pd.DataFrame(history)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--quick", action="store_true"); parser.add_argument("--full", action="store_true"); args = parser.parse_args()
    if args.quick and args.full: parser.error("Choose only --quick or --full.")
    cfg = SearchConfig(population_size=3 if args.quick else (20 if args.full else 8), max_iterations=1 if args.quick else (30 if args.full else 5), cv_folds=2 if args.quick else 3, nn_epochs=2 if args.quick else 50)
    df = load_data(); X, y = df.drop(columns=TARGET), pd.Categorical(df[TARGET], categories=CLASSES).codes.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=cfg.seed, stratify=y)
    log(f"Train={len(y_train)}, test={len(y_test)}, columns={X_train.shape[1]}; config={asdict(cfg)}")
    verify_xgboost_cuda()
    started = time.perf_counter(); best, history = run_mgo(X_train, y_train, cfg, args.quick); elapsed = time.perf_counter()-started
    log(f"Final held-out evaluation with best training-CV configuration: {best}")
    transformer = make_preprocessor(X_train); train_array = transformer.fit_transform(X_train).astype(np.float32); test_array = transformer.transform(X_test).astype(np.float32)
    prediction = fit_predict(train_array, y_train, test_array, best, cfg, cfg.seed)
    metrics = {"accuracy": accuracy_score(y_test, prediction), "precision_weighted": precision_score(y_test, prediction, average="weighted", zero_division=0), "recall_weighted": recall_score(y_test, prediction, average="weighted", zero_division=0), "f1_weighted": f1_score(y_test, prediction, average="weighted", zero_division=0), "mgo_seconds": elapsed, **best}
    # Keep accuracy-optimized results separate from the completed weighted-F1 run.
    output = ROOT / "mgo_accuracy_tuning_results"; output.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output / "best_configuration_and_test_metrics.csv", index=False); history.to_csv(output / "mgo_tuning_history.csv", index=False)
    (output / "run_configuration.json").write_text(json.dumps({"search": asdict(cfg), "objective": "cv_accuracy", "quick": args.quick, "best": best}, indent=2), encoding="utf-8")
    for name, value in metrics.items(): log(f"{name}={value}")
    log(f"Saved outputs in {output}")

if __name__ == "__main__": main()
