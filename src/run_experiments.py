"""
Leakage-Free Financial Fraud Detection Experiments
==================================================

Run all experiments used in the capstone project.

Expected input files:
- data/raw/train_dataset.csv
- data/raw/test_dataset.csv

Example:
    python src/run_experiments.py --train data/raw/train_dataset.csv --test data/raw/test_dataset.csv

Outputs:
- results/metrics/*.csv
- results/figures/**/*.png

Note:
- The original dataset CSV files are not included in this repository.
- Place train_dataset.csv and test_dataset.csv under data/raw/ before execution.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


CAT_COLS = [
    "Transaction_Type", "Device_Type", "Location", "Merchant_Category",
    "Card_Type", "Authentication_Method",
]
DROP_COLS = [
    "Transaction_ID", "User_ID", "Timestamp", "prev_time", "user_avg_amount",
    "user_avg_distance", "user_main_device", "user_main_location",
]
LEAKAGE_COLS = ["Risk_Score", "Failed_Transaction_Count_7d"]
NEW_FEATURES = [
    "Amount_vs_User_Avg", "Distance_vs_User_Avg", "Transaction_Interval", "Velocity",
    "Weekend_Night", "Is_New_Device", "Is_New_Location", "Balance_to_Amount",
    "High_Daily_Count",
]


def ensure_dirs(out_dir: Path) -> None:
    for sub in [
        "metrics", "figures/confusion_matrix", "figures/roc_pr", "figures/threshold_tuning",
        "figures/feature_importance", "figures/shap_summary_plot", "figures/feature_distribution",
    ]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def base_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    return df


def feature_engineering(
    df: pd.DataFrame,
    user_stats: pd.DataFrame | None = None,
    user_device_map: Dict[str, str] | None = None,
    user_loc_map: Dict[str, str] | None = None,
    is_train: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """Create behavior-based features using train statistics only.

    For leakage prevention, user-level statistics are computed on train data and
    applied to test data.
    """
    df = df.copy()

    if is_train:
        user_stats = df.groupby("User_ID").agg(
            user_avg_amount=("Transaction_Amount", "mean"),
            user_avg_distance=("Transaction_Distance", "mean"),
        ).reset_index()
        user_device_map = df.groupby("User_ID")["Device_Type"].agg(lambda x: x.mode()[0]).to_dict()
        user_loc_map = df.groupby("User_ID")["Location"].agg(lambda x: x.mode()[0]).to_dict()

    if user_stats is None or user_device_map is None or user_loc_map is None:
        raise ValueError("Train-based user statistics/maps are required for test feature engineering.")

    df = df.merge(user_stats, on="User_ID", how="left")
    df["user_avg_amount"] = df["user_avg_amount"].fillna(df["Transaction_Amount"].median())
    df["user_avg_distance"] = df["user_avg_distance"].fillna(df["Transaction_Distance"].median())

    df["Amount_vs_User_Avg"] = df["Transaction_Amount"] / (df["user_avg_amount"] + 1e-6)
    df["Distance_vs_User_Avg"] = df["Transaction_Distance"] / (df["user_avg_distance"] + 1e-6)

    df = df.sort_values(["User_ID", "Timestamp"])
    df["prev_time"] = df.groupby("User_ID")["Timestamp"].shift(1)
    df["Transaction_Interval"] = ((df["Timestamp"] - df["prev_time"]).dt.total_seconds() / 60).fillna(-1)
    df["Velocity"] = np.where(
        df["Transaction_Interval"] <= 0,
        0,
        df["Transaction_Distance"] / (df["Transaction_Interval"] + 1e-6),
    )
    df["Weekend_Night"] = ((df["DayOfWeek"] >= 5) & (df["Hour"].between(0, 5))).astype(int)

    df["user_main_device"] = df["User_ID"].map(user_device_map).fillna("Unknown")
    df["Is_New_Device"] = (df["Device_Type"] != df["user_main_device"]).astype(int)

    df["user_main_location"] = df["User_ID"].map(user_loc_map).fillna("Unknown")
    df["Is_New_Location"] = (df["Location"] != df["user_main_location"]).astype(int)

    df["Balance_to_Amount"] = df["Transaction_Amount"] / (df["Account_Balance"] + 1e-6)
    df["High_Daily_Count"] = (df["Daily_Transaction_Count"] >= 5).astype(int)

    return df, user_stats, user_device_map, user_loc_map


def safe_label_encode(train: pd.DataFrame, test: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for col in cat_cols:
        if col not in train.columns or col not in test.columns:
            continue
        vals = train[col].astype(str).fillna("Unknown")
        mapping = {v: i for i, v in enumerate(sorted(vals.unique()))}
        train[col] = vals.map(mapping).fillna(-1).astype(int)
        test[col] = test[col].astype(str).fillna("Unknown").map(mapping).fillna(-1).astype(int)
    return train, test


def encode_and_split(train: pd.DataFrame, test: pd.DataFrame):
    tr, te = safe_label_encode(train, test, CAT_COLS)
    tr = tr.drop(columns=[c for c in DROP_COLS if c in tr.columns])
    te = te.drop(columns=[c for c in DROP_COLS if c in te.columns])
    X_tr = tr.drop("Fraud_Label", axis=1).fillna(0).replace([np.inf, -np.inf], 0)
    y_tr = tr["Fraud_Label"].astype(int)
    X_te = te.drop("Fraud_Label", axis=1).fillna(0).replace([np.inf, -np.inf], 0)
    y_te = te["Fraud_Label"].astype(int)
    return X_tr, y_tr, X_te, y_te


def evaluate_proba(proba: np.ndarray, y_true: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "Threshold": threshold,
        "AUC-ROC": round(roc_auc_score(y_true, proba), 4),
        "AUC-PR": round(average_precision_score(y_true, proba), 4),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, pred), 4),
        "F1": round(f1_score(y_true, pred), 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def plot_cm(vals: Tuple[int, int, int, int], title: str, save_path: Path) -> None:
    tn, fp, fn, tp = vals
    arr = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{arr[i, j]:,}", ha="center", va="center",
                    fontsize=13, color="white" if arr[i, j] > arr.max() / 2 else "black")
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(proba: np.ndarray, y_true: pd.Series, title: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y_true, proba):.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6)
    axes[0].set_title(f"ROC Curve — {title}")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
    axes[1].plot(rec, prec, lw=2, label=f"AP={average_precision_score(y_true, proba):.4f}")
    axes[1].set_title(f"PR Curve — {title}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distribution(train_fe: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, feat in enumerate(NEW_FEATURES):
        ax = axes[i]
        for label, color in [(0, "steelblue"), (1, "tomato")]:
            vals = train_fe.loc[train_fe["Fraud_Label"] == label, feat].replace([np.inf, -np.inf], np.nan).dropna()
            ax.hist(vals, bins=40, alpha=0.5, color=color,
                    label="Normal" if label == 0 else "Fraud", density=True)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7)
    for j in range(len(NEW_FEATURES), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("New Feature Distribution: Fraud vs Normal", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "figures/feature_distribution/new_feature_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_supervised_experiments(X_tr, y_tr, X_te, y_te, out_dir: Path, skip_shap: bool = False):
    base_cols = [c for c in X_tr.columns if c not in NEW_FEATURES]
    experiments = [
        ("Baseline_no_leakage", X_tr[base_cols], y_tr, X_te[base_cols], y_te),
        ("FE_added_no_leakage", X_tr, y_tr, X_te, y_te),
    ]
    models = {
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    }

    results = []
    trained = {}
    for exp_label, Xtr, Ytr, Xte, Yte in experiments:
        for model_name, model in models.items():
            m = copy.deepcopy(model)
            m.fit(Xtr, Ytr)
            proba = m.predict_proba(Xte)[:, 1]
            r = evaluate_proba(proba, Yte, threshold=0.5)
            r.update({"Model": model_name, "Dataset": exp_label})
            results.append(r)
            trained[(model_name, exp_label)] = (m, Xte, Yte, proba)
            safe = f"{model_name}_{exp_label}"
            plot_cm((r["TN"], r["FP"], r["FN"], r["TP"]), f"{model_name} ({exp_label})",
                    out_dir / f"figures/confusion_matrix/cm_{safe}.png")
            plot_roc_pr(proba, Yte, f"{model_name} ({exp_label})",
                        out_dir / f"figures/roc_pr/roc_pr_{safe}.png")

    metrics_df = pd.DataFrame(results)
    cols = ["Dataset", "Model", "Threshold", "AUC-ROC", "AUC-PR", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"]
    metrics_df[cols].to_csv(out_dir / "metrics/model_comparison.csv", index=False)

    plot_compare_roc_pr(trained, experiments, out_dir)
    plot_cm_grid(results, out_dir)
    run_threshold_tuning(trained, out_dir)
    run_feature_importance_and_shap(trained, out_dir, skip_shap=skip_shap)
    return trained


def plot_compare_roc_pr(trained, experiments, out_dir: Path) -> None:
    colors = ["steelblue", "tomato", "seagreen"]
    for exp_label, *_ in experiments:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for model_name, color in zip(["XGBoost", "LightGBM", "RandomForest"], colors):
            _, _, Yte, proba = trained[(model_name, exp_label)]
            fpr, tpr, _ = roc_curve(Yte, proba)
            prec, rec, _ = precision_recall_curve(Yte, proba)
            axes[0].plot(fpr, tpr, lw=2, color=color, label=f"{model_name} (AUC={roc_auc_score(Yte, proba):.4f})")
            axes[1].plot(rec, prec, lw=2, color=color, label=f"{model_name} (AP={average_precision_score(Yte, proba):.4f})")
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].set_title(f"ROC Curve ({exp_label})"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
        axes[1].set_title(f"PR Curve ({exp_label})"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"figures/roc_pr/roc_pr_compare_{exp_label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_cm_grid(results: List[Dict[str, Any]], out_dir: Path) -> None:
    exp_names = ["Baseline_no_leakage", "FE_added_no_leakage"]
    model_names = ["XGBoost", "LightGBM", "RandomForest"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    for row_i, exp_label in enumerate(exp_names):
        for col_j, model_name in enumerate(model_names):
            r = next(x for x in results if x["Dataset"] == exp_label and x["Model"] == model_name)
            arr = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
            ax = axes[row_i][col_j]
            im = ax.imshow(arr, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{arr[i, j]:,}", ha="center", va="center",
                            fontsize=11, color="white" if arr[i, j] > arr.max()/2 else "black")
            ax.set_title(f"{model_name}\n{exp_label}", fontsize=9)
            plt.colorbar(im, ax=ax)
    plt.suptitle("Confusion Matrix Summary", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "figures/confusion_matrix/cm_all_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_threshold_tuning(trained, out_dir: Path) -> None:
    thresholds = np.round(np.arange(0.1, 0.95, 0.05), 2)
    rows = []
    for exp_label in ["Baseline_no_leakage", "FE_added_no_leakage"]:
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        for ax, model_name in zip(axes, ["XGBoost", "LightGBM", "RandomForest"]):
            _, _, Yte, proba = trained[(model_name, exp_label)]
            local = []
            for thr in thresholds:
                pred = (proba >= thr).astype(int)
                row = {
                    "Experiment": exp_label,
                    "Model": model_name,
                    "Threshold": thr,
                    "Precision": round(precision_score(Yte, pred, zero_division=0), 4),
                    "Recall": round(recall_score(Yte, pred), 4),
                    "F1": round(f1_score(Yte, pred), 4),
                }
                local.append(row); rows.append(row)
            df_thr = pd.DataFrame(local)
            best = df_thr.loc[df_thr["F1"].idxmax()]
            ax.plot(df_thr["Threshold"], df_thr["Precision"], marker="o", label="Precision")
            ax.plot(df_thr["Threshold"], df_thr["Recall"], marker="s", label="Recall")
            ax.plot(df_thr["Threshold"], df_thr["F1"], marker="^", label="F1")
            ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default=0.5")
            ax.axvline(best["Threshold"], color="tomato", linestyle=":", label=f"Best F1={best['F1']} @ {best['Threshold']}")
            ax.set_title(f"Threshold — {model_name}\n({exp_label})")
            ax.set_xlabel("Threshold"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"figures/threshold_tuning/threshold_compare_{exp_label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "metrics/threshold_tuning_all.csv", index=False)


def run_feature_importance_and_shap(trained, out_dir: Path, skip_shap: bool = False) -> None:
    for (model_name, exp_label), (model, Xte, _, _) in trained.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({"feature": Xte.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            imp.to_csv(out_dir / f"metrics/feature_importance_{model_name}_{exp_label}.csv", index=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            top = imp.head(20)
            ax.barh(top["feature"][::-1], top["importance"][::-1])
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance — {model_name} ({exp_label})")
            plt.tight_layout()
            plt.savefig(out_dir / f"figures/feature_importance/fi_{model_name}_{exp_label}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        if not skip_shap and shap is not None:
            sample_idx = np.random.choice(len(Xte), min(1000, len(Xte)), replace=False)
            X_sample = Xte.iloc[sample_idx]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif getattr(shap_values, "ndim", 0) == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values
            shap.summary_plot(sv, X_sample, show=False, max_display=X_sample.shape[1])
            fig = plt.gcf()
            fig.set_size_inches(14, 10)
            plt.subplots_adjust(right=0.85, bottom=0.1)
            plt.savefig(out_dir / f"figures/shap_summary_plot/shap_{model_name}_{exp_label}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def run_autoencoder(X_tr, y_tr, X_te, y_te, out_dir: Path) -> None:
    if torch is None or nn is None:
        print("PyTorch is not installed. Skipping Autoencoder.")
        return

    X_tr_normal = X_tr[y_tr == 0].copy()
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_normal)
    X_te_sc = scaler.transform(X_te)
    X_tr_tensor = torch.FloatTensor(X_tr_sc)
    X_te_tensor = torch.FloatTensor(X_te_sc)
    n_feat = X_tr_sc.shape[1]

    class Autoencoder(nn.Module):
        def __init__(self, n_feat: int):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(n_feat, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, n_feat))
        def forward(self, x):
            return self.decoder(self.encoder(x))

    ae = Autoencoder(n_feat)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr_tensor), batch_size=256, shuffle=True)
    losses = []
    ae.train()
    for _ in range(50):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad(); loss = criterion(ae(batch), batch); loss.backward(); optimizer.step(); epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, label="Train Loss")
    ax.set_title("Autoencoder Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "figures/autoencoder_train_loss.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    ae.eval()
    with torch.no_grad():
        recon = ae(X_te_tensor).numpy()
    recon_err = np.mean(np.square(X_te_sc - recon), axis=1)
    r = evaluate_proba(recon_err, y_te, threshold=np.percentile(recon_err, 95))
    pd.DataFrame([{**r, "Model": "Autoencoder", "Dataset": "FE_added_no_leakage"}]).to_csv(out_dir / "metrics/autoencoder_results.csv", index=False)
    plot_roc_pr(recon_err, y_te, "Autoencoder", out_dir / "figures/roc_pr/roc_pr_Autoencoder.png")


def run_stacking(trained, X_tr, y_tr, X_te, y_te, out_dir: Path) -> None:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = {"XGBoost": np.zeros(len(y_tr)), "LightGBM": np.zeros(len(y_tr)), "RandomForest": np.zeros(len(y_tr))}
    model_factories = {
        "XGBoost": lambda: XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": lambda: LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        "RandomForest": lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
    }
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr, y_tr), start=1):
        Xf_tr, Xf_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        yf_tr = y_tr.iloc[tr_idx]
        for name, make_model in model_factories.items():
            m = make_model(); m.fit(Xf_tr, yf_tr); oof[name][va_idx] = m.predict_proba(Xf_va)[:, 1]
        print(f"Stacking fold {fold}/5 completed")

    meta_tr = np.column_stack([oof["XGBoost"], oof["LightGBM"], oof["RandomForest"]])
    meta_te = np.column_stack([
        trained[("XGBoost", "FE_added_no_leakage")][0].predict_proba(X_te)[:, 1],
        trained[("LightGBM", "FE_added_no_leakage")][0].predict_proba(X_te)[:, 1],
        trained[("RandomForest", "FE_added_no_leakage")][0].predict_proba(X_te)[:, 1],
    ])
    meta_clf = LogisticRegression(max_iter=500)
    meta_clf.fit(meta_tr, y_tr)
    proba = meta_clf.predict_proba(meta_te)[:, 1]
    r = evaluate_proba(proba, y_te, threshold=0.5)
    pd.DataFrame([{**r, "Model": "Stacking", "Dataset": "FE_added_no_leakage"}]).to_csv(out_dir / "metrics/stacking_results.csv", index=False)
    plot_roc_pr(proba, y_te, "Stacking(XGB+LGBM+RF)", out_dir / "figures/roc_pr/roc_pr_Stacking.png")
    plot_cm((r["TN"], r["FP"], r["FN"], r["TP"]), "Stacking(XGB+LGBM+RF)", out_dir / "figures/confusion_matrix/cm_Stacking.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/train_dataset.csv")
    parser.add_argument("--test", default="data/raw/test_dataset.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-autoencoder", action="store_true")
    parser.add_argument("--skip-stacking", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dirs(out_dir)

    train_path = Path(args.train)
    test_path = Path(args.test)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Dataset files are missing. Place train_dataset.csv and test_dataset.csv in data/raw/, "
            "or pass paths with --train and --test."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = train.drop(columns=[c for c in LEAKAGE_COLS if c in train.columns])
    test = test.drop(columns=[c for c in LEAKAGE_COLS if c in test.columns])
    train = base_preprocess(train)
    test = base_preprocess(test)
    train_fe, user_stats, dev_map, loc_map = feature_engineering(train, is_train=True)
    test_fe, _, _, _ = feature_engineering(test, user_stats, dev_map, loc_map, is_train=False)
    plot_feature_distribution(train_fe, out_dir)

    X_tr, y_tr, X_te, y_te = encode_and_split(train_fe, test_fe)
    trained = run_supervised_experiments(X_tr, y_tr, X_te, y_te, out_dir, skip_shap=args.skip_shap)

    if not args.skip_autoencoder:
        run_autoencoder(X_tr, y_tr, X_te, y_te, out_dir)
    if not args.skip_stacking:
        run_stacking(trained, X_tr, y_tr, X_te, y_te, out_dir)

    print("All experiments completed. Outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
