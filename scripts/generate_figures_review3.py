"""
Regenerate all 6 figures with unified English labels, no on-figure titles,
and descriptions extracted to docs/figures/desc.txt.
Run with: source common/Scripts/activate && python scripts/generate_figures_v2.py
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Global style settings
# ------------------------------------------------------------------
OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DESC_PATH = OUTPUT_DIR / "desc.txt"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="tab10", font="DejaVu Sans")

DPI = 300

# Container for descriptions: filename -> description
descriptions: dict[str, str] = {}

# ------------------------------------------------------------------
# Load data and reproduce split
# ------------------------------------------------------------------
DATA_DIR = Path("data/runs/main_seed42")  # 先运行主实验生成该目录（见 README）
SEMANTIC_DIR = Path("data/runs/semantic_deepseek")  # 语义对照实验输出目录

# 从原始采集文件重建特征（与主实验同一代码路径）。
# 不读 cleaned_features.csv 快照：CSV 往返浮点噪声会被提升树混沌放大
# （实测同协议 MAE 21.9899 -> 22.2192），图内数值需逐位可复现。
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.research.pipeline import (
    _apply_reference_group_medians,
    _build_features,
    _clean_and_join,
    _compute_region_ai_medians,
    _load_communities,
    _load_houses,
)

df = _build_features(
    _clean_and_join(
        _load_houses(Path("data/full/nanchang_houses.jsonl")),
        _load_communities(Path("data/full/nanchang_communities.jsonl")),
    )[0]
)
df = df.sort_values(by=["extracted_at", "house_id"], na_position="first").reset_index(drop=True)

test_count = max(int(len(df) * 0.20), 1)
train_df = df.iloc[:-test_count].copy()
test_df = df.iloc[-test_count:].copy()

train_df, test_df = _apply_reference_group_medians(
    train_df, test_df, _compute_region_ai_medians(Path("data/full/nanchang_communities.jsonl"))
)

NUMERIC_FEATURES = [
    "area_sqm", "room_count", "hall_count", "bath_count", "total_floors",
    "house_certificate_years",
    "floor_level_score", "floor_ratio_proxy", "area_per_room", "layout_density",
    "poi_bank_count", "poi_bus_count", "poi_subway_count", "poi_school_count",
    "poi_restaurant_count", "poi_shop_count", "poi_hospital_count", "poi_total_count",
    "accessibility_index", "poi_balance_score", "transit_medical_ratio",
    "education_commerce_ratio", "area_layout_efficiency", "room_area_interaction",
    "bath_room_ratio", "region_accessibility_interaction", "floor_area_interaction",
]
CATEGORICAL_FEATURES = [
    "region_slug", "decoration_type",
    "orientation", "floor_level", "has_elevator", "is_unique_housing", "house_layout",
]

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df["total_price_wan"].values
X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_test = test_df["total_price_wan"].values

groups = train_df["community_id"].fillna("unknown").astype(str).values

# GroupKFold first fold as unified validation set
gkf = GroupKFold(n_splits=3)
train_idx, val_idx = next(gkf.split(X_train, y_train, groups))
X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
y_tr, y_val = y_train[train_idx], y_train[val_idx]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ],
    sparse_threshold=0.0,
)
preprocessor.fit(X_train)

x_tr_proc = preprocessor.transform(X_tr)
x_val_proc = preprocessor.transform(X_val)
x_test_proc = preprocessor.transform(X_test)

# ------------------------------------------------------------------
# Train models and record convergence histories
# Use FULL training set (X_train) to match paper Table 4-2; GroupKFold
# val fold is only for convergence monitoring via eval_set.
# ------------------------------------------------------------------
RANDOM_STATE = 42
convergence_histories: dict[str, dict] = {}

# Preprocess full train/val/test
preprocessor.fit(X_train)
x_train_proc = preprocessor.transform(X_train)
x_val_proc = preprocessor.transform(X_val)
x_test_proc = preprocessor.transform(X_test)

# XGBoost
xgb_model = XGBRegressor(
    n_estimators=220, max_depth=4, learning_rate=0.045,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    objective="reg:squarederror", eval_metric="mae",
    random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_model.fit(x_train_proc, y_train, eval_set=[(x_val_proc, y_val), (x_test_proc, y_test)], verbose=False)
xgb_evals = xgb_model.evals_result()
xgb_val_mae = xgb_evals["validation_0"]["mae"][-1]
xgb_test_mae = xgb_evals["validation_1"]["mae"][-1]
convergence_histories["XGBoost"] = {
    "val_history": xgb_evals["validation_0"]["mae"],
    "test_history": xgb_evals["validation_1"]["mae"],
    "val_final_mae": float(xgb_val_mae),
    "test_final_mae": float(xgb_test_mae),
}

# LightGBM
lgb_model = LGBMRegressor(
    n_estimators=260, max_depth=-1, learning_rate=0.045, num_leaves=31,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=0.1,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
)
lgb_model.fit(x_train_proc, y_train, eval_set=[(x_val_proc, y_val), (x_test_proc, y_test)], eval_metric="mae")
lgb_evals = lgb_model.evals_result_
lgb_val_mae = lgb_evals["valid_0"]["l1"][-1]
lgb_test_mae = lgb_evals["valid_1"]["l1"][-1]
convergence_histories["LightGBM"] = {
    "val_history": lgb_evals["valid_0"]["l1"],
    "test_history": lgb_evals["valid_1"]["l1"],
    "val_final_mae": float(lgb_val_mae),
    "test_final_mae": float(lgb_test_mae),
}

# CatBoost — loss_function='RMSE' matches paper; eval_metric='MAE' for MAE curve
cat_model = CatBoostRegressor(
    iterations=260, depth=6, learning_rate=0.045, l2_leaf_reg=3.0,
    loss_function="RMSE", eval_metric="MAE",
    random_seed=RANDOM_STATE, verbose=False,
    allow_writing_files=False,
)
cat_model.fit(x_train_proc, y_train, eval_set=[(x_val_proc, y_val), (x_test_proc, y_test)], verbose=False)
cat_evals = cat_model.get_evals_result()
cat_val_mae = cat_evals["validation_0"]["MAE"][-1]
cat_test_mae = cat_evals["validation_1"]["MAE"][-1]
convergence_histories["CatBoost"] = {
    "val_history": cat_evals["validation_0"]["MAE"],
    "test_history": cat_evals["validation_1"]["MAE"],
    "val_final_mae": float(cat_val_mae),
    "test_final_mae": float(cat_test_mae),
}

# HistGradientBoosting (manual eval history every 10 iterations)
hgb_val_history = []
hgb_test_history = []
for n_iter in range(1, 241, 10):
    hgb_temp = HistGradientBoostingRegressor(
        max_iter=n_iter, learning_rate=0.055, l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )
    hgb_temp.fit(x_train_proc, y_train)
    val_pred = hgb_temp.predict(x_val_proc)
    test_pred = hgb_temp.predict(x_test_proc)
    hgb_val_history.append(float(np.mean(np.abs(y_val - val_pred))))
    hgb_test_history.append(float(np.mean(np.abs(y_test - test_pred))))
hgb_model = HistGradientBoostingRegressor(
    max_iter=240, learning_rate=0.055, l2_regularization=0.05, random_state=RANDOM_STATE,
)
hgb_model.fit(x_train_proc, y_train)
convergence_histories["HistGradientBoosting"] = {
    "val_history": hgb_val_history,
    "test_history": hgb_test_history,
    "iterations": list(range(1, 241, 10)),
    "val_final_mae": float(hgb_val_history[-1]),
    "test_final_mae": float(hgb_test_history[-1]),
}

print("Model training completed. Convergence histories recorded.")

# ------------------------------------------------------------------
# Train full StackingRidge and extract coefficients
# ------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=180, max_depth=14, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
hgb = HistGradientBoostingRegressor(max_iter=240, learning_rate=0.055, l2_regularization=0.05, random_state=RANDOM_STATE)
xgb = XGBRegressor(n_estimators=220, max_depth=4, learning_rate=0.045, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1)
lgb = LGBMRegressor(n_estimators=260, max_depth=-1, learning_rate=0.045, num_leaves=31, subsample=0.85, colsample_bytree=0.85, reg_lambda=0.1, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
cat = CatBoostRegressor(iterations=260, depth=6, learning_rate=0.045, l2_leaf_reg=3.0, loss_function="MAE", random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)

stacking_estimators = [
    ("RandomForest", rf), ("GradientBoosting", gb), ("HistGradientBoosting", hgb),
    ("XGBoost", xgb), ("LightGBM", lgb), ("CatBoost", cat),
]
stacking = StackingRegressor(
    estimators=stacking_estimators,
    final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
    cv=3, n_jobs=-1,
)
stacking_pipe = Pipeline([("preprocessor", preprocessor), ("regressor", stacking)])
stacking_pipe.fit(X_train, y_train)

final_estimator = stacking_pipe.named_steps["regressor"].final_estimator_
stacking_coefs = final_estimator.coef_
stacking_names = [name for name, _ in stacking_estimators]
stacking_coef_dict = dict(zip(stacking_names, stacking_coefs))
print("Stacking coefficients:", stacking_coef_dict)

# ------------------------------------------------------------------
# Helper to save figures
# ------------------------------------------------------------------
def save_figure(fig: plt.Figure, basename: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{basename}.svg", format="svg", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{basename}.png", format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------
# Fig 1: Correlation Heatmap
# ------------------------------------------------------------------
def plot_fig1():
    basename = "fig1_correlation_heatmap"
    descriptions[basename] = "Correlation Heatmap of Core Numerical Features and Total Price (10⁴ YUAN)"

    numeric_cols = NUMERIC_FEATURES + ["total_price_wan"]
    corr_df = df[numeric_cols].corr()
    core_features = [
        "total_price_wan", "area_sqm", "poi_subway_count", "accessibility_index",
        "poi_balance_score", "region_accessibility_interaction", "house_certificate_years",
        "poi_bank_count", "total_floors", "poi_total_count", "bath_count", "area_per_room",
    ]
    selected = [c for c in core_features if c in corr_df.columns]
    sub_corr = corr_df.loc[selected, selected]
    # Rename total_price_wan to total_price for cleaner display
    sub_corr = sub_corr.rename(index={"total_price_wan": "total_price"}, columns={"total_price_wan": "total_price"})

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        sub_corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        annot=True, fmt=".2f", annot_kws={"size": 9}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    save_figure(fig, basename)
    print(f"{basename} generated")

# ------------------------------------------------------------------
# Fig 2: Convergence Curves
# ------------------------------------------------------------------
def plot_fig2():
    basename = "fig2_convergence_curves"
    test_mae_info = " | ".join(
        f"{name} Test MAE={hist['test_final_mae']:.2f}"
        for name, hist in convergence_histories.items()
    )
    descriptions[basename] = (
        "Training Convergence Curves: Holdout Test MAE (solid) vs Validation MAE (dashed) "
        f"for Gradient Boosting Models (10⁴ YUAN). Final holdout test MAE: {test_mae_info}"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"XGBoost": "#1f77b4", "LightGBM": "#ff7f0e", "CatBoost": "#2ca02c", "HistGradientBoosting": "#d62728"}

    for model_name, hist in convergence_histories.items():
        color = colors[model_name]
        if model_name == "XGBoost":
            val_vals = hist["val_history"]
            test_vals = hist["test_history"]
            iterations = list(range(1, len(val_vals) + 1))
            ax.plot(iterations, val_vals, linestyle="--", alpha=0.45, linewidth=1.2,
                    color=color)
            ax.plot(iterations, test_vals, linestyle="-", alpha=1.0, linewidth=2.0,
                    color=color, label=f"{model_name} (Test MAE={hist['test_final_mae']:.2f})")
        elif model_name == "LightGBM":
            val_vals = hist["val_history"]
            test_vals = hist["test_history"]
            iterations = list(range(1, len(val_vals) + 1))
            ax.plot(iterations, val_vals, linestyle="--", alpha=0.45, linewidth=1.2,
                    color=color)
            ax.plot(iterations, test_vals, linestyle="-", alpha=1.0, linewidth=2.0,
                    color=color, label=f"{model_name} (Test MAE={hist['test_final_mae']:.2f})")
        elif model_name == "CatBoost":
            val_vals = hist["val_history"]
            test_vals = hist["test_history"]
            iterations = list(range(1, len(val_vals) + 1))
            ax.plot(iterations, val_vals, linestyle="--", alpha=0.45, linewidth=1.2,
                    color=color)
            ax.plot(iterations, test_vals, linestyle="-", alpha=1.0, linewidth=2.0,
                    color=color, label=f"{model_name} (Test MAE={hist['test_final_mae']:.2f})")
        elif model_name == "HistGradientBoosting":
            iters = hist["iterations"]
            ax.plot(iters, hist["val_history"], linestyle="--", alpha=0.45, linewidth=1.2,
                    color=color, marker="s", markersize=3)
            ax.plot(iters, hist["test_history"], linestyle="-", alpha=1.0, linewidth=2.0,
                    color=color, marker="s", markersize=4,
                    label=f"{model_name} (Test MAE={hist['test_final_mae']:.2f})")

    # Unified legend entry for validation lines
    ax.plot([], [], linestyle="--", alpha=0.45, linewidth=1.2, color="gray", label="Validation MAE")

    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("MAE (10⁴ YUAN)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, basename)
    print(f"{basename} generated")

# ------------------------------------------------------------------
# Fig 3: Stacking Coefficients
# ------------------------------------------------------------------
def plot_fig3():
    basename = "fig3_stacking_coefficients"
    descriptions[basename] = "Stacking Ridge Meta-Learner Coefficients for Base Learners"

    coefs = pd.DataFrame({
        "BaseLearner": stacking_names,
        "Coefficient": stacking_coefs,
    }).sort_values(by="Coefficient", key=lambda x: x.abs(), ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in coefs["Coefficient"]]
    bars = ax.barh(coefs["BaseLearner"][::-1], coefs["Coefficient"][::-1],
                   color=colors_bar[::-1], edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Ridge Meta-Learner Coefficient", fontsize=12)
    for bar, val in zip(bars, coefs["Coefficient"][::-1]):
        ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left" if val >= 0 else "right", va="center", fontsize=10)
    ax.set_xlim(min(coefs["Coefficient"]) - 0.1, max(coefs["Coefficient"]) + 0.1)
    plt.tight_layout()
    save_figure(fig, basename)
    print(f"{basename} generated")

# ------------------------------------------------------------------
# Fig 4: Semantic Positive Rate Comparison (unified 29-label system,
# category-banded; LLM annotates 28 labels — bus_hub is rule-only)
# ------------------------------------------------------------------
FIG4_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Landscape & Environment", ["river_view", "park_nearby"]),
    ("Housing Quality", [
        "luxury_decor", "new_community", "old_community", "quality_community",
        "simple_decor", "rough", "low_density", "board_building",
    ]),
    ("Transaction Attributes", [
        "urgent_sale", "price_negotiable", "below_market", "sincere", "direct_seller",
    ]),
    ("Facility Configuration", [
        "subway", "bus_hub", "school_district", "school_nearby", "shopping", "elevator",
    ]),
    ("Layout Characteristics", [
        "good_lighting", "garden", "garage", "duplex", "furnished", "double_bath",
    ]),
    ("Floor Preferences", ["high_floor_view", "good_floor"]),
]
FIG4_RULE_ONLY = {"bus_hub"}


def plot_fig4():
    basename = "fig4_semantic_positive_rate"
    descriptions[basename] = ("Positive Sample Rate by Annotator Across the 29-Label "
                              "Semantic System (LLM annotates 28 labels; bus_hub is rule-only)")

    df = pd.read_csv(SEMANTIC_DIR / "semantic_feature_comparison.csv").set_index("feature_name")

    # y 坐标：组间留 1.2 的间隔放组标题（自下而上堆，第一组最终在最上方）
    GAP = 1.2
    y_of: dict[str, float] = {}
    group_head: list[tuple[str, float]] = []
    group_span: list[tuple[float, float]] = []
    order: list[str] = []
    y = 0.0
    for cat, labels in reversed(FIG4_CATEGORIES):
        sub = sorted(labels, key=lambda k: df.loc[k, "rule_positive_rate"], reverse=True)
        y0 = y
        for k in sub:
            y_of[k] = y
            order.append(k)
            y += 1.0
        group_span.append((y0 - 0.55, y - 0.45))
        group_head.append((cat, y - 0.45 + GAP * 0.5))
        y += GAP

    assert len(order) == 29, f"expect 29 labels, got {len(order)}"
    rule = np.array([df.loc[k, "rule_positive_rate"] * 100 for k in order])
    llm = np.array([
        np.nan if k in FIG4_RULE_ONLY else df.loc[k, "llm_positive_rate"] * 100
        for k in order
    ])

    fig, ax = plt.subplots(figsize=(12, 12))
    bar_h = 0.38
    rule_y = np.array([y_of[k] for k in order])

    ax.barh(rule_y + bar_h / 2 + 0.01, rule, bar_h, label="Rule-based",
            color="#4c78a8", edgecolor="black", linewidth=0.3)
    ax.barh(rule_y - bar_h / 2 - 0.01, np.nan_to_num(llm), bar_h, label="LLM-annotated",
            color="#f58518", edgecolor="black", linewidth=0.3)

    for k in FIG4_RULE_ONLY:
        ax.text(0.4, y_of[k] - bar_h / 2, "n/a (rule only)", va="center",
                fontsize=8.5, color="#888888", style="italic")

    for gi, ((y_bot, y_top), (cat, hy)) in enumerate(zip(group_span, group_head)):
        if gi % 2 == 0:
            ax.axhspan(y_bot, y_top, color="#f2f2f2", alpha=0.7, zorder=0)
        ax.axhline(y_top, color="#bbbbbb", linewidth=0.8, zorder=1)
        ax.text(0.2, hy, cat, va="center", ha="left",
                fontsize=10, fontweight="bold", color="#444444")

    ax.set_yticks([y_of[k] for k in order])
    ax.set_yticklabels(order, fontsize=10)
    ax.set_xlabel("Positive Sample Rate (%)", fontsize=12)
    ax.set_xlim(0, 63)
    ax.set_ylim(-0.7, y - GAP + 0.3)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, basename)
    print(f"{basename} generated")

# ------------------------------------------------------------------
# Fig 5: Permutation Importance (a) + SHAP Beeswarm (b)
# ------------------------------------------------------------------
def _translate_feature_name(name: str) -> str:
    ori_map = {
        "南北": "N-S", "南": "S", "东西": "E-W", "西北": "NW", "西南": "SW",
        "西": "W", "东南": "SE", "北": "N", "东北": "NE", "东": "E",
    }
    cert_map = {
        "满二": "Over2Yrs", "满五年": "Over5Yrs", "满五": "Over5Yrs",
        "满二年": "Over2Yrs", "满2年": "Over2Yrs", "满20": "Over20Yrs",
        "满24": "Over24Yrs",
    }
    decor_map = {
        "毛坯": "Rough", "豪华装修": "Luxury", "简单装修": "Simple",
        "精装修": "FineDecor",
    }
    floor_map = {"中层": "Mid", "高层": "High", "低层": "Low"}
    bool_map = {"True": "Yes", "False": "No"}
    for cn, en in ori_map.items():
        name = name.replace(cn, en)
    for cn, en in cert_map.items():
        name = name.replace(cn, en)
    for cn, en in decor_map.items():
        name = name.replace(cn, en)
    for cn, en in floor_map.items():
        name = name.replace(cn, en)
    for cn, en in bool_map.items():
        name = name.replace(cn, en)
    name = name.replace("unknown", "Unknown")

    def repl_layout(m):
        return f"{m.group(1)}R{m.group(2)}H{m.group(3)}B"
    name = re.sub(r"(\d+)室(\d+)厅(\d+)卫", repl_layout, name)
    name = name.replace("numeric__", "").replace("categorical__", "")
    return name


def plot_fig5():
    basename = "fig5_feature_importance"
    descriptions[basename] = (
        "Feature Importance Visualization: (a) Top 15 Permutation Importance of Stacking Ridge "
        "(MAE Change, 10⁴ YUAN); (b) SHAP Beeswarm of HistGradientBoosting Base Learner (300 Test Samples)"
    )

    fig = plt.figure(figsize=(20, 10))

    # --- Subplot (a): Permutation Importance ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("(a)", loc="left", fontsize=12, fontweight="bold")

    fi = pd.read_csv(DATA_DIR / "feature_importance.csv")
    fi = fi.sort_values(by="importance_mean", key=lambda x: x.abs(), ascending=False).head(15)
    colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in fi["importance_mean"]]
    bars = ax1.barh(fi["feature"][::-1], fi["importance_mean"][::-1],
                    color=colors_bar[::-1], edgecolor="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.8)
    ax1.set_xlabel("Permutation Importance (MAE Change, 10⁴ YUAN)", fontsize=12)
    for bar, val in zip(bars, fi["importance_mean"][::-1]):
        ax1.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", ha="left" if val >= 0 else "right", va="center", fontsize=9)
    ax1.set_xlim(min(fi["importance_mean"]) - 0.3, max(fi["importance_mean"]) + 0.3)
    ax1.grid(True, axis="x", alpha=0.3)

    # --- Subplot (b): SHAP Beeswarm ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("(b)", loc="left", fontsize=12, fontweight="bold")

    # Train HistGB on full training set
    hgb_model = HistGradientBoostingRegressor(
        max_iter=240, learning_rate=0.055, l2_regularization=0.05, random_state=RANDOM_STATE
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("regressor", hgb_model)])
    pipe.fit(X_train, y_train)

    feature_names = list(preprocessor.get_feature_names_out())
    feature_names = [_translate_feature_name(n) for n in feature_names]

    x_test_proc = preprocessor.transform(X_test)
    sample_size = min(300, len(x_test_proc))
    x_sample = x_test_proc[:sample_size]

    explainer = shap.TreeExplainer(hgb_model)
    shap_values = explainer.shap_values(x_sample)

    plt.sca(ax2)
    shap.summary_plot(
        shap_values, x_sample, feature_names=feature_names,
        show=False, plot_size=None, max_display=15,
    )

    plt.tight_layout()
    save_figure(fig, basename)
    print(f"{basename} generated")

# ------------------------------------------------------------------
# Fig 5a: Standalone Permutation Importance (same style as fig5 subplot a)
# ------------------------------------------------------------------
def plot_fig5a():
    basename = "fig5a_permutation_importance"
    descriptions[basename] = "Top 15 Permutation Importance of Stacking Ridge (MAE Change, 10⁴ YUAN)"

    fi = pd.read_csv(DATA_DIR / "feature_importance.csv")
    fi = fi.sort_values(by="importance_mean", key=lambda x: x.abs(), ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in fi["importance_mean"]]
    bars = ax.barh(fi["feature"][::-1], fi["importance_mean"][::-1],
                   color=colors_bar[::-1], edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Permutation Importance (MAE Change, 10⁴ YUAN)", fontsize=12)
    for bar, val in zip(bars, fi["importance_mean"][::-1]):
        ax.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left" if val >= 0 else "right", va="center", fontsize=9)
    ax.set_xlim(min(fi["importance_mean"]) - 0.3, max(fi["importance_mean"]) + 0.3)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, basename)
    print(f"{basename} generated")

# ------------------------------------------------------------------
# Fig 6: Stratified Errors
# ------------------------------------------------------------------
def plot_fig6():
    basename = "fig6_stratified_errors"
    descriptions[basename] = "Stratified Error Visualization: (a) Regional MAE, (b) Area-Stratified MAE Trend, (c) Price Segment MAPE Distribution"

    err_df = pd.read_csv(DATA_DIR / "error_stratification.csv")
    fig = plt.figure(figsize=(16, 5))

    # (a) Regional MAE
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("(a)", loc="left", fontsize=12, fontweight="bold")
    region_df = err_df[err_df["stratify_column"] == "region_slug"].sort_values(by="mae", ascending=False)
    colors = ["#d62728" if m > 20 else "#2ca02c" if m < 15 else "#ff7f0e" for m in region_df["mae"]]
    ax1.bar(region_df["stratify_value"], region_df["mae"], color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("MAE (10⁴ YUAN)", fontsize=11)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, axis="y", alpha=0.3)
    for i, (_, row) in enumerate(region_df.iterrows()):
        ax1.text(i, row["mae"] + 0.3, f"{row['mae']:.1f}", ha="center", va="bottom", fontsize=9)

    # (b) Area-stratified MAE
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("(b)", loc="left", fontsize=12, fontweight="bold")
    area_df = err_df[err_df["stratify_column"] == "area_bin"].copy()
    area_df["sort_key"] = area_df["stratify_value"].apply(lambda x: float(x.strip("()[]").split(",")[0]))
    area_df = area_df.sort_values(by="sort_key")
    ax2.plot(area_df["stratify_value"], area_df["mae"], marker="o", color="#1f77b4", linewidth=2, markersize=8)
    ax2.set_ylabel("MAE (10⁴ YUAN)", fontsize=11)
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(True, alpha=0.3)
    for _, row in area_df.iterrows():
        ax2.text(row["stratify_value"], row["mae"] + 0.5, f"{row['mae']:.1f}", ha="center", va="bottom", fontsize=9)

    # (c) Price segment MAPE
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("(c)", loc="left", fontsize=12, fontweight="bold")
    price_df = err_df[err_df["stratify_column"] == "price_bin"].copy()
    price_df["sort_key"] = price_df["stratify_value"].apply(lambda x: float(x.strip("()[]").split(",")[0]))
    price_df = price_df.sort_values(by="sort_key")
    colors3 = ["#d62728" if m > 0.4 else "#2ca02c" if m < 0.15 else "#ff7f0e" for m in price_df["mape"]]
    ax3.bar(price_df["stratify_value"], price_df["mape"] * 100, color=colors3, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("MAPE (%)", fontsize=11)
    ax3.tick_params(axis="x", rotation=15)
    ax3.grid(True, axis="y", alpha=0.3)
    for _, row in price_df.iterrows():
        ax3.text(row["stratify_value"], row["mape"] * 100 + 1, f"{row['mape'] * 100:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_figure(fig, basename)
    print(f"{basename} generated")

# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig5a()
    plot_fig6()

    # Write descriptions
    with open(DESC_PATH, "w", encoding="utf-8") as f:
        for name, desc in descriptions.items():
            f.write(f"{name}: {desc}\n")

    print(f"\nAll figures regenerated in {OUTPUT_DIR}")
    print(f"Descriptions saved to {DESC_PATH}")
