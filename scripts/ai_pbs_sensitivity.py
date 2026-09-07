"""AI 权重与 PBS 构造的敏感性分析实验.

评审质疑: accessibility_index 的固定专家权重与 poi_balance_score 的熵构造缺乏依据.
本脚本在不修改现有代码的前提下, 复用 src/research/pipeline.py 的建模样板
(_split_by_time / _apply_reference_group_medians / _build_pipeline_for_features /
_cross_validate_model / _build_model_zoo), 对 AI 权重变体与 PBS 构造变体做对照实验.

协议与主实验完全同构: 时间切分后调用 _apply_reference_group_medians
(area_layout_efficiency 用训练集户型中位数, region_accessibility_interaction
用全市社区 POI 底表的区域基准), 因此 expert/entropy 变体可精确复现主实验
(Table 13 的 hist_gradient_boosting 与 stacking_ridge 数值), 作为内置校验锚点.

实验分组:
  A 组 (AI 权重变体, PBS 固定为现行熵版):
    expert / equal / data_driven / no_ai
  B 组 (PBS 构造变体, AI 固定为 expert):
    entropy / gini / herfindahl / no_pbs
  C 组 (确认性): A 组最好/最差变体 x stacking_ridge

注意: NUMERIC_FEATURES 中的 region_accessibility_interaction 依赖 accessibility_index
(accessibility_index / 按 region_slug 分组的中位数), AI 变体需级联重算该列;
no_ai 消融时两列一并从特征列表移除.

完整数据集（本仓库未随附，见 data/full/README.md）放入 data/full/ 后运行；
实验输出写入 data/runs/（gitignore）。

用法:
    uv run python scripts/ai_pbs_sensitivity.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.pipeline import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    POI_TYPES,
    TARGET_COLUMN,
    _apply_reference_group_medians,
    _build_features,
    _build_model_zoo,
    _build_pipeline_for_features,
    _clean_and_join,
    _compute_region_ai_medians,
    _cross_validate_model,
    _evaluate_predictions,
    _load_communities,
    _load_houses,
    _split_by_time,
)

HOUSES_PATH = PROJECT_ROOT / "data" / "full" / "nanchang_houses.jsonl"
COMMUNITIES_PATH = PROJECT_ROOT / "data" / "full" / "nanchang_communities.jsonl"
REGION_MEDIANS_PATH = PROJECT_ROOT / "data" / "reference" / "region_ai_medians.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3

POI_COLUMNS = [f"poi_{poi_type}_count" for poi_type in POI_TYPES]

EXPERT_WEIGHTS: dict[str, float] = {
    "poi_bank_count": 0.6,
    "poi_bus_count": 1.0,
    "poi_subway_count": 1.5,
    "poi_school_count": 1.4,
    "poi_restaurant_count": 0.7,
    "poi_shop_count": 0.8,
    "poi_hospital_count": 1.2,
}

# 依赖 accessibility_index 的派生列, AI 变体需级联重算
AI_DEPENDENT_FEATURES = ("region_accessibility_interaction",)


def build_accessibility_index(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """与 pipeline._build_accessibility_index 同构, 权重可配."""
    score = pd.Series(np.zeros(len(df)), index=df.index)
    for column_name, weight in weights.items():
        score = score + np.log1p(pd.to_numeric(df[column_name], errors="coerce").fillna(0)) * weight
    return score


def load_region_ai_medians() -> dict[str, float]:
    """加载区域 AI 中位数基准（与主实验 _compute_region_ai_medians 同来源）。

    优先读取发布的聚合参考文件 data/reference/region_ai_medians.json；
    缺失时从按区域目录的社区采集文件重新计算（完整数据集场景）。
    """
    if REGION_MEDIANS_PATH.exists():
        return {
            k: float(v)
            for k, v in json.loads(REGION_MEDIANS_PATH.read_text(encoding="utf-8")).items()
        }
    return _compute_region_ai_medians(COMMUNITIES_PATH)


def compute_poi_shares(df: pd.DataFrame) -> pd.DataFrame:
    poi_matrix = df[POI_COLUMNS].astype(float)
    poi_total = poi_matrix.sum(axis=1).clip(lower=1)
    return poi_matrix.div(poi_total, axis=0)


def pbs_entropy(df: pd.DataFrame) -> pd.Series:
    """现行构造: 7 类 POI 占比的归一化信息熵."""
    share = compute_poi_shares(df)
    entropy = -(share * np.log(share.replace(0, np.nan))).sum(axis=1).fillna(0)
    return entropy / np.log(len(POI_TYPES))


def pbs_gini_balance(df: pd.DataFrame) -> pd.Series:
    """Gini 均衡度 = 1 - Gini(占比). Gini 越大约不均衡, 取 1-Gini 使其与熵同向(越大越均衡)."""
    share = compute_poi_shares(df).to_numpy()
    n = share.shape[1]
    sorted_share = np.sort(share, axis=1)
    index = np.arange(1, n + 1)
    # Gini = (2 * sum(i * x_(i)) / (n * sum(x))) - (n + 1) / n
    gini = (2.0 * (sorted_share * index).sum(axis=1) / (n * share.sum(axis=1).clip(min=1e-12))) - (n + 1.0) / n
    return pd.Series(1.0 - gini, index=df.index)


def pbs_herfindahl_balance(df: pd.DataFrame) -> pd.Series:
    """Herfindahl 均衡度: (1 - HHI) / (1 - 1/n), 归一化到 [0, 1], 越大越均衡."""
    share = compute_poi_shares(df)
    n = len(POI_TYPES)
    hhi = (share**2).sum(axis=1)
    return (1.0 - hhi) / (1.0 - 1.0 / n)


def fit_data_driven_weights(train_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """仅用训练集: log1p(poi 计数) -> 标准化 -> Ridge -> log1p(total_price_wan).

    取标准化系数, 负值截 0, 归一化到均值=1 (与现行权重同量纲).
    返回 (归一化权重, 原始标准化系数).
    """
    x = np.log1p(train_df[POI_COLUMNS].astype(float).fillna(0))
    y = np.log1p(train_df[TARGET_COLUMN].astype(float))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(x_scaled, y)
    raw_coefs = dict(zip(POI_COLUMNS, ridge.coef_, strict=True))
    clipped = np.clip(ridge.coef_, 0.0, None)
    if clipped.sum() <= 0:
        # 全部系数非正, 退化为等权
        weights = {column: 1.0 for column in POI_COLUMNS}
    else:
        normalized = clipped / clipped.mean()
        weights = dict(zip(POI_COLUMNS, normalized.tolist(), strict=True))
    return weights, raw_coefs


def run_variant(
    df: pd.DataFrame,
    model_name: str,
    numeric_features: tuple[str, ...],
    region_ai_medians: dict[str, float],
) -> dict[str, object]:
    """按协议训练评估: 时间 holdout + 无泄漏参考中位数 + community_id 分组 3 折 CV."""
    train_df, test_df = _split_by_time(df, TEST_SIZE)
    train_df, test_df = _apply_reference_group_medians(train_df, test_df, region_ai_medians)
    x_train = train_df[list(numeric_features) + list(CATEGORICAL_FEATURES)]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[list(numeric_features) + list(CATEGORICAL_FEATURES)]
    y_test = test_df[TARGET_COLUMN]
    groups = train_df["community_id"].fillna("unknown").astype(str)

    regressor = _build_model_zoo(RANDOM_STATE)[model_name]
    pipeline = _build_pipeline_for_features(regressor, numeric_features, CATEGORICAL_FEATURES)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        cv_mae_mean, cv_mae_std, _ = _cross_validate_model(
            pipeline, x_train, y_train, groups, CV_FOLDS
        )
        pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metric = _evaluate_predictions(model_name, y_test, predictions, cv_mae_mean, cv_mae_std)
    return {
        "mae": metric.mae,
        "rmse": metric.rmse,
        "r2": metric.r2,
        "cv_mae_mean": metric.cv_mae_mean,
        "cv_mae_std": metric.cv_mae_std,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }


def main() -> None:
    started_at = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "runs" / f"ai_pbs_sensitivity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从原始采集文件重建特征（与主实验同一代码路径）。
    # 不得读取 cleaned_features.csv 快照: CSV 往返的浮点噪声会被提升树
    # 分箱/分裂混沌放大 (实测 MAE 21.9899 -> 22.2192), 且快照不含
    # _apply_reference_group_medians 的无泄漏重算。
    houses_df = _load_houses(HOUSES_PATH)
    communities_df = _load_communities(COMMUNITIES_PATH)
    merged_df, _ = _clean_and_join(houses_df, communities_df)
    base_df = _build_features(merged_df)
    print(f"rebuilt {len(base_df)} feature rows from raw jsonl")

    region_ai_medians = load_region_ai_medians()
    print(f"region AI medians loaded for {len(region_ai_medians)} regions")

    results: list[dict[str, object]] = []

    # ---- A 组: AI 权重变体 (PBS 固定现行熵版) ----
    # 先用现行特征划分, 仅在训练集上拟合数据驱动权重
    base_train_df, _ = _split_by_time(base_df, TEST_SIZE)
    data_driven_weights, data_driven_raw_coefs = fit_data_driven_weights(base_train_df)
    weight_report = {
        "expert_weights": EXPERT_WEIGHTS,
        "data_driven_weights_normalized_mean1": data_driven_weights,
        "data_driven_ridge_standardized_coefs_raw": data_driven_raw_coefs,
        "note": "data_driven: 仅训练集, Ridge(alpha=1.0) 拟合 log1p(poi)->log1p(total_price_wan), "
        "标准化系数负值截 0 后归一化到均值=1",
    }
    (output_dir / "ai_weights.json").write_text(
        json.dumps(weight_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("data-driven weights:", json.dumps(data_driven_weights, indent=2))

    ai_weight_variants: dict[str, dict[str, float] | None] = {
        "expert": EXPERT_WEIGHTS,
        "equal": {column: 1.0 for column in POI_COLUMNS},
        "data_driven": data_driven_weights,
        "no_ai": None,
    }
    for variant_name, weights in ai_weight_variants.items():
        variant_df = base_df.copy()
        if weights is None:
            numeric_features = tuple(
                f for f in NUMERIC_FEATURES
                if f != "accessibility_index" and f not in AI_DEPENDENT_FEATURES
            )
        else:
            variant_df["accessibility_index"] = build_accessibility_index(variant_df, weights)
            # region_accessibility_interaction 由 run_variant 内的
            # _apply_reference_group_medians 以无泄漏区域基准级联重算
            numeric_features = NUMERIC_FEATURES
        metrics = run_variant(variant_df, "hist_gradient_boosting", numeric_features, region_ai_medians)
        results.append({"group": "A_ai_weights", "variant": variant_name, "model": "hist_gradient_boosting", **metrics})
        print(f"[A/{variant_name}] mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} "
              f"r2={metrics['r2']:.4f} cv_mae={metrics['cv_mae_mean']:.4f}")

    # ---- B 组: PBS 构造变体 (AI 固定 expert, 列值沿用数据表现行值) ----
    pbs_variants = {
        "entropy": pbs_entropy,
        "gini": pbs_gini_balance,
        "herfindahl": pbs_herfindahl_balance,
        "no_pbs": None,
    }
    for variant_name, builder in pbs_variants.items():
        variant_df = base_df.copy()
        if builder is None:
            numeric_features = tuple(f for f in NUMERIC_FEATURES if f != "poi_balance_score")
        else:
            variant_df["poi_balance_score"] = builder(variant_df)
            numeric_features = NUMERIC_FEATURES
        metrics = run_variant(variant_df, "hist_gradient_boosting", numeric_features, region_ai_medians)
        results.append({"group": "B_pbs_construct", "variant": variant_name, "model": "hist_gradient_boosting", **metrics})
        print(f"[B/{variant_name}] mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} "
              f"r2={metrics['r2']:.4f} cv_mae={metrics['cv_mae_mean']:.4f}")

    # ---- C 组: A 组最好/最差变体 x stacking_ridge ----
    a_results = [r for r in results if r["group"] == "A_ai_weights"]
    best_a = min(a_results, key=lambda r: r["mae"])["variant"]
    worst_a = max(a_results, key=lambda r: r["mae"])["variant"]
    for variant_name in dict.fromkeys([best_a, worst_a]):
        weights = ai_weight_variants[variant_name]
        variant_df = base_df.copy()
        if weights is None:
            numeric_features = tuple(
                f for f in NUMERIC_FEATURES
                if f != "accessibility_index" and f not in AI_DEPENDENT_FEATURES
            )
        else:
            variant_df["accessibility_index"] = build_accessibility_index(variant_df, weights)
            numeric_features = NUMERIC_FEATURES
        metrics = run_variant(variant_df, "stacking_ridge", numeric_features, region_ai_medians)
        tag = "best" if variant_name == best_a else "worst"
        results.append({
            "group": "C_confirm",
            "variant": f"{variant_name}({tag}_of_A)",
            "model": "stacking_ridge",
            **metrics,
        })
        print(f"[C/{variant_name}-{tag}] mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} "
              f"r2={metrics['r2']:.4f} cv_mae={metrics['cv_mae_mean']:.4f}")

    results_df = pd.DataFrame(results)
    results_path = output_dir / "sensitivity_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    # ---- 内置校验锚点: expert/entropy 变体协议与主实验同构, 应复现 Table 13 ----
    anchor = results_df[
        (results_df["group"] == "A_ai_weights") & (results_df["variant"] == "expert")
    ].iloc[0]
    anchor_ok = abs(anchor["mae"] - 21.9899) < 1e-3 and abs(anchor["r2"] - 0.2192) < 1e-3
    print(f"[anchor] A/expert mae={anchor['mae']:.4f} r2={anchor['r2']:.4f} "
          f"vs main-run 21.9899/0.2192 -> {'MATCH' if anchor_ok else 'MISMATCH'}")
    c_expert = results_df[
        (results_df["group"] == "C_confirm") & results_df["variant"].str.startswith("expert")
    ]
    stacking_anchor_ok = None
    if not c_expert.empty:
        stacking_anchor_ok = abs(c_expert.iloc[0]["mae"] - 20.4936) < 1e-3
        print(f"[anchor] C/expert stacking mae={c_expert.iloc[0]['mae']:.4f} "
              f"vs main-run 20.4936 -> {'MATCH' if stacking_anchor_ok else 'MISMATCH'}")

    elapsed = time.perf_counter() - started_at
    summary = {
        "data_path": f"{HOUSES_PATH} + {COMMUNITIES_PATH} (raw rebuild)",
        "rows": int(len(base_df)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "default_model": "hist_gradient_boosting",
        "confirm_model": "stacking_ridge",
        "protocol": "temporal holdout + _apply_reference_group_medians "
        "(train-only layout medians, community-registry region AI medians) "
        "+ community_id grouped 3-fold CV; identical to the main run",
        "region_ai_medians_source": str(REGION_MEDIANS_PATH)
        if REGION_MEDIANS_PATH.exists() else str(COMMUNITIES_PATH),
        "anchor_check_a_expert_matches_main_run": bool(anchor_ok),
        "anchor_check_c_expert_stacking_matches_main_run": (
            bool(stacking_anchor_ok) if stacking_anchor_ok is not None else None
        ),
        "ai_dependent_features_recomputed": list(AI_DEPENDENT_FEATURES),
        "elapsed_seconds": round(elapsed, 1),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nresults -> {results_path}")
    print(f"weights -> {output_dir / 'ai_weights.json'}")
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
