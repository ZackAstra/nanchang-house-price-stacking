"""
任务 2.2（务实替代）：高阶多项式交互特征探索

核心逻辑：
1. 复用 pipeline 的数据加载、清洗、特征工程
2. 对核心数值特征（area_sqm, poi_subway_count, accessibility_index, poi_balance_score）
   构造二阶交互项（degree=2, interaction_only=True）
3. 评估新增交互特征对 stacking_ridge 的效果
4. 对比实验：原始特征 vs 原始+多项式交互特征

替代理由：
- PySR 依赖 Julia，安装复杂
- PolynomialFeatures 已内置于 sklearn，零额外依赖
- 二阶交互是符号回归最常见的发现形式
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import PolynomialFeatures

from src.research.models import JsonValue, ModelMetric, ResearchRunConfig, ResearchResult
from src.research.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    _apply_sample_limit,
    _build_features,
    _build_model_zoo,
    _build_pipeline_for_features,
    _clean_and_join,
    _cross_validate_model,
    _evaluate_predictions,
    _load_communities,
    _load_houses,
    _split_by_time,
    _apply_reference_group_medians,
    _compute_region_ai_medians,
    _write_metrics,
)

CURRENT_BEST_FULL_MAE = 20.911922387801575

# 核心数值特征用于构造交互项
CORE_INTERACTION_FEATURES: tuple[str, ...] = (
    "area_sqm",
    "poi_subway_count",
    "accessibility_index",
    "poi_balance_score",
    "room_count",
    "floor_level_score",
)


def run_polynomial_feature_experiment(config: ResearchRunConfig) -> ResearchResult:
    """执行多项式交互特征实验。"""
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_run_log(output_dir, "run_started", {"run_id": config.run_id, "experiment": "polynomial_feature"})

    # 1. 数据加载
    houses_df = _load_houses(config.houses_path)
    communities_df = _load_communities(config.communities_path)
    merged_df, audit = _clean_and_join(houses_df, communities_df)
    feature_df = _build_features(merged_df)
    feature_df = _apply_sample_limit(feature_df, config.sample_limit, config.random_state)

    # 2. 构造多项式交互特征
    feature_df, poly_feature_names = _build_polynomial_features(feature_df)

    # 写入交互特征列表
    with (output_dir / "polynomial_feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(poly_feature_names, f, ensure_ascii=False, indent=2)

    # 3. 划分训练/测试集
    train_df, test_df = _split_by_time(feature_df, config.test_size)
    train_df, test_df = _apply_reference_group_medians(train_df, test_df, _compute_region_ai_medians(config.communities_path))

    # 4. 实验配置
    experiments = _build_experiment_configs(poly_feature_names)

    is_quick_mode = config.sample_limit is not None
    if is_quick_mode:
        experiments = [e for e in experiments if str(e["name"]) in {"original", "poly_enhanced"}]

    all_metrics: list[ModelMetric] = []
    summary_rows: list[dict[str, JsonValue]] = []

    for exp_config in experiments:
        exp_name = str(exp_config["name"])
        numeric_features = cast(tuple[str, ...], exp_config["numeric_features"])
        categorical_features = cast(tuple[str, ...], exp_config["categorical_features"])

        _write_run_log(
            output_dir,
            "experiment_started",
            {
                "experiment_name": exp_name,
                "numeric_feature_count": len(numeric_features),
                "categorical_feature_count": len(categorical_features),
                "poly_feature_count": len(poly_feature_names) if "poly" in exp_name else 0,
            },
        )

        metrics, predictions_by_model = _run_poly_variant(
            train_df=train_df,
            test_df=test_df,
            config=config,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            output_dir=output_dir,
            experiment_name=exp_name,
            quick_mode=is_quick_mode,
        )

        all_metrics.extend(metrics)

        best_metric = min(metrics, key=lambda m: m.mae)
        summary_rows.append(
            {
                "experiment_name": exp_name,
                "best_model_name": best_metric.model_name.split("__")[-1],
                "best_mae": best_metric.mae,
                "best_rmse": best_metric.rmse,
                "best_r2": best_metric.r2,
                "best_mape": best_metric.mape,
                "delta_mae_vs_baseline": best_metric.mae - CURRENT_BEST_FULL_MAE,
            }
        )

    # 5. 写入产物
    _write_metrics(output_dir, all_metrics)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "poly_experiment_summary.csv", index=False, encoding="utf-8-sig"
    )
    _write_stacking_comparison(output_dir, all_metrics)

    best_overall = min(all_metrics, key=lambda m: m.mae)
    _write_run_log(
        output_dir,
        "run_completed",
        {
            "best_experiment_model": best_overall.model_name,
            "best_mae": best_overall.mae,
            "baseline_mae": CURRENT_BEST_FULL_MAE,
            "delta_vs_baseline": best_overall.mae - CURRENT_BEST_FULL_MAE,
        },
    )

    return ResearchResult(
        run_id=config.run_id,
        output_dir=output_dir,
        audit=audit,
        metrics=all_metrics,
        best_model_name=best_overall.model_name,
    )


def _build_polynomial_features(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """构造二阶交互特征（仅交互项，不含平方项）。"""
    df = feature_df.copy()
    available_core_features = [f for f in CORE_INTERACTION_FEATURES if f in df.columns]

    if len(available_core_features) < 2:
        return df, []

    core_values = df[available_core_features].to_numpy(dtype=float)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_values = poly.fit_transform(core_values)
    poly_feature_names = poly.get_feature_names_out(available_core_features).tolist()

    # 只保留交互项（名称中包含空格的），排除原始特征
    interaction_indices = [
        i for i, name in enumerate(poly_feature_names)
        if " " in name and name not in available_core_features
    ]
    interaction_names = [
        f"poly__{poly_feature_names[i].replace(' ', '_x_')}"
        for i in interaction_indices
    ]

    for idx, col_name in zip(interaction_indices, interaction_names, strict=True):
        df[col_name] = poly_values[:, idx]

    return df, interaction_names


def _build_experiment_configs(poly_feature_names: list[str]) -> list[dict[str, JsonValue | tuple[str, ...]]]:
    """构建实验配置。"""
    original_numeric = NUMERIC_FEATURES
    poly_numeric = NUMERIC_FEATURES + tuple(poly_feature_names)

    return [
        {
            "name": "original",
            "numeric_features": original_numeric,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        {
            "name": "poly_enhanced",
            "numeric_features": poly_numeric,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        {
            "name": "poly_only",
            "numeric_features": tuple(poly_feature_names),
            "categorical_features": CATEGORICAL_FEATURES,
        },
    ]


def _run_poly_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ResearchRunConfig,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    output_dir: Path,
    experiment_name: str,
    quick_mode: bool = False,
) -> tuple[list[ModelMetric], dict[str, np.ndarray]]:
    """执行单个实验变体。"""
    x_train = train_df[list(numeric_features) + list(categorical_features)]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[list(numeric_features) + list(categorical_features)]
    y_test = test_df[TARGET_COLUMN]

    model_zoo = _build_model_zoo(config.random_state)
    if quick_mode:
        model_zoo = {k: v for k, v in model_zoo.items() if k == "stacking_ridge"}

    metrics: list[ModelMetric] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    groups = train_df["community_id"].fillna("unknown").astype(str)

    for model_name, regressor in model_zoo.items():
        prefixed_name = f"{experiment_name}__{model_name}"
        pipeline = _build_pipeline_for_features(regressor, numeric_features, categorical_features)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                started_at = time.perf_counter()
                cv_mae_mean, cv_mae_std, cv_mae_scores = _cross_validate_model(
                    pipeline, x_train, y_train, groups, config.cv_folds
                )
                pipeline.fit(x_train, y_train)
                train_seconds = time.perf_counter() - started_at

            predict_started_at = time.perf_counter()
            predictions = pipeline.predict(x_test)
            predict_seconds = time.perf_counter() - predict_started_at

            predictions = np.clip(predictions, a_min=1.0, a_max=None)

            metric = _evaluate_predictions(
                prefixed_name,
                y_test,
                predictions,
                cv_mae_mean,
                cv_mae_std,
            )
            metrics.append(metric)
            predictions_by_model[prefixed_name] = predictions

            _append_experiment_log(
                output_dir,
                {
                    "event": "model_completed",
                    "experiment": experiment_name,
                    "model_name": model_name,
                    "prefixed_name": prefixed_name,
                    "cv_mae_mean": cv_mae_mean,
                    "cv_mae_std": cv_mae_std,
                    "mae": metric.mae,
                    "rmse": metric.rmse,
                    "r2": metric.r2,
                    "mape": metric.mape,
                    "train_seconds": train_seconds,
                    "predict_seconds": predict_seconds,
                },
            )
        except Exception as exc:
            _append_experiment_log(
                output_dir,
                {
                    "event": "model_failed",
                    "experiment": experiment_name,
                    "model_name": model_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    return metrics, predictions_by_model


def _write_stacking_comparison(output_dir: Path, all_metrics: list[ModelMetric]) -> None:
    rows: list[dict[str, JsonValue]] = []
    for metric in all_metrics:
        if "__stacking_ridge" not in metric.model_name:
            continue
        parts = metric.model_name.split("__")
        experiment_name = parts[0]
        rows.append(
            {
                "experiment": experiment_name,
                "model_name": metric.model_name,
                "mae": metric.mae,
                "rmse": metric.rmse,
                "r2": metric.r2,
                "mape": metric.mape,
                "cv_mae_mean": metric.cv_mae_mean,
                "cv_mae_std": metric.cv_mae_std,
                "delta_vs_baseline": metric.mae - CURRENT_BEST_FULL_MAE,
            }
        )
    if rows:
        pd.DataFrame(rows).sort_values(by="mae").to_csv(
            output_dir / "stacking_comparison.csv", index=False, encoding="utf-8-sig"
        )


def _write_run_log(output_dir: Path, event: str, payload: dict[str, JsonValue]) -> None:
    log_record: dict[str, JsonValue] = {
        "event": event,
        "created_at": datetime.now().isoformat(),
        "payload": payload,
    }
    with (output_dir / "run_log.jsonl").open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(log_record, ensure_ascii=False) + "\n")


def _append_experiment_log(output_dir: Path, record: dict[str, JsonValue]) -> None:
    log_record: dict[str, JsonValue] = {
        "created_at": datetime.now().isoformat(),
        **record,
    }
    with (output_dir / "experiment_log.jsonl").open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(log_record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="多项式交互特征实验")
    parser.add_argument("--houses", required=True, help="房源JSONL路径")
    parser.add_argument("--communities", required=True, help="小区JSONL路径")
    parser.add_argument("--output-dir", required=True, help="研究产物根目录")
    parser.add_argument("--run-id", help="研究运行ID，不传则自动生成")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2, help="留出集比例")
    parser.add_argument("--cv-folds", type=int, default=3, help="交叉验证折数")
    parser.add_argument("--sample-limit", type=int, help="抽样上限，用于快速验证")
    args = parser.parse_args()

    run_id = args.run_id or f"poly_feature_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = ResearchRunConfig(
        houses_path=Path(args.houses),
        communities_path=Path(args.communities),
        output_root=Path(args.output_dir),
        run_id=run_id,
        random_state=args.random_state,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        sample_limit=args.sample_limit,
    )
    result = run_polynomial_feature_experiment(config)
    print("=" * 60)
    print("Polynomial Feature Experiment Complete")
    print(f"Run ID: {result.run_id}")
    print(f"Output: {result.output_dir}")
    print(f"Best model: {result.best_model_name}")
    best_metric = min(result.metrics, key=lambda m: m.mae)
    print(f"Best MAE: {best_metric.mae:.4f}")
    print(f"Baseline MAE: {CURRENT_BEST_FULL_MAE:.4f}")
    print(f"Delta: {best_metric.mae - CURRENT_BEST_FULL_MAE:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
