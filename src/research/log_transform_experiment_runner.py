"""
任务 1.1：对数变换总价评估运行器

核心逻辑：
1. 复用 pipeline 的数据加载、清洗、特征工程
2. 新增 log_total_price_wan（对数目标）和 log_area_sqm（对数面积特征）
3. 支持三种实验配置横向对比：
   - original：原始目标 + 原始特征（基准复现）
   - log_target：对数目标 + 原始特征（预测后 expm1 还原）
   - log_target_and_area：对数目标 + 对数面积替换原始面积
4. 所有评估统一在万元原始尺度上计算 MAE，确保与基准 stacking_ridge(MAE=20.9119) 可比
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


def run_log_transform_experiment(config: ResearchRunConfig) -> ResearchResult:
    """执行对数变换对比实验。"""
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_run_log(output_dir, "run_started", {"run_id": config.run_id, "experiment": "log_transform"})

    # 1. 数据加载与清洗（复用现有逻辑）
    houses_df = _load_houses(config.houses_path)
    communities_df = _load_communities(config.communities_path)
    merged_df, audit = _clean_and_join(houses_df, communities_df)

    # 2. 特征工程（复用现有逻辑）
    feature_df = _build_features(merged_df)
    feature_df = _apply_sample_limit(feature_df, config.sample_limit, config.random_state)

    # 3. 新增对数特征
    feature_df = _add_log_features(feature_df)

    # 4. 时间划分
    train_df, test_df = _split_by_time(feature_df, config.test_size)
    train_df, test_df = _apply_reference_group_medians(train_df, test_df, _compute_region_ai_medians(config.communities_path))

    # 5. 定义三种实验配置
    experiments = _build_experiment_configs()

    all_metrics: list[ModelMetric] = []
    all_predictions: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, JsonValue]] = []

    for exp_config in experiments:
        exp_name = str(exp_config["name"])
        target_col = str(exp_config["target_column"])
        numeric_features = cast(tuple[str, ...], exp_config["numeric_features"])
        categorical_features = cast(tuple[str, ...], exp_config["categorical_features"])
        use_log_target = bool(exp_config["use_log_target"])

        _write_run_log(
            output_dir,
            "experiment_started",
            {
                "experiment_name": exp_name,
                "target_column": target_col,
                "use_log_target": use_log_target,
            },
        )

        metrics, predictions_by_model = _run_experiment_variant(
            train_df=train_df,
            test_df=test_df,
            config=config,
            target_col=target_col,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            use_log_target=use_log_target,
            output_dir=output_dir,
            experiment_name=exp_name,
        )

        # 记录每个模型的指标
        for metric in metrics:
            all_metrics.append(metric)
            all_predictions[f"{exp_name}__{metric.model_name}"] = predictions_by_model[metric.model_name]

        best_metric = min(metrics, key=lambda m: m.mae)
        summary_rows.append(
            {
                "experiment_name": exp_name,
                "target_column": target_col,
                "use_log_target": use_log_target,
                "best_model_name": best_metric.model_name,
                "best_mae": best_metric.mae,
                "best_rmse": best_metric.rmse,
                "best_r2": best_metric.r2,
                "best_mape": best_metric.mape,
                "delta_mae_vs_baseline": best_metric.mae - CURRENT_BEST_FULL_MAE,
                "all_models": [m.model_name for m in metrics],
            }
        )

        _write_run_log(
            output_dir,
            "experiment_completed",
            {
                "experiment_name": exp_name,
                "best_model": best_metric.model_name,
                "best_mae": best_metric.mae,
            },
        )

    # 6. 写入产物
    _write_metrics(output_dir, all_metrics)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "log_transform_summary.csv", index=False, encoding="utf-8-sig"
    )

    # 写入 stacking_ridge 横向对比表
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


def _add_log_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """在特征 DataFrame 中新增对数特征。"""
    df = feature_df.copy()
    # 对数目标：log(1 + total_price_wan)
    df["log_total_price_wan"] = np.log1p(df[TARGET_COLUMN].clip(lower=1e-6))
    # 对数面积：log(1 + area_sqm)
    df["log_area_sqm"] = np.log1p(df["area_sqm"].clip(lower=1e-6))
    return df


def _build_experiment_configs() -> list[dict[str, JsonValue | tuple[str, ...] | bool]]:
    """构建三种实验配置。"""
    # 原始特征（数值特征中 area_sqm 保留原始值）
    original_numeric = NUMERIC_FEATURES
    # 对数面积替换原始面积
    log_area_numeric = tuple(
        "log_area_sqm" if f == "area_sqm" else f for f in NUMERIC_FEATURES
    )

    return [
        {
            "name": "original",
            "target_column": TARGET_COLUMN,
            "numeric_features": original_numeric,
            "categorical_features": CATEGORICAL_FEATURES,
            "use_log_target": False,
        },
        {
            "name": "log_target",
            "target_column": "log_total_price_wan",
            "numeric_features": original_numeric,
            "categorical_features": CATEGORICAL_FEATURES,
            "use_log_target": True,
        },
        {
            "name": "log_target_and_area",
            "target_column": "log_total_price_wan",
            "numeric_features": log_area_numeric,
            "categorical_features": CATEGORICAL_FEATURES,
            "use_log_target": True,
        },
    ]


def _run_experiment_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ResearchRunConfig,
    target_col: str,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    use_log_target: bool,
    output_dir: Path,
    experiment_name: str,
) -> tuple[list[ModelMetric], dict[str, np.ndarray]]:
    """执行单个实验变体的训练和评估（带 experiment 前缀）。"""
    x_train = train_df[list(numeric_features) + list(categorical_features)]
    y_train = train_df[target_col]
    x_test = test_df[list(numeric_features) + list(categorical_features)]
    y_test_original = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model_zoo = _build_model_zoo(config.random_state)
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
            predictions_transformed = pipeline.predict(x_test)
            predict_seconds = time.perf_counter() - predict_started_at

            if use_log_target:
                predictions_original = np.expm1(predictions_transformed)
            else:
                predictions_original = predictions_transformed

            predictions_original = np.clip(predictions_original, a_min=1.0, a_max=None)

            metric = _evaluate_predictions(
                prefixed_name,
                pd.Series(y_test_original),
                predictions_original,
                cv_mae_mean,
                cv_mae_std,
            )
            metrics.append(metric)
            predictions_by_model[prefixed_name] = predictions_original

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
    """提取并写入 stacking_ridge 的三种配置对比表。"""
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
        comparison_df = pd.DataFrame(rows).sort_values(by="mae")
        comparison_df.to_csv(
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
    parser = argparse.ArgumentParser(description="对数变换总价评估实验")
    parser.add_argument("--houses", required=True, help="房源JSONL路径")
    parser.add_argument("--communities", required=True, help="小区JSONL路径")
    parser.add_argument("--output-dir", required=True, help="研究产物根目录")
    parser.add_argument("--run-id", help="研究运行ID，不传则自动生成")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2, help="留出集比例")
    parser.add_argument("--cv-folds", type=int, default=3, help="交叉验证折数")
    parser.add_argument("--sample-limit", type=int, help="抽样上限，用于快速验证")
    args = parser.parse_args()

    run_id = args.run_id or f"log_transform_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
    result = run_log_transform_experiment(config)
    print("=" * 60)
    print("对数变换总价评估实验完成")
    print(f"运行ID: {result.run_id}")
    print(f"输出目录: {result.output_dir}")
    print(f"最优模型: {result.best_model_name}")
    best_metric = min(result.metrics, key=lambda m: m.mae)
    print(f"最优MAE: {best_metric.mae:.4f}")
    print(f"基准MAE: {CURRENT_BEST_FULL_MAE:.4f}")
    print(f"差值: {best_metric.mae - CURRENT_BEST_FULL_MAE:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
