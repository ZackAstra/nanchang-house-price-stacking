"""
组合突破实验：验证两个有效方向的叠加效果

核心逻辑：
1. 严格清洗（strict_cleaning，移除 4.5% 噪声样本）
2. 对数目标（log_total_price_wan）
3. stacking_ridge 训练，预测后 expm1 还原
4. 与基准（原始数据 + 原始目标）做横向对比

实验配置：
- original: 原始数据 + 原始目标（基准复现）
- log_only: 全量训练集 + 对数目标
- clean_only: 严格清洗训练集 + 原始目标
- combined: 严格清洗训练集 + 对数目标（组合最优）

评审修复协议（review2）：
1. 全量特征表先按时间划分训练/测试集，所有实验共用同一固定测试集；
2. 标签噪声检测只在训练集上运行，分位数边界仅由训练集估计；
3. 严格清洗只删除训练集样本，测试集保持不动。
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


def run_combined_breakthrough_experiment(config: ResearchRunConfig) -> ResearchResult:
    """执行组合突破实验。"""
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_run_log(output_dir, "run_started", {"run_id": config.run_id, "experiment": "combined_breakthrough"})

    # 1. 数据加载
    houses_df = _load_houses(config.houses_path)
    communities_df = _load_communities(config.communities_path)
    merged_df, audit = _clean_and_join(houses_df, communities_df)
    feature_df = _build_features(merged_df)
    feature_df = _apply_sample_limit(feature_df, config.sample_limit, config.random_state)

    # 2. 新增对数特征
    feature_df["log_total_price_wan"] = np.log1p(feature_df[TARGET_COLUMN].clip(lower=1e-6))

    # 3. 先按时间划分固定训练/测试集（所有实验共用同一测试集）
    train_full_df, fixed_test_df = _split_by_time(feature_df, config.test_size)
    train_full_df, fixed_test_df = _apply_reference_group_medians(train_full_df, fixed_test_df, _compute_region_ai_medians(config.communities_path))

    # 4. 噪声检测仅作用于训练集（分位数边界仅由训练集估计）
    noise_report = _detect_label_noise(train_full_df)
    strict_mask = ~noise_report["is_noise_strict"]

    _write_run_log(
        output_dir,
        "train_label_cleaning",
        {
            "train_full_samples": int(len(train_full_df)),
            "strict_removed_samples": int((~strict_mask).sum()),
            "strict_removal_rate": float((~strict_mask).mean()),
            "fixed_test_samples": int(len(fixed_test_df)),
        },
    )

    # 5. 定义四种实验配置（keep_mask 基于训练集索引）
    experiments = [
        {
            "name": "original",
            "target_col": TARGET_COLUMN,
            "keep_mask": pd.Series(True, index=train_full_df.index),
            "use_log_target": False,
        },
        {
            "name": "log_only",
            "target_col": "log_total_price_wan",
            "keep_mask": pd.Series(True, index=train_full_df.index),
            "use_log_target": True,
        },
        {
            "name": "clean_only",
            "target_col": TARGET_COLUMN,
            "keep_mask": strict_mask,
            "use_log_target": False,
        },
        {
            "name": "combined",
            "target_col": "log_total_price_wan",
            "keep_mask": strict_mask,
            "use_log_target": True,
        },
    ]

    all_metrics: list[ModelMetric] = []
    summary_rows: list[dict[str, JsonValue]] = []

    for exp_config in experiments:
        exp_name = str(exp_config["name"])
        target_col = str(exp_config["target_col"])
        keep_mask = cast(pd.Series, exp_config["keep_mask"])
        use_log_target = bool(exp_config["use_log_target"])

        # 清洗只作用于训练集，测试集固定不动
        train_df = train_full_df.loc[keep_mask].copy()
        test_df = fixed_test_df

        _write_run_log(
            output_dir,
            "experiment_started",
            {
                "experiment_name": exp_name,
                "target_col": target_col,
                "use_log_target": use_log_target,
                "train_full_samples": int(len(train_full_df)),
                "train_samples": int(len(train_df)),
                "removed_samples": int(len(train_full_df) - len(train_df)),
                "test_samples": int(len(test_df)),
            },
        )

        metrics, predictions_by_model = _run_combined_variant(
            train_df=train_df,
            test_df=test_df,
            config=config,
            target_col=target_col,
            use_log_target=use_log_target,
            output_dir=output_dir,
            experiment_name=exp_name,
        )

        all_metrics.extend(metrics)

        best_metric = min(metrics, key=lambda m: m.mae)
        summary_rows.append(
            {
                "experiment_name": exp_name,
                "target_col": target_col,
                "use_log_target": use_log_target,
                "kept_samples": int(len(train_df)),
                "removed_samples": int(len(train_full_df) - len(train_df)),
                "test_samples": int(len(test_df)),
                "best_model": best_metric.model_name.split("__")[-1],
                "mae": best_metric.mae,
                "rmse": best_metric.rmse,
                "r2": best_metric.r2,
                "mape": best_metric.mape,
                "delta_vs_baseline": best_metric.mae - CURRENT_BEST_FULL_MAE,
            }
        )

    # 6. 写入产物
    _write_metrics(output_dir, all_metrics)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "combined_experiment_summary.csv", index=False, encoding="utf-8-sig"
    )
    _write_stacking_comparison(output_dir, all_metrics)

    best_overall = min(all_metrics, key=lambda m: m.mae)
    _write_run_log(
        output_dir,
        "run_completed",
        {
            "best_experiment": best_overall.model_name,
            "best_mae": best_overall.mae,
            "baseline_mae": CURRENT_BEST_FULL_MAE,
            "delta_vs_baseline": best_overall.mae - CURRENT_BEST_FULL_MAE,
        },
    )

    print("=" * 60)
    print("Combined Breakthrough Experiment Complete")
    print(f"Run ID: {config.run_id}")
    print(f"Output: {output_dir}")
    print(f"Best model: {best_overall.model_name}")
    print(f"Best MAE: {best_overall.mae:.4f}")
    print(f"Baseline MAE: {CURRENT_BEST_FULL_MAE:.4f}")
    print(f"Delta: {best_overall.mae - CURRENT_BEST_FULL_MAE:.4f}")
    print("=" * 60)

    return ResearchResult(
        run_id=config.run_id,
        output_dir=output_dir,
        audit=audit,
        metrics=all_metrics,
        best_model_name=best_overall.model_name,
    )


def _detect_label_noise(feature_df: pd.DataFrame) -> pd.DataFrame:
    """检测标签噪声（复用 label_noise_experiment_runner 的逻辑）。"""
    df = feature_df.copy()
    df["unit_price"] = df[TARGET_COLUMN] / df["area_sqm"].clip(lower=1.0)

    # 单价异常
    group_keys = df["region_slug"].astype(str) + "__" + df["house_layout"].astype(str)
    df["unit_price_group"] = group_keys
    unit_price_low = df.groupby("unit_price_group")["unit_price"].transform(lambda x: x.quantile(0.01))
    unit_price_high = df.groupby("unit_price_group")["unit_price"].transform(lambda x: x.quantile(0.99))
    group_counts = df.groupby("unit_price_group")["unit_price"].transform("count")
    global_price_low = df["unit_price"].quantile(0.01)
    global_price_high = df["unit_price"].quantile(0.99)
    df["unit_price_low"] = np.where(group_counts >= 10, unit_price_low, global_price_low)
    df["unit_price_high"] = np.where(group_counts >= 10, unit_price_high, global_price_high)
    df["is_unit_price_outlier"] = (df["unit_price"] < df["unit_price_low"]) | (df["unit_price"] > df["unit_price_high"])

    # 总价异常
    total_price_low = df[TARGET_COLUMN].quantile(0.005)
    total_price_high = df[TARGET_COLUMN].quantile(0.995)
    df["is_total_price_outlier"] = (df[TARGET_COLUMN] < total_price_low) | (df[TARGET_COLUMN] > total_price_high)

    # 面积异常
    area_low = df["area_sqm"].quantile(0.005)
    area_high = df["area_sqm"].quantile(0.995)
    df["is_area_outlier"] = (df["area_sqm"] < area_low) | (df["area_sqm"] > area_high)

    df["is_noise_strict"] = df["is_unit_price_outlier"] | df["is_total_price_outlier"] | df["is_area_outlier"]

    return df


def _run_combined_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ResearchRunConfig,
    target_col: str,
    use_log_target: bool,
    output_dir: Path,
    experiment_name: str,
) -> tuple[list[ModelMetric], dict[str, np.ndarray]]:
    """执行单个实验变体。"""
    x_train = train_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_train = train_df[target_col]
    x_test = test_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_test_original = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model_zoo = _build_model_zoo(config.random_state)
    metrics: list[ModelMetric] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    groups = train_df["community_id"].fillna("unknown").astype(str)

    for model_name, regressor in model_zoo.items():
        prefixed_name = f"{experiment_name}__{model_name}"
        pipeline = _build_pipeline_for_features(regressor, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
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
    parser = argparse.ArgumentParser(description="组合突破实验")
    parser.add_argument("--houses", required=True, help="房源JSONL路径")
    parser.add_argument("--communities", required=True, help="小区JSONL路径")
    parser.add_argument("--output-dir", required=True, help="研究产物根目录")
    parser.add_argument("--run-id", help="研究运行ID，不传则自动生成")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2, help="留出集比例")
    parser.add_argument("--cv-folds", type=int, default=3, help="交叉验证折数")
    parser.add_argument("--sample-limit", type=int, help="抽样上限")
    args = parser.parse_args()

    run_id = args.run_id or f"combined_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
    result = run_combined_breakthrough_experiment(config)


if __name__ == "__main__":
    main()
