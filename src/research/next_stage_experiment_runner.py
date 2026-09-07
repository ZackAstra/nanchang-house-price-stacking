from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal
from typing import cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel
from pydantic import Field
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from src.research.models import JsonValue
from src.research.models import ModelMetric
from src.research.models import ResearchRunConfig
from src.research.pipeline import BASE_NUMERIC_FEATURES
from src.research.pipeline import CATEGORICAL_FEATURES
from src.research.pipeline import NUMERIC_FEATURES
from src.research.pipeline import TARGET_COLUMN
from src.research.pipeline import _apply_sample_limit
from src.research.pipeline import _build_features
from src.research.pipeline import _build_model_zoo
from src.research.pipeline import _build_pipeline_for_features
from src.research.pipeline import _clean_and_join
from src.research.pipeline import _cross_validate_model
from src.research.pipeline import _evaluate_predictions
from src.research.pipeline import _load_communities
from src.research.pipeline import _load_houses
from src.research.pipeline import _model_params
from src.research.pipeline import _split_by_time
from src.research.pipeline import _apply_reference_group_medians
from src.research.pipeline import _compute_region_ai_medians
from src.research.pipeline import _write_error_stratification
from src.research.pipeline import _write_feature_generation_log
from src.research.pipeline import _write_feature_importance
from src.research.pipeline import _write_metrics
from src.research.pipeline import _write_model_prediction_errors
from src.research.pipeline import _write_plots
from src.research.pipeline import _write_shap_analysis


CURRENT_BEST_FULL_MAE = 20.911922387801575
ExperimentMode = Literal[
    "ensemble_control",
    "algorithm_baseline",
    "stacking_meta_optimization",
    "advanced_algorithm_comparison",
]
POI_STRUCTURE_FEATURES: tuple[str, ...] = (
    "poi_balance_score",
    "transit_medical_ratio",
    "education_commerce_ratio",
)


class NextStageCandidateResult(BaseModel):
    model_name: str = Field(min_length=1, description="候选模型名称")
    feature_set: str = Field(min_length=1, description="特征集合名称")
    member_models: str = Field(min_length=1, description="Stacking基学习器")
    ensemble_method: str = Field(min_length=1, description="融合方法")
    combination_name: str = Field(min_length=1, description="基学习器组合名称")
    removed_member: str | None = Field(default=None, description="移除的基学习器")
    metric: ModelMetric = Field(description="模型指标")
    predictions: list[float] = Field(description="测试集预测结果")


class FWLSLiteRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimators: list[tuple[str, RegressorMixin]],
        meta_feature_indices: tuple[int, ...],
        alphas: tuple[float, ...],
        random_state: int,
    ) -> None:
        self.estimators = estimators
        self.meta_feature_indices = meta_feature_indices
        self.alphas = alphas
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FWLSLiteRegressor":
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        self.estimators_ = [(name, cast(RegressorMixin, clone(model))) for name, model in self.estimators]
        oof_predictions = np.zeros((x_array.shape[0], len(self.estimators_)), dtype=float)
        splitter = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        for train_idx, valid_idx in splitter.split(x_array):
            for model_idx, (_, estimator) in enumerate(self.estimators_):
                fold_estimator = cast(RegressorMixin, clone(estimator))
                fold_estimator.fit(x_array[train_idx], y_array[train_idx])
                oof_predictions[valid_idx, model_idx] = fold_estimator.predict(x_array[valid_idx])
        self.fitted_estimators_ = []
        for name, estimator in self.estimators_:
            fitted_estimator = cast(RegressorMixin, clone(estimator))
            fitted_estimator.fit(x_array, y_array)
            self.fitted_estimators_.append((name, fitted_estimator))
        meta_matrix = self._build_meta_matrix(x_array, oof_predictions)
        self.meta_model_ = RidgeCV(alphas=np.array(self.alphas, dtype=float))
        self.meta_model_.fit(meta_matrix, y_array)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x)
        base_predictions = np.column_stack([estimator.predict(x_array) for _, estimator in self.fitted_estimators_])
        meta_matrix = self._build_meta_matrix(x_array, base_predictions)
        return cast(np.ndarray, self.meta_model_.predict(meta_matrix))

    def _build_meta_matrix(self, x_array: np.ndarray, base_predictions: np.ndarray) -> np.ndarray:
        meta_features = x_array[:, list(self.meta_feature_indices)]
        prediction_std = np.std(base_predictions, axis=1, keepdims=True)
        features = np.column_stack([np.ones((x_array.shape[0], 1)), meta_features, prediction_std])
        interaction_blocks = [base_predictions * features[:, [feature_idx]] for feature_idx in range(features.shape[1])]
        return np.column_stack(interaction_blocks)


class SuperLearnerConvexRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators: list[tuple[str, RegressorMixin]], random_state: int) -> None:
        self.estimators = estimators
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SuperLearnerConvexRegressor":
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        self.estimators_ = [(name, cast(RegressorMixin, clone(model))) for name, model in self.estimators]
        oof_predictions = np.zeros((x_array.shape[0], len(self.estimators_)), dtype=float)
        splitter = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        for train_idx, valid_idx in splitter.split(x_array):
            for model_idx, (_, estimator) in enumerate(self.estimators_):
                fold_estimator = cast(RegressorMixin, clone(estimator))
                fold_estimator.fit(x_array[train_idx], y_array[train_idx])
                oof_predictions[valid_idx, model_idx] = fold_estimator.predict(x_array[valid_idx])

        initial_weights = np.ones(len(self.estimators_), dtype=float) / len(self.estimators_)
        optimized = minimize(
            lambda weights: float(np.mean((y_array - oof_predictions @ weights) ** 2)),
            initial_weights,
            method="SLSQP",
            bounds=[(0.0, 1.0) for _ in self.estimators_],
            constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        )
        self.weights_ = optimized.x if optimized.success else initial_weights
        self.weights_ = np.clip(self.weights_, 0.0, 1.0)
        weight_sum = float(np.sum(self.weights_))
        self.weights_ = self.weights_ / weight_sum if weight_sum > 0 else initial_weights
        self.oof_predictions_ = oof_predictions
        self.fitted_estimators_ = []
        for name, estimator in self.estimators_:
            fitted_estimator = cast(RegressorMixin, clone(estimator))
            fitted_estimator.fit(x_array, y_array)
            self.fitted_estimators_.append((name, fitted_estimator))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x)
        base_predictions = np.column_stack([estimator.predict(x_array) for _, estimator in self.fitted_estimators_])
        return cast(np.ndarray, base_predictions @ self.weights_)


class MoERegionAreaLiteRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        base_estimator: RegressorMixin,
        area_feature_index: int,
        region_feature_start: int,
        region_feature_count: int,
        min_samples: int,
        random_state: int,
    ) -> None:
        self.base_estimator = base_estimator
        self.area_feature_index = area_feature_index
        self.region_feature_start = region_feature_start
        self.region_feature_count = region_feature_count
        self.min_samples = min_samples
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MoERegionAreaLiteRegressor":
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        self.global_estimator_ = cast(RegressorMixin, clone(self.base_estimator))
        self.global_estimator_.fit(x_array, y_array)
        area_values = x_array[:, self.area_feature_index]
        self.area_quantiles_ = np.quantile(area_values, [0.33, 0.67])
        gate_keys = self._gate_keys(x_array)
        self.experts_: dict[str, RegressorMixin] = {}
        self.expert_profiles_: list[dict[str, JsonValue]] = []
        for gate_key in sorted(set(gate_keys)):
            indices = np.flatnonzero(gate_keys == gate_key)
            use_expert = len(indices) >= self.min_samples
            if use_expert:
                expert = cast(RegressorMixin, clone(self.base_estimator))
                expert.fit(x_array[indices], y_array[indices])
                self.experts_[gate_key] = expert
            self.expert_profiles_.append(
                {
                    "gate_key": gate_key,
                    "sample_count": int(len(indices)),
                    "uses_local_expert": bool(use_expert),
                }
            )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x)
        predictions = self.global_estimator_.predict(x_array)
        gate_keys = self._gate_keys(x_array)
        for gate_key, expert in self.experts_.items():
            indices = np.flatnonzero(gate_keys == gate_key)
            if len(indices) > 0:
                predictions[indices] = expert.predict(x_array[indices])
        return cast(np.ndarray, predictions)

    def _gate_keys(self, x_array: np.ndarray) -> np.ndarray:
        area_bins = np.digitize(x_array[:, self.area_feature_index], self.area_quantiles_, right=False)
        region_slice = x_array[:, self.region_feature_start : self.region_feature_start + self.region_feature_count]
        region_indices = np.argmax(region_slice, axis=1) if self.region_feature_count > 0 else np.zeros(x_array.shape[0], dtype=int)
        return np.array([f"region_{region}_area_{area}" for region, area in zip(region_indices, area_bins, strict=True)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下一阶段Stacking优化候选验证入口")
    parser.add_argument("--houses", required=True, help="房源JSONL路径")
    parser.add_argument("--communities", required=True, help="小区JSONL路径")
    parser.add_argument("--output-dir", required=True, help="研究产物根目录")
    parser.add_argument("--run-id", help="研究运行ID，不传则自动生成")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2, help="留出集比例")
    parser.add_argument("--cv-folds", type=int, default=3, help="交叉验证折数")
    parser.add_argument("--sample-limit", type=int, help="抽样上限，用于快速验证")
    parser.add_argument(
        "--experiment-mode",
        choices=["ensemble_control", "algorithm_baseline", "stacking_meta_optimization", "advanced_algorithm_comparison"],
        default="ensemble_control",
        help="实验模式",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"next_stage_full_validation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
    output_dir = run_next_stage_experiment(config, cast(ExperimentMode, args.experiment_mode))
    print("=" * 60)
    print("下一阶段Stacking优化候选验证完成")
    print(f"运行ID: {config.run_id}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


def run_next_stage_experiment(config: ResearchRunConfig, experiment_mode: ExperimentMode) -> Path:
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    houses_df = _load_houses(config.houses_path)
    communities_df = _load_communities(config.communities_path)
    merged_df, audit = _clean_and_join(houses_df, communities_df)
    feature_df = _build_features(merged_df)
    feature_df = _apply_sample_limit(feature_df, config.sample_limit, config.random_state)
    feature_df.to_csv(output_dir / "cleaned_features.csv", index=False, encoding="utf-8-sig")
    _write_feature_generation_log(output_dir)
    (output_dir / "data_audit.json").write_text(audit.model_dump_json(indent=2), encoding="utf-8")

    train_df, test_df = _split_by_time(feature_df, config.test_size)
    train_df, test_df = _apply_reference_group_medians(
        train_df, test_df, _compute_region_ai_medians(config.communities_path)
    )
    results, pipelines = _run_candidate_models(train_df, test_df, config, output_dir, experiment_mode)
    metrics = [result.metric for result in results]
    _write_metrics(output_dir, metrics)
    _write_candidate_artifacts(output_dir, results, experiment_mode, pipelines, train_df, test_df)

    predictions_by_model = {result.model_name: np.array(result.predictions, dtype=float) for result in results}
    _write_model_prediction_errors(output_dir, test_df, predictions_by_model)
    best_result = min(results, key=lambda item: item.metric.mae)
    best_predictions = predictions_by_model[best_result.model_name]
    predictions_df = test_df[["house_id", TARGET_COLUMN]].copy()
    predictions_df["predicted_price_wan"] = best_predictions
    predictions_df["residual_wan"] = predictions_df[TARGET_COLUMN] - predictions_df["predicted_price_wan"]
    predictions_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    _write_error_stratification(output_dir, test_df, predictions_df)
    _write_feature_importance(output_dir, pipelines[best_result.model_name], train_df, test_df, config.random_state)
    _write_shap_analysis(output_dir, pipelines[best_result.model_name], train_df, test_df, config.random_state)
    _write_plots(output_dir, feature_df, predictions_df)
    _write_summary(output_dir, config, results, experiment_mode)
    return output_dir


def _run_candidate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ResearchRunConfig,
    output_dir: Path,
    experiment_mode: ExperimentMode,
) -> tuple[list[NextStageCandidateResult], dict[str, Pipeline]]:
    model_zoo = _build_model_zoo(config.random_state)
    candidate_specs = _build_candidate_specs(model_zoo, config.random_state, experiment_mode, train_df)
    results: list[NextStageCandidateResult] = []
    pipelines: dict[str, Pipeline] = {}
    groups = train_df["community_id"].fillna("unknown").astype(str)
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    for spec in candidate_specs:
        model_name = str(spec["model_name"])
        feature_set = str(spec["feature_set"])
        member_models = str(spec["member_models"])
        ensemble_method = str(spec["ensemble_method"])
        combination_name = str(spec["combination_name"])
        removed_member_value = spec["removed_member"]
        removed_member = str(removed_member_value) if removed_member_value is not None else None
        numeric_features = cast(tuple[str, ...], spec["numeric_features"])
        categorical_features = cast(tuple[str, ...], spec["categorical_features"])
        regressor = cast(RegressorMixin, spec["regressor"])
        x_train = train_df[list(numeric_features) + list(categorical_features)]
        x_test = test_df[list(numeric_features) + list(categorical_features)]
        pipeline = _build_pipeline_for_features(regressor, numeric_features, categorical_features)

        _append_next_stage_experiment(
            output_dir,
            {
                "event": "model_started",
                "model_name": model_name,
                "feature_set": feature_set,
                "member_models": member_models,
                "ensemble_method": ensemble_method,
                "combination_name": combination_name,
                "removed_member": removed_member,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "params": _model_params(regressor),
            },
        )
        started_at = time.perf_counter()
        warning_messages: list[str] = []
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", category=ConvergenceWarning)
            cv_mae_mean, cv_mae_std, cv_mae_scores = _cross_validate_model(
                pipeline,
                x_train,
                y_train,
                groups,
                config.cv_folds,
            )
            pipeline.fit(x_train, y_train)
            warning_messages = [str(warning.message) for warning in caught_warnings]
        train_seconds = time.perf_counter() - started_at
        predict_started_at = time.perf_counter()
        predictions = pipeline.predict(x_test)
        predict_seconds = time.perf_counter() - predict_started_at
        metric = _evaluate_predictions(model_name, y_test, predictions, cv_mae_mean, cv_mae_std)
        _append_next_stage_experiment(
            output_dir,
            {
                "event": "model_completed",
                "model_name": model_name,
                "feature_set": feature_set,
                "member_models": member_models,
                "ensemble_method": ensemble_method,
                "combination_name": combination_name,
                "removed_member": removed_member,
                "cv_mae_scores": [float(score) for score in cv_mae_scores],
                "cv_mae_mean": cv_mae_mean,
                "cv_mae_std": cv_mae_std,
                "mae": metric.mae,
                "rmse": metric.rmse,
                "r2": metric.r2,
                "mape": metric.mape,
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "warnings": warning_messages,
            },
        )
        results.append(
            NextStageCandidateResult(
                model_name=model_name,
                feature_set=feature_set,
                member_models=member_models,
                ensemble_method=ensemble_method,
                combination_name=combination_name,
                removed_member=removed_member,
                metric=metric,
                predictions=[float(value) for value in predictions],
            )
        )
        pipelines[model_name] = pipeline
    return results, pipelines


def _build_candidate_specs(
    model_zoo: dict[str, RegressorMixin],
    random_state: int,
    experiment_mode: ExperimentMode,
    train_df: pd.DataFrame,
) -> list[dict[str, JsonValue | tuple[str, ...] | RegressorMixin]]:
    if experiment_mode == "algorithm_baseline":
        return _build_algorithm_baseline_specs(model_zoo, random_state)
    if experiment_mode == "stacking_meta_optimization":
        return _build_stacking_meta_optimization_specs(model_zoo, random_state)
    if experiment_mode == "advanced_algorithm_comparison":
        return _build_advanced_algorithm_specs(model_zoo, random_state, train_df)
    return _build_ensemble_control_specs(model_zoo, random_state)


def _build_ensemble_control_specs(
    model_zoo: dict[str, RegressorMixin],
    random_state: int,
) -> list[dict[str, JsonValue | tuple[str, ...] | RegressorMixin]]:
    full_member_names = _available_stacking_member_names(model_zoo, ())
    combination_specs: list[tuple[str, tuple[str, ...], str | None]] = [("full", full_member_names, None)]
    for removed_member in full_member_names:
        member_names = tuple(model_name for model_name in full_member_names if model_name != removed_member)
        if len(member_names) < 2:
            continue
        combination_specs.append((f"without_{removed_member}", member_names, removed_member))

    specs: list[dict[str, JsonValue | tuple[str, ...] | RegressorMixin]] = []
    for combination_name, member_names, removed_member in combination_specs:
        specs.append(
            {
                "model_name": f"stacking_{combination_name}",
                "feature_set": "stage3_enhanced_features",
                "member_models": "|".join(member_names),
                "ensemble_method": "stacking_ridge",
                "combination_name": combination_name,
                "removed_member": removed_member,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "regressor": _build_stacking_regressor(model_zoo, member_names, random_state),
            }
        )
        specs.append(
            {
                "model_name": f"voting_{combination_name}",
                "feature_set": "stage3_enhanced_features",
                "member_models": "|".join(member_names),
                "ensemble_method": "voting",
                "combination_name": combination_name,
                "removed_member": removed_member,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "regressor": _build_voting_regressor(model_zoo, member_names),
            }
        )
    return specs


def _build_algorithm_baseline_specs(
    model_zoo: dict[str, RegressorMixin],
    random_state: int,
) -> list[dict[str, JsonValue | tuple[str, ...] | RegressorMixin]]:
    full_member_names = _available_stacking_member_names(model_zoo, ())
    return [
        {
            "model_name": "stacking_full",
            "feature_set": "stage3_enhanced_features",
            "member_models": "|".join(full_member_names),
            "ensemble_method": "stacking_ridge",
            "combination_name": "algorithm_baseline",
            "removed_member": None,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "regressor": _build_stacking_regressor(model_zoo, full_member_names, random_state),
        },
        {
            "model_name": "svr_rbf",
            "feature_set": "stage3_enhanced_features",
            "member_models": "not_applicable",
            "ensemble_method": "single_model",
            "combination_name": "algorithm_baseline",
            "removed_member": None,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "regressor": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale", cache_size=1000),
        },
        {
            "model_name": "svr_linear",
            "feature_set": "stage3_enhanced_features",
            "member_models": "not_applicable",
            "ensemble_method": "single_model",
            "combination_name": "algorithm_baseline",
            "removed_member": None,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "regressor": SVR(kernel="linear", C=1.0, epsilon=0.1, cache_size=1000),
        },
        {
            "model_name": "mlp_regressor",
            "feature_set": "stage3_enhanced_features",
            "member_models": "not_applicable",
            "ensemble_method": "single_model",
            "combination_name": "algorithm_baseline",
            "removed_member": None,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "regressor": MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=random_state,
            ),
        },
    ]


def _build_stacking_meta_optimization_specs(
    model_zoo: dict[str, RegressorMixin],
    random_state: int,
) -> list[dict[str, JsonValue | tuple[str, ...] | RegressorMixin]]:
    full_member_names = _available_stacking_member_names(model_zoo, ())
    compact_member_names = tuple(
        model_name for model_name in ("random_forest", "hist_gradient_boosting", "xgboost") if model_name in model_zoo
    )
    ridge_member_names = full_member_names + ("ridge",)
    diverse_member_names = tuple(
        model_name for model_name in ("random_forest", "hist_gradient_boosting", "xgboost", "ridge", "svr_rbf")
    )
    extended_model_zoo = {
        **model_zoo,
        "svr_rbf": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale", cache_size=1000),
    }
    alpha_values = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    specs: list[dict[str, JsonValue | tuple[str, ...] | RegressorMixin]] = [
        _stacking_spec(
            "stacking_ridge_default",
            full_member_names,
            "meta_learner",
            _build_stacking_regressor(model_zoo, full_member_names, random_state),
        ),
        _stacking_spec(
            "stacking_ridge_cv",
            full_member_names,
            "meta_learner",
            _build_stacking_regressor_with_final(
                model_zoo,
                full_member_names,
                RidgeCV(alphas=np.array(alpha_values, dtype=float)),
            ),
        ),
        _stacking_spec(
            "stacking_lasso_cv",
            full_member_names,
            "meta_learner",
            _build_stacking_regressor_with_final(
                model_zoo,
                full_member_names,
                LassoCV(alphas=np.array(alpha_values, dtype=float), cv=3, random_state=random_state, max_iter=10000),
            ),
        ),
        _stacking_spec(
            "stacking_elasticnet_cv",
            full_member_names,
            "meta_learner",
            _build_stacking_regressor_with_final(
                model_zoo,
                full_member_names,
                ElasticNetCV(
                    alphas=np.array(alpha_values, dtype=float),
                    l1_ratio=[0.2, 0.5, 0.8],
                    cv=3,
                    random_state=random_state,
                    max_iter=10000,
                ),
            ),
        ),
        _stacking_spec(
            "stacking_plus_linear_ridge",
            ridge_member_names,
            "diverse_stacking",
            _build_stacking_regressor(extended_model_zoo, ridge_member_names, random_state),
        ),
        _stacking_spec(
            "stacking_plus_svr_rbf",
            full_member_names + ("svr_rbf",),
            "diverse_stacking",
            _build_stacking_regressor(extended_model_zoo, full_member_names + ("svr_rbf",), random_state),
        ),
    ]
    if len(compact_member_names) >= 2:
        specs.append(
            _stacking_spec(
                "stacking_diverse_compact",
                diverse_member_names,
                "diverse_stacking",
                _build_stacking_regressor(extended_model_zoo, diverse_member_names, random_state),
            )
        )
        specs.append(
            _stacking_spec(
                "fwls_lite",
                diverse_member_names,
                "fwls_lite",
                _build_fwls_lite_regressor(extended_model_zoo, diverse_member_names, random_state),
            )
        )
    for alpha in alpha_values:
        specs.append(
            _stacking_spec(
                f"stacking_ridge_alpha_{str(alpha).replace('.', '_')}",
                full_member_names,
                "meta_alpha_grid",
                _build_stacking_regressor_with_final(model_zoo, full_member_names, Ridge(alpha=alpha, random_state=random_state)),
            )
        )
    return specs


def _build_advanced_algorithm_specs(
    model_zoo: dict[str, RegressorMixin],
    random_state: int,
    train_df: pd.DataFrame,
) -> list[dict[str, JsonValue | tuple[str, ...] | RegressorMixin]]:
    full_member_names = _available_stacking_member_names(model_zoo, ())
    super_member_names = tuple(
        model_name for model_name in ("random_forest", "hist_gradient_boosting", "xgboost", "ridge", "svr_rbf")
    )
    extended_model_zoo = {
        **model_zoo,
        "svr_rbf": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale", cache_size=1000),
    }
    region_feature_count = int(train_df["region_slug"].astype("string").fillna("unknown").nunique())
    return [
        _stacking_spec(
            "stacking_full",
            full_member_names,
            "advanced_reference",
            _build_stacking_regressor(model_zoo, full_member_names, random_state),
        ),
        _stacking_spec(
            "super_learner_convex",
            super_member_names,
            "advanced_algorithm",
            _build_super_learner_regressor(extended_model_zoo, super_member_names, random_state),
        ),
        _stacking_spec(
            "moe_region_area_lite",
            ("hist_gradient_boosting",),
            "advanced_algorithm",
            _build_moe_region_area_lite_regressor(model_zoo, region_feature_count, random_state),
        ),
    ]


def _stacking_spec(
    model_name: str,
    member_names: tuple[str, ...],
    combination_name: str,
    regressor: RegressorMixin,
) -> dict[str, JsonValue | tuple[str, ...] | RegressorMixin]:
    return {
        "model_name": model_name,
        "feature_set": "stage3_enhanced_features",
        "member_models": "|".join(member_names),
        "ensemble_method": "stacking_ridge" if combination_name != "fwls_lite" else "fwls_lite",
        "combination_name": combination_name,
        "removed_member": None,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "regressor": regressor,
    }


def _available_stacking_member_names(
    model_zoo: dict[str, RegressorMixin],
    excluded_names: tuple[str, ...],
) -> tuple[str, ...]:
    candidate_names = ("random_forest", "gradient_boosting", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost")
    return tuple(model_name for model_name in candidate_names if model_name in model_zoo and model_name not in excluded_names)


def _build_stacking_regressor(
    model_zoo: dict[str, RegressorMixin],
    member_names: tuple[str, ...],
    random_state: int,
) -> StackingRegressor:
    estimators = [(model_name, cast(RegressorMixin, clone(model_zoo[model_name]))) for model_name in member_names]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0, random_state=random_state),
        cv=3,
        n_jobs=None,
    )


def _build_stacking_regressor_with_final(
    model_zoo: dict[str, RegressorMixin],
    member_names: tuple[str, ...],
    final_estimator: RegressorMixin,
) -> StackingRegressor:
    estimators = [(model_name, cast(RegressorMixin, clone(model_zoo[model_name]))) for model_name in member_names]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=3,
        n_jobs=None,
    )


def _build_fwls_lite_regressor(
    model_zoo: dict[str, RegressorMixin],
    member_names: tuple[str, ...],
    random_state: int,
) -> FWLSLiteRegressor:
    meta_feature_indices = tuple(
        list(NUMERIC_FEATURES).index(feature_name)
        for feature_name in ("area_sqm", "poi_balance_score", "poi_subway_count")
        if feature_name in NUMERIC_FEATURES
    )
    estimators = [(model_name, cast(RegressorMixin, clone(model_zoo[model_name]))) for model_name in member_names]
    return FWLSLiteRegressor(
        estimators=estimators,
        meta_feature_indices=meta_feature_indices,
        alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
        random_state=random_state,
    )


def _build_super_learner_regressor(
    model_zoo: dict[str, RegressorMixin],
    member_names: tuple[str, ...],
    random_state: int,
) -> SuperLearnerConvexRegressor:
    estimators = [(model_name, cast(RegressorMixin, clone(model_zoo[model_name]))) for model_name in member_names]
    return SuperLearnerConvexRegressor(estimators=estimators, random_state=random_state)


def _build_moe_region_area_lite_regressor(
    model_zoo: dict[str, RegressorMixin],
    region_feature_count: int,
    random_state: int,
) -> MoERegionAreaLiteRegressor:
    return MoERegionAreaLiteRegressor(
        base_estimator=cast(RegressorMixin, clone(model_zoo["hist_gradient_boosting"])),
        area_feature_index=0,
        region_feature_start=len(NUMERIC_FEATURES),
        region_feature_count=region_feature_count,
        min_samples=80,
        random_state=random_state,
    )


def _build_voting_regressor(
    model_zoo: dict[str, RegressorMixin],
    member_names: tuple[str, ...],
) -> VotingRegressor:
    estimators = [(model_name, cast(RegressorMixin, clone(model_zoo[model_name]))) for model_name in member_names]
    return VotingRegressor(estimators=estimators, n_jobs=None)


def _write_candidate_artifacts(
    output_dir: Path,
    results: list[NextStageCandidateResult],
    experiment_mode: ExperimentMode,
    pipelines: dict[str, Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    ablation_rows: list[dict[str, JsonValue]] = []
    for result in results:
        ablation_rows.append(
            {
                "variant_name": result.model_name,
                "ensemble_method": result.ensemble_method,
                "combination_name": result.combination_name,
                "removed_member": result.removed_member,
                "member_models": result.member_models,
                "feature_set": result.feature_set,
                **result.metric.model_dump(),
            }
        )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(
        output_dir / "ensemble_member_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ablation_df.to_csv(
        output_dir / "stacking_component_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison_rows: list[dict[str, JsonValue]] = []
    for combination_name, combination_df in ablation_df.groupby("combination_name", dropna=False):
        stacking_rows = combination_df.loc[combination_df["ensemble_method"] == "stacking_ridge"]
        voting_rows = combination_df.loc[combination_df["ensemble_method"] == "voting"]
        if len(stacking_rows) != 1 or len(voting_rows) != 1:
            continue
        stacking_row = stacking_rows.iloc[0]
        voting_row = voting_rows.iloc[0]
        comparison_rows.append(
            {
                "combination_name": str(combination_name),
                "removed_member": None if pd.isna(stacking_row["removed_member"]) else str(stacking_row["removed_member"]),
                "member_models": str(stacking_row["member_models"]),
                "stacking_model_name": str(stacking_row["variant_name"]),
                "voting_model_name": str(voting_row["variant_name"]),
                "stacking_mae": float(stacking_row["mae"]),
                "voting_mae": float(voting_row["mae"]),
                "voting_minus_stacking_mae": float(voting_row["mae"]) - float(stacking_row["mae"]),
                "stacking_cv_mae_mean": float(stacking_row["cv_mae_mean"]),
                "voting_cv_mae_mean": float(voting_row["cv_mae_mean"]),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "voting_vs_stacking_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_rows: list[dict[str, JsonValue]] = [
        {
            "feature_set": result.feature_set,
            "model_name": result.model_name,
            "numeric_feature_count": _numeric_feature_count(result.feature_set),
            "categorical_feature_count": len(CATEGORICAL_FEATURES),
            **result.metric.model_dump(),
        }
        for result in results
    ]
    pd.DataFrame(feature_rows).to_csv(
        output_dir / "stable_feature_group_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if experiment_mode == "algorithm_baseline":
        pd.DataFrame([result.metric.model_dump() for result in results]).sort_values(by="mae").to_csv(
            output_dir / "algorithm_baseline_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _write_algorithm_baseline_summary(output_dir, results)
    if experiment_mode == "stacking_meta_optimization":
        _write_stacking_meta_artifacts(output_dir, results, pipelines, train_df, test_df)
    if experiment_mode == "advanced_algorithm_comparison":
        _write_advanced_algorithm_artifacts(output_dir, results, pipelines, train_df, test_df)


def _numeric_feature_count(feature_set: str) -> int:
    if feature_set == "base_plus_poi_structure":
        return len(BASE_NUMERIC_FEATURES + POI_STRUCTURE_FEATURES)
    return len(NUMERIC_FEATURES)


def _write_stacking_meta_artifacts(
    output_dir: Path,
    results: list[NextStageCandidateResult],
    pipelines: dict[str, Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    metrics_df = pd.DataFrame([result.metric.model_dump() for result in results]).sort_values(by="mae")
    metrics_df.to_csv(output_dir / "stacking_meta_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df.loc[metrics_df["model_name"].str.contains("plus|diverse", regex=True)].to_csv(
        output_dir / "diverse_stacking_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_df.loc[metrics_df["model_name"].eq("fwls_lite")].to_csv(
        output_dir / "fwls_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_meta_learner_coefficients(output_dir, pipelines)
    _write_meta_learner_alpha_search(output_dir, results, pipelines)
    _write_base_prediction_diagnostics(output_dir, train_df, test_df)
    _write_fwls_artifacts(output_dir, pipelines, test_df)
    _write_stacking_meta_summary(output_dir, results)


def _write_advanced_algorithm_artifacts(
    output_dir: Path,
    results: list[NextStageCandidateResult],
    pipelines: dict[str, Pipeline],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    pd.DataFrame([result.metric.model_dump() for result in results]).sort_values(by="mae").to_csv(
        output_dir / "advanced_algorithm_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_super_learner_artifacts(output_dir, pipelines, train_df)
    _write_moe_lite_artifacts(output_dir, pipelines, test_df)
    _write_advanced_algorithm_summary(output_dir, results)


def _write_super_learner_artifacts(output_dir: Path, pipelines: dict[str, Pipeline], train_df: pd.DataFrame) -> None:
    pipeline = pipelines.get("super_learner_convex")
    if pipeline is None:
        pd.DataFrame([]).to_csv(output_dir / "super_learner_weights.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([]).to_csv(output_dir / "super_learner_oof_predictions.csv", index=False, encoding="utf-8-sig")
        return
    regressor = pipeline.named_steps["regressor"]
    if not isinstance(regressor, SuperLearnerConvexRegressor):
        return
    member_names = [name for name, _ in regressor.fitted_estimators_]
    pd.DataFrame(
        [{"member_name": member_name, "weight": float(weight)} for member_name, weight in zip(member_names, regressor.weights_, strict=True)]
    ).to_csv(output_dir / "super_learner_weights.csv", index=False, encoding="utf-8-sig")
    oof_df = pd.DataFrame(regressor.oof_predictions_, columns=member_names)
    oof_df.insert(0, "house_id", train_df["house_id"].astype(str).to_numpy())
    oof_df.to_csv(output_dir / "super_learner_oof_predictions.csv", index=False, encoding="utf-8-sig")


def _write_moe_lite_artifacts(output_dir: Path, pipelines: dict[str, Pipeline], test_df: pd.DataFrame) -> None:
    pipeline = pipelines.get("moe_region_area_lite")
    if pipeline is None:
        pd.DataFrame([]).to_csv(output_dir / "moe_gate_profile.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([]).to_csv(output_dir / "moe_expert_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([]).to_csv(output_dir / "moe_predictions.csv", index=False, encoding="utf-8-sig")
        return
    regressor = pipeline.named_steps["regressor"]
    if not isinstance(regressor, MoERegionAreaLiteRegressor):
        return
    pd.DataFrame(regressor.expert_profiles_).to_csv(output_dir / "moe_gate_profile.csv", index=False, encoding="utf-8-sig")
    x_test = test_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    transformed_test = pipeline.named_steps["preprocessor"].transform(x_test)
    predictions = pipeline.predict(x_test)
    gate_keys = regressor._gate_keys(transformed_test)
    prediction_df = pd.DataFrame(
        {
            "house_id": test_df["house_id"].astype(str).to_numpy(),
            "gate_key": gate_keys,
            "actual_price_wan": y_test,
            "predicted_price_wan": predictions,
            "absolute_error_wan": np.abs(y_test - predictions),
        }
    )
    prediction_df.to_csv(output_dir / "moe_predictions.csv", index=False, encoding="utf-8-sig")
    rows: list[dict[str, JsonValue]] = []
    for gate_key, group_df in prediction_df.groupby("gate_key", dropna=False):
        rows.append(
            {
                "gate_key": str(gate_key),
                "sample_count": int(len(group_df)),
                "mae": float(group_df["absolute_error_wan"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "moe_expert_metrics.csv", index=False, encoding="utf-8-sig")


def _write_advanced_algorithm_summary(output_dir: Path, results: list[NextStageCandidateResult]) -> None:
    best_result = min(results, key=lambda item: item.metric.mae)
    reference_result = next(result for result in results if result.model_name == "stacking_full")
    summary: dict[str, JsonValue] = {
        "best_model": best_result.model_name,
        "best_model_mae": best_result.metric.mae,
        "reference_model": reference_result.model_name,
        "reference_mae": reference_result.metric.mae,
        "mae_delta_vs_reference": best_result.metric.mae - reference_result.metric.mae,
        "mae_delta_vs_current_full_best": best_result.metric.mae - CURRENT_BEST_FULL_MAE,
        "update_current_best": best_result.metric.mae < CURRENT_BEST_FULL_MAE,
        "candidate_models": [result.model_name for result in results],
    }
    (output_dir / "advanced_algorithm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_meta_learner_coefficients(output_dir: Path, pipelines: dict[str, Pipeline]) -> None:
    rows: list[dict[str, JsonValue]] = []
    for model_name, pipeline in pipelines.items():
        regressor = pipeline.named_steps["regressor"]
        if isinstance(regressor, StackingRegressor):
            final_estimator = getattr(regressor, "final_estimator_", None)
            coefficients = getattr(final_estimator, "coef_", None)
            estimator_names = [name for name, _ in regressor.estimators]
            if coefficients is not None:
                coefficient_array = np.ravel(np.asarray(coefficients, dtype=float))
                for member_name, coefficient in zip(estimator_names, coefficient_array, strict=False):
                    rows.append(
                        {
                            "model_name": model_name,
                            "member_name": member_name,
                            "coefficient": float(coefficient),
                            "abs_coefficient": float(abs(coefficient)),
                            "selected": bool(abs(coefficient) > 1e-8),
                        }
                    )
        if isinstance(regressor, FWLSLiteRegressor):
            coefficients = getattr(regressor.meta_model_, "coef_", None)
            if coefficients is not None:
                for coefficient_idx, coefficient in enumerate(np.ravel(np.asarray(coefficients, dtype=float))):
                    rows.append(
                        {
                            "model_name": model_name,
                            "member_name": f"fwls_term_{coefficient_idx}",
                            "coefficient": float(coefficient),
                            "abs_coefficient": float(abs(coefficient)),
                            "selected": bool(abs(coefficient) > 1e-8),
                        }
                    )
    pd.DataFrame(rows).to_csv(output_dir / "meta_learner_coefficients.csv", index=False, encoding="utf-8-sig")
    coefficient_df = pd.DataFrame(rows)
    if len(coefficient_df) > 0 and "model_name" in coefficient_df.columns:
        fwls_coefficient_df = coefficient_df.loc[coefficient_df["model_name"].eq("fwls_lite")]
    else:
        fwls_coefficient_df = pd.DataFrame([])
    fwls_coefficient_df.to_csv(output_dir / "fwls_interaction_coefficients.csv", index=False, encoding="utf-8-sig")


def _write_meta_learner_alpha_search(
    output_dir: Path,
    results: list[NextStageCandidateResult],
    pipelines: dict[str, Pipeline],
) -> None:
    rows: list[dict[str, JsonValue]] = []
    for result in results:
        pipeline = pipelines[result.model_name]
        regressor = pipeline.named_steps["regressor"]
        alpha_value: float | None = None
        l1_ratio_value: float | None = None
        if isinstance(regressor, StackingRegressor):
            final_estimator = getattr(regressor, "final_estimator_", None)
            raw_alpha = getattr(final_estimator, "alpha_", getattr(final_estimator, "alpha", None))
            raw_l1_ratio = getattr(final_estimator, "l1_ratio_", getattr(final_estimator, "l1_ratio", None))
            alpha_value = float(raw_alpha) if isinstance(raw_alpha, int | float | np.floating) else None
            l1_ratio_value = float(raw_l1_ratio) if isinstance(raw_l1_ratio, int | float | np.floating) else None
        if result.model_name.startswith("stacking_ridge_alpha_") or "cv" in result.model_name:
            rows.append(
                {
                    "model_name": result.model_name,
                    "mae": result.metric.mae,
                    "rmse": result.metric.rmse,
                    "cv_mae_mean": result.metric.cv_mae_mean,
                    "alpha": alpha_value,
                    "l1_ratio": l1_ratio_value,
                }
            )
    pd.DataFrame(rows).sort_values(by="mae").to_csv(
        output_dir / "meta_learner_alpha_search.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_base_prediction_diagnostics(output_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    model_zoo = _build_model_zoo(42)
    base_names = _available_stacking_member_names(model_zoo, ())
    x_train = train_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y_test = test_df[TARGET_COLUMN].to_numpy()
    prediction_columns: dict[str, np.ndarray] = {}
    error_columns: dict[str, np.ndarray] = {}
    for model_name in base_names:
        pipeline = _build_pipeline_for_features(cast(RegressorMixin, clone(model_zoo[model_name])), NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        prediction_columns[model_name] = predictions
        error_columns[model_name] = np.abs(y_test - predictions)
    prediction_df = pd.DataFrame(prediction_columns)
    pearson_corr = prediction_df.corr(method="pearson")
    spearman_corr = prediction_df.corr(method="spearman")
    pearson_corr.to_csv(output_dir / "base_prediction_correlation.csv", encoding="utf-8-sig")
    spearman_corr.to_csv(output_dir / "base_prediction_spearman_correlation.csv", encoding="utf-8-sig")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pearson_corr, annot=True, fmt=".3f", cmap="viridis")
    plt.tight_layout()
    plt.savefig(output_dir / "base_prediction_correlation_heatmap.png", dpi=180)
    plt.close()
    error_df = pd.DataFrame(error_columns)
    threshold_by_model = error_df.quantile(0.75)
    rows: list[dict[str, JsonValue]] = []
    for left_name in base_names:
        for right_name in base_names:
            left_bad = error_df[left_name] >= threshold_by_model[left_name]
            right_bad = error_df[right_name] >= threshold_by_model[right_name]
            rows.append(
                {
                    "left_model": left_name,
                    "right_model": right_name,
                    "joint_top_quartile_error_rate": float((left_bad & right_bad).mean()),
                    "left_mae": float(error_df[left_name].mean()),
                    "right_mae": float(error_df[right_name].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "base_model_error_overlap.csv", index=False, encoding="utf-8-sig")


def _write_fwls_artifacts(output_dir: Path, pipelines: dict[str, Pipeline], test_df: pd.DataFrame) -> None:
    fwls_pipeline = pipelines.get("fwls_lite")
    if fwls_pipeline is None:
        pd.DataFrame([]).to_csv(output_dir / "fwls_meta_features.csv", index=False, encoding="utf-8-sig")
        return
    x_test = test_df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    preprocessor = fwls_pipeline.named_steps["preprocessor"]
    regressor = fwls_pipeline.named_steps["regressor"]
    transformed_test = preprocessor.transform(x_test)
    if not isinstance(regressor, FWLSLiteRegressor):
        pd.DataFrame([]).to_csv(output_dir / "fwls_meta_features.csv", index=False, encoding="utf-8-sig")
        return
    base_predictions = np.column_stack([estimator.predict(transformed_test) for _, estimator in regressor.fitted_estimators_])
    rows = {
        "house_id": test_df["house_id"].astype(str).to_numpy(),
        "area_sqm": test_df["area_sqm"].to_numpy(dtype=float),
        "poi_balance_score": test_df["poi_balance_score"].to_numpy(dtype=float),
        "poi_subway_count": test_df["poi_subway_count"].to_numpy(dtype=float),
        "base_prediction_std": np.std(base_predictions, axis=1),
    }
    pd.DataFrame(rows).to_csv(output_dir / "fwls_meta_features.csv", index=False, encoding="utf-8-sig")


def _write_stacking_meta_summary(output_dir: Path, results: list[NextStageCandidateResult]) -> None:
    best_result = min(results, key=lambda item: item.metric.mae)
    reference_result = next(result for result in results if result.model_name == "stacking_ridge_default")
    meta_results = [result for result in results if result.combination_name in {"meta_learner", "meta_alpha_grid"}]
    diverse_results = [result for result in results if result.combination_name == "diverse_stacking"]
    fwls_results = [result for result in results if result.model_name == "fwls_lite"]
    best_meta = min(meta_results, key=lambda item: item.metric.mae)
    best_diverse = min(diverse_results, key=lambda item: item.metric.mae) if len(diverse_results) > 0 else None
    best_fwls = fwls_results[0] if len(fwls_results) > 0 else None
    summary: dict[str, JsonValue] = {
        "best_model": best_result.model_name,
        "best_model_mae": best_result.metric.mae,
        "reference_model": reference_result.model_name,
        "reference_mae": reference_result.metric.mae,
        "best_meta_model": best_meta.model_name,
        "best_meta_mae": best_meta.metric.mae,
        "best_diverse_model": best_diverse.model_name if best_diverse is not None else None,
        "best_diverse_mae": best_diverse.metric.mae if best_diverse is not None else None,
        "fwls_model": best_fwls.model_name if best_fwls is not None else None,
        "fwls_mae": best_fwls.metric.mae if best_fwls is not None else None,
        "mae_delta_vs_reference": best_result.metric.mae - reference_result.metric.mae,
        "mae_delta_vs_current_full_best": best_result.metric.mae - CURRENT_BEST_FULL_MAE,
        "update_current_best": best_result.metric.mae < CURRENT_BEST_FULL_MAE,
    }
    (output_dir / "stacking_meta_optimization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_summary(
    output_dir: Path,
    config: ResearchRunConfig,
    results: list[NextStageCandidateResult],
    experiment_mode: ExperimentMode,
) -> None:
    if experiment_mode == "algorithm_baseline":
        _write_algorithm_mode_summary(output_dir, config, results)
        return
    if experiment_mode == "stacking_meta_optimization":
        _write_stacking_meta_mode_summary(output_dir, config, results)
        return
    if experiment_mode == "advanced_algorithm_comparison":
        _write_advanced_algorithm_mode_summary(output_dir, config, results)
        return
    reference_result = next(result for result in results if result.model_name == "stacking_full")
    best_result = min(results, key=lambda item: item.metric.mae)
    stacking_results = [result for result in results if result.ensemble_method == "stacking_ridge"]
    voting_results = [result for result in results if result.ensemble_method == "voting"]
    best_stacking_result = min(stacking_results, key=lambda item: item.metric.mae)
    best_voting_result = min(voting_results, key=lambda item: item.metric.mae)
    available_members = reference_result.member_models.split("|")
    expected_members = ["random_forest", "gradient_boosting", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"]
    unavailable_members = [model_name for model_name in expected_members if model_name not in available_members]
    summary: dict[str, JsonValue] = {
        "run_id": config.run_id,
        "generated_at": datetime.now().isoformat(),
        "current_full_best_model": "stacking_ridge",
        "current_full_best_mae": CURRENT_BEST_FULL_MAE,
        "reference_model": reference_result.model_name,
        "reference_mae": reference_result.metric.mae,
        "best_model": best_result.model_name,
        "best_model_mae": best_result.metric.mae,
        "best_model_ensemble_method": best_result.ensemble_method,
        "best_model_member_models": best_result.member_models,
        "best_stacking_model": best_stacking_result.model_name,
        "best_stacking_mae": best_stacking_result.metric.mae,
        "best_stacking_member_models": best_stacking_result.member_models,
        "best_voting_model": best_voting_result.model_name,
        "best_voting_mae": best_voting_result.metric.mae,
        "best_voting_member_models": best_voting_result.member_models,
        "mae_delta_vs_reference": best_result.metric.mae - reference_result.metric.mae,
        "mae_delta_vs_current_full_best": best_result.metric.mae - CURRENT_BEST_FULL_MAE,
        "update_current_best": best_result.metric.mae < CURRENT_BEST_FULL_MAE,
        "available_members": available_members,
        "unavailable_members": unavailable_members,
        "temporal_generalization_note": "本轮不做时间泛化验证；当前缺少有效跨月份数据支撑。",
        "artifacts": {
            "model_metrics": str(output_dir / "model_metrics.csv"),
            "ensemble_member_ablation": str(output_dir / "ensemble_member_ablation.csv"),
            "voting_vs_stacking_metrics": str(output_dir / "voting_vs_stacking_metrics.csv"),
            "stacking_component_ablation": str(output_dir / "stacking_component_ablation.csv"),
            "stable_feature_group_ablation": str(output_dir / "stable_feature_group_ablation.csv"),
            "predictions": str(output_dir / "predictions.csv"),
            "model_prediction_errors": str(output_dir / "model_prediction_errors.csv"),
            "error_stratification": str(output_dir / "error_stratification.csv"),
            "feature_importance": str(output_dir / "feature_importance.csv"),
            "shap_importance": str(output_dir / "shap_importance.csv"),
        },
    }
    (output_dir / "next_stage_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_algorithm_baseline_summary(output_dir: Path, results: list[NextStageCandidateResult]) -> None:
    best_result = min(results, key=lambda item: item.metric.mae)
    reference_result = next(result for result in results if result.model_name == "stacking_full")
    summary: dict[str, JsonValue] = {
        "best_model": best_result.model_name,
        "best_model_mae": best_result.metric.mae,
        "reference_model": reference_result.model_name,
        "reference_mae": reference_result.metric.mae,
        "mae_delta_vs_reference": best_result.metric.mae - reference_result.metric.mae,
        "mae_delta_vs_current_full_best": best_result.metric.mae - CURRENT_BEST_FULL_MAE,
        "update_current_best": best_result.metric.mae < CURRENT_BEST_FULL_MAE,
        "candidate_models": [result.model_name for result in results],
    }
    (output_dir / "algorithm_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_algorithm_mode_summary(
    output_dir: Path,
    config: ResearchRunConfig,
    results: list[NextStageCandidateResult],
) -> None:
    best_result = min(results, key=lambda item: item.metric.mae)
    reference_result = next(result for result in results if result.model_name == "stacking_full")
    summary: dict[str, JsonValue] = {
        "run_id": config.run_id,
        "generated_at": datetime.now().isoformat(),
        "experiment_mode": "algorithm_baseline",
        "current_full_best_model": "stacking_ridge",
        "current_full_best_mae": CURRENT_BEST_FULL_MAE,
        "reference_model": reference_result.model_name,
        "reference_mae": reference_result.metric.mae,
        "best_model": best_result.model_name,
        "best_model_mae": best_result.metric.mae,
        "best_model_ensemble_method": best_result.ensemble_method,
        "mae_delta_vs_reference": best_result.metric.mae - reference_result.metric.mae,
        "mae_delta_vs_current_full_best": best_result.metric.mae - CURRENT_BEST_FULL_MAE,
        "update_current_best": best_result.metric.mae < CURRENT_BEST_FULL_MAE,
        "candidate_models": [result.model_name for result in results],
        "temporal_generalization_note": "本轮不做时间泛化验证；当前缺少有效跨月份数据支撑。",
        "artifacts": {
            "model_metrics": str(output_dir / "model_metrics.csv"),
            "algorithm_baseline_metrics": str(output_dir / "algorithm_baseline_metrics.csv"),
            "algorithm_baseline_summary": str(output_dir / "algorithm_baseline_summary.json"),
            "predictions": str(output_dir / "predictions.csv"),
            "model_prediction_errors": str(output_dir / "model_prediction_errors.csv"),
            "error_stratification": str(output_dir / "error_stratification.csv"),
            "feature_importance": str(output_dir / "feature_importance.csv"),
            "shap_importance": str(output_dir / "shap_importance.csv"),
        },
    }
    (output_dir / "next_stage_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_stacking_meta_mode_summary(
    output_dir: Path,
    config: ResearchRunConfig,
    results: list[NextStageCandidateResult],
) -> None:
    best_result = min(results, key=lambda item: item.metric.mae)
    reference_result = next(result for result in results if result.model_name == "stacking_ridge_default")
    summary_path = output_dir / "stacking_meta_optimization_summary.json"
    detail_summary: dict[str, JsonValue] = {}
    if summary_path.exists():
        parsed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(parsed_summary, dict):
            detail_summary = cast(dict[str, JsonValue], parsed_summary)
    summary: dict[str, JsonValue] = {
        "run_id": config.run_id,
        "generated_at": datetime.now().isoformat(),
        "experiment_mode": "stacking_meta_optimization",
        "current_full_best_model": "stacking_ridge",
        "current_full_best_mae": CURRENT_BEST_FULL_MAE,
        "reference_model": reference_result.model_name,
        "reference_mae": reference_result.metric.mae,
        "best_model": best_result.model_name,
        "best_model_mae": best_result.metric.mae,
        "best_model_ensemble_method": best_result.ensemble_method,
        "mae_delta_vs_reference": best_result.metric.mae - reference_result.metric.mae,
        "mae_delta_vs_current_full_best": best_result.metric.mae - CURRENT_BEST_FULL_MAE,
        "update_current_best": best_result.metric.mae < CURRENT_BEST_FULL_MAE,
        "detail_summary": detail_summary,
        "temporal_generalization_note": "本轮不做时间泛化验证；当前缺少有效跨月份数据支撑。",
        "artifacts": {
            "model_metrics": str(output_dir / "model_metrics.csv"),
            "stacking_meta_metrics": str(output_dir / "stacking_meta_metrics.csv"),
            "meta_learner_coefficients": str(output_dir / "meta_learner_coefficients.csv"),
            "meta_learner_alpha_search": str(output_dir / "meta_learner_alpha_search.csv"),
            "base_prediction_correlation": str(output_dir / "base_prediction_correlation.csv"),
            "base_model_error_overlap": str(output_dir / "base_model_error_overlap.csv"),
            "diverse_stacking_metrics": str(output_dir / "diverse_stacking_metrics.csv"),
            "fwls_metrics": str(output_dir / "fwls_metrics.csv"),
            "fwls_meta_features": str(output_dir / "fwls_meta_features.csv"),
            "stacking_meta_optimization_summary": str(output_dir / "stacking_meta_optimization_summary.json"),
            "predictions": str(output_dir / "predictions.csv"),
            "model_prediction_errors": str(output_dir / "model_prediction_errors.csv"),
            "error_stratification": str(output_dir / "error_stratification.csv"),
            "feature_importance": str(output_dir / "feature_importance.csv"),
            "shap_importance": str(output_dir / "shap_importance.csv"),
        },
    }
    (output_dir / "next_stage_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_advanced_algorithm_mode_summary(
    output_dir: Path,
    config: ResearchRunConfig,
    results: list[NextStageCandidateResult],
) -> None:
    best_result = min(results, key=lambda item: item.metric.mae)
    reference_result = next(result for result in results if result.model_name == "stacking_full")
    summary_path = output_dir / "advanced_algorithm_summary.json"
    detail_summary: dict[str, JsonValue] = {}
    if summary_path.exists():
        parsed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(parsed_summary, dict):
            detail_summary = cast(dict[str, JsonValue], parsed_summary)
    summary: dict[str, JsonValue] = {
        "run_id": config.run_id,
        "generated_at": datetime.now().isoformat(),
        "experiment_mode": "advanced_algorithm_comparison",
        "current_full_best_model": "stacking_ridge",
        "current_full_best_mae": CURRENT_BEST_FULL_MAE,
        "reference_model": reference_result.model_name,
        "reference_mae": reference_result.metric.mae,
        "best_model": best_result.model_name,
        "best_model_mae": best_result.metric.mae,
        "best_model_ensemble_method": best_result.ensemble_method,
        "mae_delta_vs_reference": best_result.metric.mae - reference_result.metric.mae,
        "mae_delta_vs_current_full_best": best_result.metric.mae - CURRENT_BEST_FULL_MAE,
        "update_current_best": best_result.metric.mae < CURRENT_BEST_FULL_MAE,
        "detail_summary": detail_summary,
        "temporal_generalization_note": "本轮不做时间泛化验证；当前缺少有效跨月份数据支撑。",
        "artifacts": {
            "model_metrics": str(output_dir / "model_metrics.csv"),
            "advanced_algorithm_metrics": str(output_dir / "advanced_algorithm_metrics.csv"),
            "advanced_algorithm_summary": str(output_dir / "advanced_algorithm_summary.json"),
            "super_learner_weights": str(output_dir / "super_learner_weights.csv"),
            "super_learner_oof_predictions": str(output_dir / "super_learner_oof_predictions.csv"),
            "moe_gate_profile": str(output_dir / "moe_gate_profile.csv"),
            "moe_expert_metrics": str(output_dir / "moe_expert_metrics.csv"),
            "moe_predictions": str(output_dir / "moe_predictions.csv"),
            "predictions": str(output_dir / "predictions.csv"),
            "model_prediction_errors": str(output_dir / "model_prediction_errors.csv"),
            "error_stratification": str(output_dir / "error_stratification.csv"),
            "feature_importance": str(output_dir / "feature_importance.csv"),
            "shap_importance": str(output_dir / "shap_importance.csv"),
        },
    }
    (output_dir / "next_stage_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_next_stage_experiment(output_dir: Path, record: dict[str, JsonValue]) -> None:
    log_record: dict[str, JsonValue] = {
        "created_at": datetime.now().isoformat(),
        **record,
    }
    with (output_dir / "model_experiments.jsonl").open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(log_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
