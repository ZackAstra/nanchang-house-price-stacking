# -*- coding: utf-8 -*-
"""语义实验关键对比的配对显著性检验（review5 固定测试集协议）。

输入：data/runs/semantic_deepseek/predictions/predictions_{exp}__stacking_ridge.csv
      （由 llm_semantic_vs_rule_runner 落盘的逐样本预测，随仓库发布）
输出：data/expected_outputs/semantic_pair_tests_review5.csv

检验对（|误差| 配对 t 检验 + Wilcoxon 符号秩检验，Δ = MAE(b) − MAE(a)）：
    ① log_original vs log_rule_semantic        log 基线上规则语义的增益
    ② log_original vs log_llm_semantic         log 基线上 LLM 语义的增益
    ③ original     vs llm_semantic             原基线上 LLM 语义的代价
    ④ log_original vs log_rule_llm_semantic    规则+LLM 组合语义的增益
    ⑤ original     vs log_original             log 目标变换本身的影响

用法：uv run python scripts/compute_semantic_pair_tests.py
"""
import os

import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(ROOT, "data", "runs", "semantic_deepseek", "predictions")
OUT_PATH = os.path.join(ROOT, "data", "expected_outputs", "semantic_pair_tests_review5.csv")

PAIRS = [
    ("log_original", "log_rule_semantic"),
    ("log_original", "log_llm_semantic"),
    ("original", "llm_semantic"),
    ("log_original", "log_rule_llm_semantic"),
    ("original", "log_original"),
]


def load_errors(exp_name: str) -> pd.Series:
    path = os.path.join(PRED_DIR, "predictions_" + exp_name + "__stacking_ridge.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.set_index("house_id")["absolute_error_wan"]


def main() -> int:
    rows = []
    for a, b in PAIRS:
        ea, eb = load_errors(a), load_errors(b)
        joint = pd.concat([ea.rename("a"), eb.rename("b")], axis=1).dropna()
        diff = joint["a"] - joint["b"]
        t_stat, t_p = stats.ttest_rel(joint["a"], joint["b"])
        try:
            w_stat, w_p = stats.wilcoxon(diff[diff != 0])
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        rows.append({
            "comparison": a + " vs " + b,
            "n_pairs": len(joint),
            "mae_a": float(joint["a"].mean()),
            "mae_b": float(joint["b"].mean()),
            "mae_diff_a_minus_b": float(joint["a"].mean() - joint["b"].mean()),
            "t_stat": float(t_stat),
            "t_p": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
        })
        print(a, "vs", b, "delta=%.6f" % (joint["b"].mean() - joint["a"].mean()), "t_p=%.4e" % t_p)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("written:", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
