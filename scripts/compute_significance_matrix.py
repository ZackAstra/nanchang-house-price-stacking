# -*- coding: utf-8 -*-
"""评审修复任务1：6个头部模型的全配对显著性矩阵（配对t检验 + Wilcoxon + Holm校正）。

输入:  data/expected_outputs/model_prediction_errors.csv
       （逐样本绝对误差文件，来自主实验完整输出；本仓库未随附，需完整数据后获取）
输出:  data/expected_outputs/significance_matrix.csv
       data/expected_outputs/significance_matrix.md
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "data", "expected_outputs", "model_prediction_errors.csv")
OUT_DIR = os.path.join(ROOT, "data", "expected_outputs")

TOP_MODELS = [
    "stacking_ridge",
    "two_stage_residual",
    "catboost",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
]

# 参考基准（类别编码版旧产物 review_full_final_202605031215/significance_tests.csv）。
# 注意：数值化特征（review3）下预测值已改变，精确复现不可能；
# sanity 口径调整为：两检验均高度显著（p < 1e-3）且 ΔMAE 方向为负（stacking 更优）。
SANITY_PAIR = ("stacking_ridge", "two_stage_residual")
SANITY_T_P = 2.572118725903821e-07
SANITY_W_P = 2.6547626265382807e-06


def holm_adjust(p_values):
    """手写 Holm-Bonferroni 逐步递减校正，保持输入顺序返回校正后 p 值。"""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * p[idx], 1.0)
        running_max = max(running_max, val)  # 保证单调性
        adjusted[idx] = running_max
    return adjusted


def fmt_p(p):
    """论文风格 p 值格式化。"""
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    models_in_data = sorted(df["model_name"].unique())
    print(f"数据中模型: {models_in_data}")

    # 按 house_id 对齐逐样本绝对误差，构建宽表
    wide = {}
    for m in TOP_MODELS:
        sub = df[df["model_name"] == m].set_index("house_id")["absolute_error_wan"]
        assert len(sub) == 1383, f"{m} 样本数 {len(sub)} != 1383"
        wide[m] = sub
    err = pd.DataFrame(wide).dropna()
    print(f"对齐后共同测试样本数: {len(err)}")

    rows = []
    for a, b in itertools.combinations(TOP_MODELS, 2):
        diff = err[a] - err[b]
        t_stat, t_p = stats.ttest_rel(err[a], err[b])
        w_stat, w_p = stats.wilcoxon(diff)
        rows.append(
            {
                "comparison": f"{a} vs {b}",
                "mae_diff": float(err[a].mean() - err[b].mean()),
                "t_p": float(t_p),
                "wilcoxon_p": float(w_p),
            }
        )

    res = pd.DataFrame(rows)
    res["t_p_holm"] = holm_adjust(res["t_p"].values)
    res["wilcoxon_p_holm"] = holm_adjust(res["wilcoxon_p"].values)
    res["significant_0.05"] = (res["t_p_holm"] < 0.05) & (res["wilcoxon_p_holm"] < 0.05)
    res.to_csv(os.path.join(OUT_DIR, "significance_matrix.csv"), index=False, encoding="utf-8-sig")

    # markdown 表（论文可用）
    lines = [
        "# 头部模型逐样本绝对误差配对显著性矩阵（n = %d）" % len(err),
        "",
        "配对检验基于 1383 个测试样本的逐样本绝对误差；mae_diff = 前者 MAE − 后者 MAE（负值表示前者更优）。",
        "p 值经 Holm-Bonferroni 多重比较校正（15 组配对）。",
        "",
        "| Comparison | ΔMAE (万元) | Paired t p | Wilcoxon p | t p (Holm) | Wilcoxon p (Holm) | Sig. @0.05 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"| {r['comparison']} | {r['mae_diff']:+.4f} | {fmt_p(r['t_p'])} | {fmt_p(r['wilcoxon_p'])} "
            f"| {fmt_p(r['t_p_holm'])} | {fmt_p(r['wilcoxon_p_holm'])} | {'Yes' if r['significant_0.05'] else 'No'} |"
        )
    lines.append("")
    with open(os.path.join(OUT_DIR, "significance_matrix.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # sanity check：stacking_ridge vs two_stage_residual 应保持高度显著且方向为负
    row = res[res["comparison"] == "stacking_ridge vs two_stage_residual"].iloc[0]
    t_ok = row["t_p"] < 1e-3
    w_ok = row["wilcoxon_p"] < 1e-3
    d_ok = row["mae_diff"] < 0
    print("\n=== Sanity check: stacking_ridge vs two_stage_residual ===")
    print(f"  本次 t_p={row['t_p']:.4e} (参考旧值 {SANITY_T_P:.4e}) -> {'PASS' if t_ok else 'FAIL'}")
    print(f"  本次 wilcoxon_p={row['wilcoxon_p']:.4e} (参考旧值 {SANITY_W_P:.4e}) -> {'PASS' if w_ok else 'FAIL'}")
    print(f"  ΔMAE={row['mae_diff']:+.4f} (应<0) -> {'PASS' if d_ok else 'FAIL'}")

    key = res[res["comparison"] == "stacking_ridge vs catboost"].iloc[0]
    print("\n=== 评审点名: StackingRidge vs CatBoost ===")
    print(
        f"  ΔMAE={key['mae_diff']:+.4f}, t_p={key['t_p']:.4e} (Holm {key['t_p_holm']:.4e}), "
        f"wilcoxon_p={key['wilcoxon_p']:.4e} (Holm {key['wilcoxon_p_holm']:.4e}), "
        f"significant={bool(key['significant_0.05'])}"
    )

    n_sig = int(res["significant_0.05"].sum())
    print(f"\n校正后仍显著 (两检验 Holm p 均 < 0.05): {n_sig}/15；不显著: {15 - n_sig}/15")
    print(f"\n输出: {OUT_DIR}/significance_matrix.csv, significance_matrix.md")


if __name__ == "__main__":
    main()
