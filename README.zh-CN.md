# 南昌二手房价格预测：特征工程与 Stacking 集成

本仓库是论文《Comparative Study of Second-Hand Housing Price Prediction Based on Machine Learning Regression Algorithms: Feature Engineering and Stacking Ensemble Modeling》的配套代码，并随附论文所报数字对应的聚合结果文件。

[English version: README.md](README.md)

## 项目简介

本仓库复现论文的完整实验链路：

- **数据采集**：面向主流房产挂牌平台（安居客·南昌）的四级穿透爬虫——房源链接 → 房源详情 → 小区 POI 配套（`main.py`、`src/fetchers/`、`src/parsers/`、`src/storage/`、`config/regions.json`）；
- **特征工程**：四维特征体系——物理属性、POI 设施、地理可达性（Accessibility Index）、户型结构（Layout Density），实现于 `src/research/pipeline.py`；另有一套 29 标签的标题语义体系：规则词典标注见 `src/research/llm_semantic_vs_rule_runner.py`；论文报告的全量 LLM 标注（即随附标签 CSV 与 `llm_full_parse_report.json` 背后的 DeepSeek-V4-Flash 标注）由 SiliconFlow 批量管线产出——输入构建见 `scripts/build_llm_batch_jsonl.py`、批量输出解析见 `scripts/parse_deepseek_full_labels.py`；`src/research/llm_semantic_extractor.py` 则是用于跨模型一致性对照的 Qwen3-32B 提取路径；
- **模型对比**：8 个基线回归模型、Ridge 元学习器的 Stacking 集成、OOF-TSRM 两阶段残差模型；固定时间敏感留出测试集 + 社区分组 3 折交叉验证（`src/research/run_price_research.py`）；
- **误差诊断实验**（论文 §5.3）：log 变换、标签清洗剂量-效应、多项式交互、分位数回归、信息上限诊断（`src/research/*_runner.py`）；
- **语义特征实验**（论文 §5.3）：规则标注 vs LLM 标注，含 prompt×温度稳健性对照（`scripts/build_llm_batch_jsonl.py`、`scripts/run_qwen_prompt_bridge.py`、`scripts/evaluate_llm_3x2_batch.py`）；
- **分析脚本**：Holm 校正显著性矩阵、多种子汇总、算力成本、描述统计、方差分解、构造指标敏感性（`scripts/`）；
- **图表**：论文全部图由 `scripts/generate_figures_review3.py` 与 `scripts/generate_shap_beeswarm_review3.py` 重绘至 `docs/figures/`，图注写入 `docs/figures/desc.txt`。

## 主要结果

- 最终模型（log 目标 + 规则语义特征 + StackingRidge）：外推留出测试集上 **MAE = 20.17（万元）、R² = 0.3214**；随机同分布划分下 R² = 0.68；
- StackingRidge 在 5 个随机种子中 4 次第一，显著优于最强单模型 CatBoost（Holm 校正 p = 4.1 × 10⁻⁷）。

论文中每一个头条数字都有随附产物支撑，见下文《数字与产物对照表》。

## 项目结构

```
├── main.py / config/            # 爬虫入口与区域配置
├── src/
│   ├── fetchers/ parsers/ storage/ models/   # 爬虫组件
│   └── research/                # 特征工程、模型、实验 runner
├── scripts/                     # 分析、语义实验与绘图脚本
├── tests/                       # 爬虫集成测试（需联网 + 人工登录）
├── data/
│   ├── samples/                 # 200 条分层抽样示例（房源/社区/LLM 标注/LLM 原始响应）
│   ├── expected_outputs/        # 聚合指标与逐样本中间产物（复现核对基准）
│   ├── reference/               # 区域 AI 中位数（聚合统计）
│   ├── runs/                    # 主实验产物：原 Table 18 feature_importance.csv、§5.5 significance_tests.csv、§5.6 error_stratification.csv，以及 Table 7 / §5.3 语义产物（semantic_deepseek 的 experiment_summary 与 predictions、各 seed 的 log-rule 与 log-area 预测）；原 Table 19 与 §5.2 的产物在 expected_outputs/ 下
│   └── full/                    # 完整数据放置处（见 data/full/README.md）
├── llm_prompts/                 # 3 个 LLM 标注 prompt 全文（baseline / few-shot / strict）
└── docs/figures/                # 论文图（PNG + SVG）与 desc.txt
```

## 环境准备（uv）

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
# 仅爬虫需要：uv run playwright install chromium
```

头条结果在 Python 3.11.10 + `requirements.txt` 锁定版本下产出（与 `pyproject.toml` 一致）。请显式指定解释器版本：裸 `uv venv` 会解析到本机最新的 Python 3.x，那**不是**产出论文数字的环境。

## 快速开始（基于随附示例数据的冒烟运行）

```bash
uv run python -m src.research.run_price_research \
  --houses data/samples/houses_sample.jsonl \
  --communities data/samples/communities_sample.jsonl \
  --output-dir data/runs --run-id smoke --sample-limit 200 --cv-folds 3
```

该运行**仅用于验证流程可运行**。在 200 条示例上，流水线保留 195 条可用样本、MAE 约为 19；哪个模型胜出以及具体 MAE 均取决于本机 BLAS/线程配置。这两个数字都不出现在论文中，冒烟结果与论文任何结果都不可比。复现论文数字请先获取完整数据（见下文），放入 `data/full/` 后按下节操作。

## 数据获取

与论文 Data Availability 声明一致：

- **示例数据**：`data/samples/` 内含 200 条分层抽样（区域×价格段），含 LLM 语义标注与 10 条 LLM 原始响应示例；
- **聚合结果**：论文报告的每一张表数值与各小节统计量都由 `data/expected_outputs/` 或 `data/runs/` 下的随附文件支撑，无需原始爬取数据即可核验。唯一例外是 §5.2 引用的基学习器预测相关系数（GradientBoosting–XGBoost 为 0.9631）：它没有单独持久化为产物，但可由随附的 `model_prediction_errors.csv` 精确复现（0.963088）；
- **完整数据**（7,140 条房源 + 1,062 个社区 + 全量 LLM 语义标注）：请联系作者（zhaoxingchen@hotmail.com）合理获取，需遵守原平台服务条款。数据采集自公开挂牌页面，仅供学术研究使用。

## 复现论文各表格

论文每一张表的聚合数字已随附在 `data/expected_outputs/` 与 `data/runs/`。如需从完整数据（放入 `data/full/` 后）自行重算：

1. 按论文的 CV 协议（社区分组 3 折）运行主实验。运行 ID 请用 `main_seed42`——绘图脚本读取 `data/runs/main_seed42`：

   ```bash
   uv run python -m src.research.run_price_research \
     --houses data/full/nanchang_houses.jsonl \
     --communities data/full/nanchang_communities.jsonl \
     --output-dir data/runs --run-id main_seed42 --cv-folds 3
   ```

2. 将分析脚本读取的三个逐样本中间产物复制到 `data/expected_outputs/`：

   ```bash
   cp data/runs/main_seed42/cleaned_features.csv data/expected_outputs/
   cp data/runs/main_seed42/model_prediction_errors.csv data/expected_outputs/
   cp data/runs/main_seed42/model_experiments.jsonl data/expected_outputs/
   ```

3. 运行分析脚本（每个脚本头部均注明输入与输出），例如 `uv run python scripts/descriptive_stats.py`。

### Table 7（语义特征）

头条语义行由专用 runner 产出，而非主流水线。备好完整数据与 LLM 标签 CSV 后：

```bash
uv run python -m src.research.llm_semantic_vs_rule_runner \
  --houses data/full/nanchang_houses.jsonl \
  --communities data/full/nanchang_communities.jsonl \
  --llm-features data/full/llm_semantic_features_deepseek_full.csv \
  --output-dir data/runs --run-id semantic_deepseek --cv-folds 3 \
  --only log_rule_semantic
```

这将复现头条行（log 目标 + 规则语义特征 + StackingRidge：**MAE = 20.1740、R² = 0.3214**）。只有在选择 `llm_*` 实验组时才需要 `--llm-features`——纯规则复现（`--only log_rule_semantic`、`log_area_rule_semantic` 等）无需该文件，这也是 LLM 标签 CSV 未随附的原因。去掉 `--only` 会重算整个语义对照（约 6 个实验组，每组数分钟）。

随附的 `data/expected_outputs/semantic_logarea_summary.csv` 与 `data/expected_outputs/semantic_pair_tests_review5.csv` 由 `scripts/summarize_semantic_logarea.py` 与 `scripts/compute_semantic_pair_tests.py` 从随附的逐实验预测文件重算得到。

### 数字与产物对照表

| 论文表格 / 小节 | 对应产物 |
|---|---|
| Table 2 描述性统计 | `descriptive_stats_numeric.csv`、`descriptive_stats_categorical.csv` |
| Table 4 特征集消融 | `feature_set_ablation_metrics.csv` |
| Table 5 模型对比 | `model_metrics.csv` |
| 仅仓库（下沉）：Voting vs Stacking | `voting_vs_stacking_metrics.csv` |
| 仅仓库（下沉）：多种子稳定性 | `multi_seed_summary.csv`、`review5_seed_{7,21,84,168}/model_metrics.csv` |
| Table 6 误差诊断 | `combined_experiment_summary.csv`、`combined_model_metrics.csv`、`combined_stacking_comparison.csv`、`label_noise_summary.csv`、`cleaning_strategy_summary.csv`、`poly_experiment_summary.csv`、`quantile_model_metrics.csv`、`log_transform_summary.csv` |
| Table 7 语义特征 | `stacking_comparison.csv`（语义各行）、`semantic_feature_comparison.csv`、`semantic_logarea_summary.csv`、`semantic_pair_tests_review5.csv` |
| 仅仓库（下沉）：置换重要性 | `data/runs/main_seed42/feature_importance.csv`（图：`fig5a_permutation_importance`） |
| 仅仓库（下沉）：显著性矩阵 | `significance_matrix.csv` |
| 仅仓库（下沉）：随机划分性能 | `random_split_model_metrics.csv` |
| Table 8 算力成本 | `compute_cost.csv` |
| 仅仓库（下沉）：指标敏感性 | `sensitivity_results.csv` |
| §2.2 预处理计数 | `data_audit.json` |
| §5.3 方差分解 | `variance_decomposition.csv` |
| §5.3 语义一致性 | `semantic_agreement.csv`、`llm_prompt_temperature_eval.csv`、`llm_label_rate_corr.csv`、`qwen_bridge_summary.csv`、`llm_full_parse_report.json` |
| §5.2 元学习器系数 | `stacking_meta_coefficients.csv`（由 `scripts/generate_figures_review3.py` 写出） |
| §5.3 信息上限 | `ceiling_report.json`、`feature_marginal_analysis.csv` |
| §5.5 显著性检验 | `data/runs/main_seed42/significance_tests.csv` |
| §5.6 分层误差 | `data/runs/main_seed42/error_stratification.csv`（图：`fig6_stratified_errors`） |
| 补充材料：特征定义 | `docs/supplementary/table3_feature_definitions.md` |
| 补充材料：超参数设置 | `docs/supplementary/table6_hyperparameter_settings.md` |
| 补充材料：算法伪代码 | `docs/supplementary/table7_algorithm_pseudocode.md` |
| 补充材料：评估指标适用场景 | `docs/supplementary/evaluation_metrics_applicability.md` |

说明：

- `stacking_comparison.csv` 存放的是**语义特征实验（Table 7）**，而非 Voting-vs-Stacking 对比；后者在 `voting_vs_stacking_metrics.csv`。
- `combined_experiment_summary.csv` 报告的是**每种配置的最优模型**（对 log 目标 + 严格清洗那次运行即 LightGBM，MAE 20.4331）。Table 6 则是固定协议表，引用同一次运行中 StackingRidge 成员的值（`combined_stacking_ridge`，MAE 20.5346）。两个数字都真实、且来自同一次运行，差别只在于表中报告的是哪一个成员；用 `combined_model_metrics.csv` 可并列查看全部成员。
- Table 6 的其余行（moderate / 单价 / 总价清洗剂量、多项式交互、分位数回归）可用对应 runner 重算（`label_noise_experiment_runner.py`、`log_transform_experiment_runner.py`、`polynomial_feature_runner.py`、`quantile_regression_runner.py`），其汇总文件均已随附，见上表。
- 换机重算**不是**逐位可复现的。`RandomForest`/`XGBoost`/`LightGBM` 以 `n_jobs=-1` 运行，会扰动末几位有效数字（实测约 1e-14），因此 `significance_matrix.csv` 的末位也会跟着变；请用容差比较（对 MAE 而言 1e-6 足够宽裕），不要用字节相等。`compute_cost.csv` 还记录墙钟耗时（Table 8）：其 MAE 列可精确复现，但秒数/倍数列依赖硬件，因此应把报告的秒数视为参考、而把成本**倍数**视为稳定量。Matplotlib 生成的 SVG 内嵌创建时间戳与随机 `clip-path` id，因此只有 PNG 在多次运行间字节稳定；SVG 等价但非逐字节相等。

## 测试

`tests/test_m2_full.py` 是**实网爬虫集成测试**：需要网络、真实浏览器会话与人工登录，还依赖 Playwright 的 Chromium，并会写入 `data/` 与 `logs/`。它在全新克隆中无法通过，必须排除在 CI 之外。研究流水线本身没有单元测试——随附的 `data/expected_outputs/` 与 `data/runs/` 产物才是核验基准。

## LLM 标注说明

语义标签由大模型从房源标题提取。三种 prompt 设计以 `prompt_baseline.md`、`prompt_fewshot.md`、`prompt_strict.md` 全文公开于 `llm_prompts/`（依次为 baseline、few-shot、structured-strict）；标注使用 SiliconFlow 异步批量推理（Batch API）。标注示例与原始响应示例见 `data/samples/`。

## 许可与声明

代码以 MIT 许可证发布（见 `LICENSE`）。数据来自公开挂牌平台，仅供学术研究，须遵守原平台服务条款。
