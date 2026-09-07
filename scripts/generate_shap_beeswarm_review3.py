"""Regenerate SHAP beeswarm with English labels and no on-figure title."""
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = __import__("pathlib").Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DESC_PATH = OUTPUT_DIR / "desc.txt"

# Load data：从原始采集文件重建特征并应用无泄漏参考中位数（与主实验同一代码路径）。
# 不读 cleaned_features.csv 快照：CSV 往返浮点噪声会被提升树混沌放大
# （实测同协议 MAE 21.9899 -> 22.2192）。
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
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
        _load_houses(_Path("data/full/nanchang_houses.jsonl")),
        _load_communities(_Path("data/full/nanchang_communities.jsonl")),
    )[0]
)
df = df.sort_values(by=["extracted_at", "house_id"], na_position="first").reset_index(drop=True)

test_count = max(int(len(df) * 0.20), 1)
train_df = df.iloc[:-test_count].copy()
test_df = df.iloc[-test_count:].copy()
train_df, test_df = _apply_reference_group_medians(
    train_df, test_df, _compute_region_ai_medians(_Path("data/full/nanchang_communities.jsonl"))
)

NUMERIC_FEATURES = [
    "area_sqm", "room_count", "hall_count", "bath_count", "total_floors",
    "floor_level_score", "floor_ratio_proxy", "area_per_room", "layout_density",
    "poi_bank_count", "poi_bus_count", "poi_subway_count", "poi_school_count",
    "poi_restaurant_count", "poi_shop_count", "poi_hospital_count", "poi_total_count",
    "accessibility_index", "poi_balance_score", "transit_medical_ratio",
    "education_commerce_ratio", "area_layout_efficiency", "room_area_interaction",
    "bath_room_ratio", "region_accessibility_interaction", "floor_area_interaction",
]
CATEGORICAL_FEATURES = [
    "region_slug", "decoration_type", "house_certificate_years",
    "orientation", "floor_level", "has_elevator", "is_unique_housing", "house_layout",
]

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df["total_price_wan"].values
X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ],
    sparse_threshold=0.0,
)

model = HistGradientBoostingRegressor(
    max_iter=240, learning_rate=0.055, l2_regularization=0.05, random_state=42
)
pipe = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
pipe.fit(X_train, y_train)

feature_names = list(preprocessor.get_feature_names_out())

# Translate Chinese categorical labels to English
def translate_feature_name(name: str) -> str:
    # Orientation mappings
    ori_map = {
        "南北": "N-S", "南": "S", "东西": "E-W", "西北": "NW", "西南": "SW",
        "西": "W", "东南": "SE", "北": "N", "东北": "NE", "东": "E",
    }
    # House certificate years
    cert_map = {
        "满二": "Over2Yrs", "满五年": "Over5Yrs", "满五": "Over5Yrs",
        "满二年": "Over2Yrs", "满2年": "Over2Yrs", "满20": "Over20Yrs",
        "满24": "Over24Yrs",
    }
    # Decoration type
    decor_map = {
        "毛坯": "Rough", "豪华装修": "Luxury", "简单装修": "Simple",
        "精装修": "FineDecor",
    }
    # Floor level
    floor_map = {"中层": "Mid", "高层": "High", "低层": "Low"}
    # Boolean-like
    bool_map = {"True": "Yes", "False": "No"}

    # Apply translations in order
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
    # Clean up house layout: e.g. "3室2厅1卫" -> "3R2H1B"
    import re
    def repl_layout(m):
        return f"{m.group(1)}R{m.group(2)}H{m.group(3)}B"
    name = re.sub(r"(\d+)室(\d+)厅(\d+)卫", repl_layout, name)
    return name

feature_names = [translate_feature_name(n) for n in feature_names]
# Strip numeric__ / categorical__ prefixes for cleaner display
feature_names = [n.replace("numeric__", "").replace("categorical__", "") for n in feature_names]

X_test_proc = preprocessor.transform(X_test)
sample_size = min(300, len(X_test_proc))
X_sample = X_test_proc[:sample_size]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(
    shap_values, X_sample, feature_names=feature_names,
    show=False, plot_size=None, max_display=20,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig5b_shap_beeswarm.svg", format="svg", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "fig5b_shap_beeswarm.png", format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Append description
desc_line = "fig5b_shap_beeswarm: SHAP Beeswarm Plot of HistGradientBoosting Base Learner (Impact on Model Output)\n"
with open(DESC_PATH, "a", encoding="utf-8") as f:
    f.write(desc_line)

print("SHAP beeswarm regenerated in docs/figures/")
