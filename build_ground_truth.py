"""
Milestone 1 — Data Integration & Ground Truth Builder
Run: python build_ground_truth.py
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

FILES = {
    "reviewer_seed1": ("csv", "discharge-adrd-seed1-50-25-zid4001.csv"),
    "reviewer_seed2": ("excel", "discharge-adrd-seed2-50-25.csv"),
    "reviewer_heg": ("excel", "discharge-adrd-seed7-50-25-heg4007-1.csv"),
    "reviewer_jim": ("csv", "discharge-adrd-seed7-50-25-jim4007.csv"),
}

LABEL_COL = (
    "is AD/ADRD? (type in 1,0, -1) 1 for yes ADRD Present, "
    "0 for No ADRD, -1 uncertain)"
)
DX_FINAL_COL = "adrd_dx(final)"
ICD_COL = "adrd_dx(icd_code)"   # ← 新增：用这列填充 jim 和 dx_only_label

# ── Step 1: Create directories ───────────────────────────────────────────────
for d in ["data", "data/patient_notes", "outputs"]:
    os.makedirs(ROOT / d, exist_ok=True)
print("[Step 1] Directories ready.")

# ── Load all files ───────────────────────────────────────────────────────────
raw_frames = {}
row_counts = {}

for reviewer, (fmt, fname) in FILES.items():
    path = ROOT / fname
    if fmt == "csv":
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    else:
        df = pd.read_excel(path)

    # ── 新增：jim 标注列为空 → 用 adrd_dx(icd_code) 填充 ──────────────────
    if reviewer == "reviewer_jim":
        empty_before = df[LABEL_COL].isna().sum()
        df[LABEL_COL] = df[LABEL_COL].fillna(df[ICD_COL])
        filled = empty_before - df[LABEL_COL].isna().sum()
        print(f"  [jim] 空标注用 adrd_dx(icd_code) 填充了 {filled} 条")

    df["source_file"] = fname
    df["reviewer"] = reviewer
    raw_frames[reviewer] = df
    row_counts[fname] = len(df)
    print(f"  Loaded {fname}: {len(df)} rows")

all_df = pd.concat(raw_frames.values(), ignore_index=True)

# ── Normalise key columns ─────────────────────────────────────────────────────
all_df["patient_id"] = all_df["subject_id"].astype(str).str.strip()
all_df["note_id"] = all_df["note_id"].astype(str).str.strip()
all_df["hadm_id"] = all_df["hadm_id"].astype(str).str.strip()

# Derive has_notes / has_dx
all_df["has_notes"] = all_df["text"].notna() & (
    all_df["text"].astype(str).str.strip() != ""
)
all_df["has_dx"] = all_df["all_icd_codes"].notna() & (
    all_df["all_icd_codes"].astype(str).str.strip().isin(["", "nan"]) == False
)


# y_true: 1/0/-1; NaN if column was empty
def parse_label(val):
    try:
        v = float(val)
        if v in (1.0, 0.0, -1.0):
            return int(v)
        return None
    except (ValueError, TypeError):
        return None


all_df["y_true"] = all_df[LABEL_COL].apply(parse_label)

# ── 新增：dx_only_label 用 adrd_dx(icd_code) 填充（ICD-only baseline）──────
all_df["dx_only_label"] = all_df[ICD_COL].apply(parse_label)

# ── 新增：Cohen's Kappa（heg vs jim，75 个重叠患者）──────────────────────────
print("\n[Kappa] 计算 heg vs jim 标注一致性...")
heg_df = raw_frames["reviewer_heg"].copy()
jim_df = raw_frames["reviewer_jim"].copy()
heg_df["pid"] = heg_df["subject_id"].astype(str).str.strip()
jim_df["pid"] = jim_df["subject_id"].astype(str).str.strip()
heg_labels = heg_df.set_index("pid")[LABEL_COL].apply(parse_label).dropna()
jim_labels = jim_df.set_index("pid")[LABEL_COL].apply(parse_label).dropna()
common = heg_labels.index.intersection(jim_labels.index)
h = heg_labels[common]
j = jim_labels[common]
# 只用 0/1，排除 -1 uncertain
mask = h.isin([0, 1]) & j.isin([0, 1])
if mask.sum() >= 2:
    kappa = cohen_kappa_score(h[mask].astype(int), j[mask].astype(int))
    level = "Excellent ✓" if kappa >= 0.8 else ("Good" if kappa >= 0.6 else "Poor — revisit rules")
    print(f"  重叠患者: {len(common)}, 有效(0/1): {mask.sum()}")
    print(f"  Cohen's Kappa = {kappa:.3f}  >> {level}")
else:
    print("  有效重叠不足，跳过 Kappa")

# ── Step 2: patient_index.csv（含 train/test 划分）──────────────────────────
# 只对 y_true=0/1 的做划分，-1 标为 uncertain
labeled = all_df[all_df["y_true"].isin([0, 1])].drop_duplicates("patient_id")
pos = labeled[labeled["y_true"] == 1].sample(frac=1, random_state=42)
neg = labeled[labeled["y_true"] == 0].sample(frac=1, random_state=42)

# 60% train / 40% test，保持正负比例
def split_ids(df, train_frac=0.6):
    n = max(1, round(len(df) * train_frac))
    return set(df.iloc[:n]["patient_id"]), set(df.iloc[n:]["patient_id"])

pos_train, pos_test = split_ids(pos)
neg_train, neg_test = split_ids(neg)
train_ids = pos_train | neg_train
test_ids  = pos_test  | neg_test

all_df["split"] = "uncertain"
all_df.loc[all_df["patient_id"].isin(train_ids), "split"] = "train"
all_df.loc[all_df["patient_id"].isin(test_ids),  "split"] = "test"

patient_index = all_df[
    ["patient_id", "note_id", "hadm_id", "has_notes", "has_dx", "source_file", "split"]
].copy()

out_path = ROOT / "data" / "patient_index.csv"
patient_index.to_csv(out_path, index=False)
print(f"\n[Step 2] patient_index.csv -> {out_path}  ({len(patient_index)} rows)")
sc = all_df["split"].value_counts()
print(f"  train={sc.get('train',0)}, test={sc.get('test',0)}, uncertain={sc.get('uncertain',0)}")

# ── Step 3: patient_notes.csv ────────────────────────────────────────────────
notes_df = all_df[all_df["has_notes"]].copy()
skipped = len(all_df) - len(notes_df)

patient_notes = notes_df[
    ["patient_id", "note_id", "note_type", "charttime", "text"]
].rename(columns={"text": "note_text"})

out_path = ROOT / "data" / "patient_notes" / "patient_notes.csv"
patient_notes.to_csv(out_path, index=False)
print(
    f"[Step 3] patient_notes.csv -> {out_path}  ({len(patient_notes)} rows, {skipped} skipped)"
)

# ── Step 4: ground_truth.csv ─────────────────────────────────────────────────
ground_truth = all_df[
    ["patient_id", "note_id", "y_true", "dx_only_label", "source_file", "split"]
].copy()

out_path = ROOT / "outputs" / "ground_truth.csv"
ground_truth.to_csv(out_path, index=False)
print(f"[Step 4] ground_truth.csv  -> {out_path}  ({len(ground_truth)} rows)")

# ── Step 5: review_log.csv ───────────────────────────────────────────────────
review_log = (
    all_df[["patient_id", "note_id", "reviewer", "y_true", "source_file"]]
    .rename(columns={"y_true": "decision"})
    .copy()
)
review_log.insert(3, "minutes_spent", None)

out_path = ROOT / "outputs" / "review_log.csv"
review_log.to_csv(out_path, index=False)
print(f"[Step 5] review_log.csv    -> {out_path}  ({len(review_log)} rows)")

# ── Step 6: dx_only_baseline.csv（ICD-only 预测，直接用于评估）──────────────
dx_base = all_df[all_df["dx_only_label"].notna()][
    ["patient_id", "note_id", "dx_only_label", "split"]
].copy()
dx_base = dx_base.rename(columns={"dx_only_label": "label"})
dx_base["probability"] = dx_base["label"].astype(float)
out_path = ROOT / "outputs" / "dx_only_baseline.csv"
dx_base.to_csv(out_path, index=False)
print(f"[Step 6] dx_only_baseline.csv -> {out_path}  ({len(dx_base)} rows)")

# ── Step 7: Quality report ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DATA QUALITY REPORT")
print("=" * 55)

print("\n[各文件行数]")
for fname, n in row_counts.items():
    print(f"  {fname}: {n}")

unique_patients = all_df["patient_id"].nunique()
print(f"\n[合并后 unique subject_id 数]: {unique_patients}")

print("\n[y_true 分布]")
vc = all_df["y_true"].value_counts(dropna=False)
for val, cnt in vc.items():
    label = {1: "1 (AD/ADRD)", 0: "0 (No ADRD)", -1: "-1 (uncertain)"}.get(val, "NaN/empty")
    print(f"  {label}: {cnt}")

print("\n[dx_only_label 分布]")
print(all_df["dx_only_label"].value_counts(dropna=False).to_string())

no_notes = (all_df["has_notes"] == False).sum()
print(f"\n[has_notes=False 的行数]: {no_notes}")

overlap = (
    all_df.groupby("patient_id")["reviewer"]
    .nunique()
    .reset_index()
    .rename(columns={"reviewer": "reviewer_count"})
)
multi = overlap[overlap["reviewer_count"] > 1]
print(f"\n[被多个标注者标注的 subject_id 数 (Kappa 候选)]: {len(multi)}")

print("\nDone.")
