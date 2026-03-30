"""
Milestone 1 — Data Integration & Ground Truth Builder
Run: python build_ground_truth.py
"""

import os
import pandas as pd
from pathlib import Path

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

# dx_only_label from adrd_dx(final) — currently empty for all files, keep as-is
all_df["dx_only_label"] = all_df[DX_FINAL_COL].apply(parse_label)

# ── Step 2: patient_index.csv ────────────────────────────────────────────────
patient_index = all_df[
    [
        "patient_id",
        "note_id",
        "hadm_id",
        "has_notes",
        "has_dx",
        "source_file",
    ]
].copy()
patient_index["split"] = "unassigned"

out_path = ROOT / "data" / "patient_index.csv"
patient_index.to_csv(out_path, index=False)
print(f"\n[Step 2] patient_index.csv -> {out_path}  ({len(patient_index)} rows)")

# ── Step 3: patient_notes.csv ────────────────────────────────────────────────
notes_df = all_df[all_df["has_notes"]].copy()
skipped = len(all_df) - len(notes_df)

patient_notes = notes_df[
    [
        "patient_id",
        "note_id",
        "note_type",
        "charttime",
        "text",
    ]
].rename(columns={"text": "note_text"})

out_path = ROOT / "data" / "patient_notes" / "patient_notes.csv"
patient_notes.to_csv(out_path, index=False)
print(
    f"[Step 3] patient_notes.csv -> {out_path}  ({len(patient_notes)} rows, {skipped} skipped)"
)

# ── Step 4: ground_truth.csv ─────────────────────────────────────────────────
ground_truth = all_df[
    [
        "patient_id",
        "note_id",
        "y_true",
        "dx_only_label",
        "source_file",
    ]
].copy()

out_path = ROOT / "outputs" / "ground_truth.csv"
ground_truth.to_csv(out_path, index=False)
print(f"[Step 4] ground_truth.csv  -> {out_path}  ({len(ground_truth)} rows)")

# ── Step 5: review_log.csv ───────────────────────────────────────────────────
review_log = (
    all_df[
        [
            "patient_id",
            "note_id",
            "reviewer",
            "y_true",
            "source_file",
        ]
    ]
    .rename(columns={"y_true": "decision"})
    .copy()
)
review_log.insert(3, "minutes_spent", None)

out_path = ROOT / "outputs" / "review_log.csv"
review_log.to_csv(out_path, index=False)
print(f"[Step 5] review_log.csv    -> {out_path}  ({len(review_log)} rows)")

# ── Step 6: Quality report ───────────────────────────────────────────────────
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
    label = {1: "1 (AD/ADRD)", 0: "0 (No ADRD)", -1: "-1 (uncertain)"}.get(
        val, f"NaN/empty"
    )
    print(f"  {label}: {cnt}")

no_notes = (all_df["has_notes"] == False).sum()
print(f"\n[has_notes=False 的行数]: {no_notes}")

# subject_id 被多个标注者标注
overlap = (
    all_df.groupby("patient_id")["reviewer"]
    .nunique()
    .reset_index()
    .rename(columns={"reviewer": "reviewer_count"})
)
multi = overlap[overlap["reviewer_count"] > 1]
print(f"\n[被多个标注者标注的 subject_id 数 (Kappa 候选)]: {len(multi)}")
if len(multi) > 0:
    print("  Top 10:")
    for _, row in multi.head(10).iterrows():
        pid = row["patient_id"]
        revs = all_df[all_df["patient_id"] == pid]["reviewer"].unique().tolist()
        print(f"    patient_id={pid}  reviewers={revs}")

print("\n[dx_only_label 填写情况]")
filled = all_df["dx_only_label"].notna().sum()
print(f"  adrd_dx(final) 非空行数: {filled} / {len(all_df)}  (0=未填写)")

print("\n[jim 标注者 y_true 状态]")
jim_df = raw_frames["reviewer_jim"]
jim_df["y_true_parsed"] = jim_df[LABEL_COL].apply(parse_label)
print(
    f"  jim 文件 y_true 非空: {jim_df['y_true_parsed'].notna().sum()} / {len(jim_df)}"
)
print("  => jim 标注列为空，需要手动补充标注")

print("\nDone.")
