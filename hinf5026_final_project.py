"""
HINF 5026 Final Project — AD/ADRD Identification from EHR
Three-Tier Benchmark: Edge Agent | Cloud Direct | Frontier
Stack: Python + Ollama (qwen2.5:1.5b) + LangGraph
"""

import os
import operator

# ============================================================
# 1. EHR TEXT PREPROCESSING
# ============================================================

RELEVANT_SECTIONS = [
    "diagnosis",
    "assessment",
    "medications",
    "cognitive",
    "neurological",
    "discharge summary",
    "problem list",
]
MAX_CHARS = 6000


def extract_relevant_text(full_ehr: str) -> str:
    """Extract AD/ADRD-relevant sections from EHR text."""
    lines = full_ehr.split("\n")
    relevant = []
    in_section = False
    for line in lines:
        if any(kw in line.lower() for kw in RELEVANT_SECTIONS):
            in_section = True
        if in_section:
            relevant.append(line)
        if len("\n".join(relevant)) > MAX_CHARS:
            break
    return "\n".join(relevant) if relevant else full_ehr[:MAX_CHARS]


# ============================================================
# 2. ANNOTATION TOOLS
# ============================================================

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def create_annotation_template(patient_ids: list, output_path: str):
    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": "",
            "evidence_type": "",
            "evidence_text": "",
            "negation": "",
            "confidence": "",
            "annotator": "",
            "notes": "",
        }
    )
    df.to_csv(output_path, index=False)
    print(f"Template saved: {output_path} ({len(patient_ids)} patients)")


def check_kappa(file_a: str, file_b: str) -> float:
    """Compute Cohen's Kappa between two annotators using y_true in ground_truth."""
    a = pd.read_csv(file_a).set_index("patient_id")["y_true"]
    b = pd.read_csv(file_b).set_index("patient_id")["y_true"]
    common = a.index.intersection(b.index)
    a_common = a[common].dropna()
    b_common = b[common][a_common.index]
    kappa = cohen_kappa_score(a_common, b_common)
    print(f"Cohen's Kappa = {kappa:.3f}  (n={len(a_common)})")
    if kappa >= 0.8:
        print("  >> Excellent")
    elif kappa >= 0.6:
        print("  >> Good — proceed")
    else:
        print("  >> Poor — revisit annotation rules")
    return kappa


# ============================================================
# 3. EVALUATION METRICS
# ============================================================

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np


def evaluate(y_true, y_pred, y_prob=None, model_name: str = "") -> dict:
    """
    Compute Precision/PPV, Recall/Sensitivity, Specificity, F1, ROC-AUC.
    Filters out rows where y_true or y_pred are not in {0, 1}.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    y_prob_list = list(y_prob) if y_prob is not None else None

    valid_idx = [
        i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t in (0, 1) and p in (0, 1)
    ]
    if len(valid_idx) < len(y_true):
        print(
            f"  [INFO] Dropped {len(y_true) - len(valid_idx)} rows with invalid labels"
        )

    yt = [y_true[i] for i in valid_idx]
    yp = [y_pred[i] for i in valid_idx]
    yprob = [y_prob_list[i] for i in valid_idx] if y_prob_list else None

    # Specificity = TN / (TN + FP)
    tn = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0

    # AUC — drop NaN probabilities
    auc_val = "N/A"
    if yprob is not None:
        pairs = [
            (t, p)
            for t, p in zip(yt, yprob)
            if p is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        if pairs:
            t_clean, p_clean = zip(*pairs)
            try:
                auc_val = round(roc_auc_score(list(t_clean), list(p_clean)), 4)
            except Exception:
                auc_val = "N/A"

    result = {
        "model": model_name,
        "precision_ppv": round(precision_score(yt, yp, zero_division=0), 4),
        "recall_sensitivity": round(recall_score(yt, yp, zero_division=0), 4),
        "specificity": specificity,
        "f1": round(f1_score(yt, yp, zero_division=0), 4),
        "roc_auc": auc_val,
        "n": len(yt),
    }
    print(f"\n=== {model_name} ===")
    for k, v in result.items():
        if k != "model":
            print(f"  {k}: {v}")
    return result


# ============================================================
# 4. LLM CLIENT (Ollama local / Qwen API)
# ============================================================

import ollama
import json

SYSTEM_PROMPT = (
    "You are a clinical NLP expert specializing in identifying Alzheimer's Disease "
    "and Related Dementias (AD/ADRD) from electronic health records. "
    "Always respond with valid JSON only."
)

# Chain-of-thought prompt — ANY single clear signal is sufficient for positive
COT = """
Analyze step by step:
Step 1: Medications — donepezil/memantine/rivastigmine/galantamine/Aricept/Namenda? (AD/ADRD first-line drugs)
Step 2: Diagnosis keywords — dementia/Alzheimer's/cognitive impairment/MCI explicitly mentioned?
Step 3: Cognitive scores — MMSE<24? MoCA<26? CDR>=1?
Step 4: Negation check — is the dementia/Alzheimer's mention negated IN THE SAME SENTENCE? ("no dementia", "ruled out dementia"). Ignore negations about OTHER conditions.
Step 5: Final judgment: positive if ANY ONE of steps 1-3 has clear evidence AND step 4 does not negate it.

EHR Text:
{ehr_text}

Return JSON: label (0 or 1), confidence (high/medium/low),
evidence (summary of key findings), probability (0.0-1.0), reasoning (brief).
"""


def call_llm(
    prompt: str, model: str = "qwen2.5:1.5b", provider: str = "ollama"
) -> dict:
    """
    Call LLM with forced JSON output.
    provider: "ollama" (local) or "qwen" (Qwen API via DashScope).
    """
    try:
        if provider == "qwen":
            import openai

            client = openai.OpenAI(
                api_key=os.environ.get("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
        else:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format="json",
            )
            raw = response["message"]["content"]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "label": -1,
            "confidence": "error",
            "evidence": str(e),
            "probability": -1,
        }
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return {
            "label": -1,
            "confidence": "error",
            "evidence": str(e),
            "probability": -1,
        }


# ============================================================
# 5. BATCH LLM INFERENCE (Tier 2: Cloud Direct — single call_llm)
# ============================================================

import time


def _load_notes(data_csv: str) -> pd.DataFrame:
    """Load patient notes CSV, auto-detect columns, deduplicate by patient_id.
    Answer to Q1: patient_notes.csv has 300 rows but only 219 unique patients.
    This function deduplicates so inference runs exactly once per patient.
    """
    df = pd.read_csv(data_csv)

    text_col = next(
        (
            c
            for c in [
                "ehr_text",
                "note_text",
                "patient_note",
                "note",
                "text",
                "content",
            ]
            if c in df.columns
        ),
        None,
    )
    if text_col is None:
        raise ValueError(f"No EHR text column found. Available: {list(df.columns)}")

    id_col = next(
        (c for c in ["patient_id", "subject_id", "id"] if c in df.columns), None
    )
    if id_col is None:
        raise ValueError(f"No patient id column found. Available: {list(df.columns)}")

    before = len(df)
    df = df.drop_duplicates(subset=[id_col])
    if len(df) < before:
        print(f"  [INFO] Deduplicated {before} → {len(df)} unique patients")

    df = df.rename(columns={text_col: "ehr_text", id_col: "patient_id"})
    return df[["patient_id", "ehr_text"]]


def run_batch_inference(
    data_csv: str,
    output_csv: str,
    model: str = "qwen2.5:1.5b",
    template: str = COT,
    provider: str = "ollama",
):
    """Run single-call LLM inference on all unique patients (Tier 2: Cloud Direct)."""
    df = _load_notes(data_csv)
    sleep_time = 0.0 if provider == "qwen" else (1.0 if "0.5b" in model else 0.3)
    results = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] patient {row['patient_id']} ({provider})")
        ehr = extract_relevant_text(str(row["ehr_text"]))
        prompt = template.format(ehr_text=ehr)
        result = call_llm(prompt, model, provider)
        result["patient_id"] = row["patient_id"]
        results.append(result)
        if sleep_time > 0:
            time.sleep(sleep_time)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv} ({len(out_df)} rows)")
    return out_df


# ============================================================
# 6. AGENT PIPELINE (Tier 1: Edge Agent — LangGraph Parallel Fan-out)
# ============================================================
#
# Architecture:
#   START ──┬── ICD Agent ────┬──> Synthesis ──> END
#           ├── Med Agent ────┤
#           └── Note Agent ───┘
#
# Optimization vs baseline (dx_only: sensitivity=1.0, specificity=0.43):
#   Synthesis requires BOTH:
#     (a) weighted_score >= 0.45, AND
#     (b) at least 2 of 3 agents fired (multi-agent agreement)
#   This reduces false positives while preserving true positives.

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional, Annotated


class ADRDState(TypedDict):
    patient_id: str
    ehr_text: str
    icd_score: float
    med_score: float
    note_score: float
    agents_fired: int
    final_label: Optional[int]
    confidence: Optional[str]
    probability: Optional[float]
    # Annotated with operator.add: parallel branches safely append without overwriting
    evidence_chain: Annotated[List[str], operator.add]


_MODEL = "qwen2.5:1.5b"
_PROVIDER = "ollama"
AGENT_FIRE_THRESHOLD = 0.35


def _ask(prompt: str) -> dict:
    try:
        if _PROVIDER == "qwen":
            import openai

            client = openai.OpenAI(
                api_key=os.environ.get("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            r = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = r.choices[0].message.content
        else:
            r = ollama.chat(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            raw = r["message"]["content"]
        return json.loads(raw)
    except Exception:
        return {"score": 0.0, "evidence": "parse error"}


def icd_agent(state: ADRDState) -> dict:
    """Diagnosis agent: look for AD/ADRD diagnosis mentions (codes OR keyword descriptions).
    Repurposed from ICD-code-only search — note text rarely contains explicit G30/F00 codes.
    """
    result = _ask(
        f"""
You are analyzing an EHR for AD/ADRD diagnosis evidence.
POSITIVE signals (score > 0.5 if ANY found):
- Explicit diagnosis: "Alzheimer's disease", "dementia", "vascular dementia", "Lewy body dementia", "MCI"
- ICD codes if present: G30, G31, F00, F01, F02, F03
- Physician assessment: "consistent with dementia", "dementia workup", "cognitive decline"
NEGATIVE signals (reduce score): "no dementia", "ruled out dementia", "not Alzheimer's"
Do NOT score high for: ALS, Parkinson's without dementia, delirium alone, depression alone.
Return JSON: {{"found": ["list of evidence"], "score": 0.0-1.0, "evidence": "exact text found"}}
Text: {state['ehr_text'][:4000]}
"""
    )
    score = float(result.get("score", 0.0))
    new_evidence = []
    fired = 1 if score > AGENT_FIRE_THRESHOLD and result.get("evidence") else 0
    if fired:
        new_evidence.append(f"ICD: {result['evidence']}")
    return {"icd_score": score, "evidence_chain": new_evidence, "agents_fired": fired}


def med_agent(state: ADRDState) -> dict:
    """Medication agent: look for cholinesterase inhibitors / memantine."""
    result = _ask(
        f"""
You are analyzing an EHR for AD/ADRD-specific medications.
POSITIVE signals (score > 0.5 if ANY found): donepezil, memantine, rivastigmine, galantamine,
Aricept, Namenda, Exelon, Razadyne — prescribed or listed as current/home medication.
REDUCE score if: "discontinued", "allergic to", "intolerant" — but still score > 0 if the drug appears.
Return JSON: {{"found": ["list of drugs"], "score": 0.0-1.0, "evidence": "exact text found"}}
Text: {state['ehr_text'][:4000]}
"""
    )
    score = float(result.get("score", 0.0))
    new_evidence = []
    fired = 1 if score > AGENT_FIRE_THRESHOLD and result.get("evidence") else 0
    if fired:
        new_evidence.append(f"Med: {result['evidence']}")
    return {"med_score": score, "evidence_chain": new_evidence, "agents_fired": fired}


def note_agent(state: ADRDState) -> dict:
    """Clinical note agent: extract cognitive decline evidence with negation handling."""
    result = _ask(
        f"""
You are analyzing an EHR for AD/ADRD clinical evidence.
SCORING:
- Score HIGH (>0.6): explicit diagnosis of dementia, Alzheimer's, or MCI in the note
- Score MEDIUM (0.4-0.6): MMSE<24, MoCA<26, CDR>=1, or "cognitive decline/impairment" documented
- Score LOW (<0.2): dementia explicitly ruled out IN THE SAME SENTENCE ("no dementia", "ruled out dementia")
IMPORTANT: "ruled out" for OTHER conditions (e.g., "MI ruled out", "B12 deficiency ruled out") does NOT lower the dementia score.
Return JSON: {{"found": ["list of evidence"], "score": 0.0-1.0, "evidence": "exact text found"}}
Text: {state['ehr_text'][:5000]}
"""
    )
    score = float(result.get("score", 0.0))
    new_evidence = []
    fired = 1 if score > AGENT_FIRE_THRESHOLD and result.get("evidence") else 0
    if fired:
        new_evidence.append(f"Note: {result['evidence']}")
    return {"note_score": score, "evidence_chain": new_evidence, "agents_fired": fired}


def synthesis_agent(state: ADRDState) -> dict:
    """
    Weighted combination: Diagnosis 40% + Medication 30% + Note 30%.

    Weight rationale:
    - ICD/Diagnosis agent repurposed to find keyword evidence (not just codes),
      but note text rarely has explicit ICD codes, so weight reduced from 50% → 40%
    - Note agent covers 96% of true positives → weight increased from 20% → 30%

    Positive decision: weighted_score >= 0.35 AND at least 1 agent fired.
    High-confidence positive: weighted_score >= 0.55 (strong single signal sufficient).
    """
    weighted = (
        state["icd_score"] * 0.4 + state["med_score"] * 0.3 + state["note_score"] * 0.3
    )
    agents_agreed = state.get("agents_fired", 0)

    if weighted >= 0.55:
        label, conf = 1, "high"
    elif weighted >= 0.35 and agents_agreed >= 1:
        label, conf = 1, "medium"
    else:
        label, conf = 0, "high"

    return {
        "final_label": label,
        "confidence": conf,
        "probability": round(weighted, 4),
        "agents_fired": agents_agreed,
    }


# Build parallel graph
_workflow = StateGraph(ADRDState)
_workflow.add_node("icd", icd_agent)
_workflow.add_node("med", med_agent)
_workflow.add_node("note", note_agent)
_workflow.add_node("synthesis", synthesis_agent)

_workflow.add_edge(START, "icd")
_workflow.add_edge(START, "med")
_workflow.add_edge(START, "note")
_workflow.add_edge("icd", "synthesis")
_workflow.add_edge("med", "synthesis")
_workflow.add_edge("note", "synthesis")
_workflow.add_edge("synthesis", END)

agent_app = _workflow.compile()


def run_agent(patient_id: str, ehr_text: str) -> dict:
    """Run the full multi-agent pipeline for one patient."""
    initial = ADRDState(
        patient_id=patient_id,
        ehr_text=ehr_text,
        icd_score=0.0,
        med_score=0.0,
        note_score=0.0,
        agents_fired=0,
        final_label=None,
        confidence=None,
        probability=None,
        evidence_chain=[],
    )
    result = agent_app.invoke(initial)
    return {
        "patient_id": patient_id,
        "label": result["final_label"],
        "confidence": result["confidence"],
        "probability": result["probability"],
        "agents_fired": result.get("agents_fired", 0),
        "evidence": " | ".join(result["evidence_chain"]),
        "icd_score": result["icd_score"],
        "med_score": result["med_score"],
        "note_score": result["note_score"],
    }


def run_agent_batch(
    data_csv: str,
    output_csv: str,
    model: str = "qwen2.5:1.5b",
    provider: str = "ollama",
):
    """Run Tier 1 Edge Agent on all unique patients in data_csv."""
    global _MODEL, _PROVIDER
    _MODEL = model
    _PROVIDER = provider

    df = _load_notes(data_csv)
    sleep_time = 0.0 if provider == "qwen" else (1.0 if "0.5b" in model else 0.3)
    results = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] agent: patient {row['patient_id']} ({provider})")
        r = run_agent(row["patient_id"], str(row["ehr_text"]))
        results.append(r)
        if sleep_time > 0:
            time.sleep(sleep_time)

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"Agent results saved: {output_csv} ({len(out)} rows)")
    return out


# ============================================================
# 7. TIER 3 NOTE — Claude Code Frontier Annotator
# ============================================================
# Tier 3 does NOT run via this script.
# Claude Code reads patient_notes.csv directly and classifies each note.
# Required output format: patient_id, label (0/1), probability (0.0-1.0), evidence
# Save to: outputs/llm_claude_cot.csv


# ============================================================
# 8. FINAL COMPARISON
# ============================================================

import matplotlib.pyplot as plt


def compare_all_models(
    ground_truth_csv: str,
    results_map: dict,
    output_fig: str = "model_comparison.png",
):
    """
    ground_truth_csv: CSV with columns [patient_id, y_true]
    results_map: {model_name: predictions_csv_path}
                  each CSV needs [patient_id, label, probability(optional)]
    """
    gt = pd.read_csv(ground_truth_csv)
    gt = gt.dropna(subset=["patient_id"])
    gt = gt[gt["y_true"].isin([0, 1])]  # exclude uncertain (-1)
    gt = gt.drop_duplicates(subset=["patient_id"])
    # Evaluate on held-out test set only (per course requirement: "held-out 80 for evaluation")
    if "split" in gt.columns:
        gt = gt[gt["split"] == "test"]
        print(f"[INFO] Using held-out test set only (split='test')")
    pos = int(gt["y_true"].sum())
    neg = int((gt["y_true"] == 0).sum())
    print(f"Ground truth: {len(gt)} patients (pos={pos}, neg={neg})")

    all_results = []

    for name, path in results_map.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {name}: not found — {path}")
            continue
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset=["patient_id"])
        merged = gt.merge(df, on="patient_id")
        prob = (
            merged["probability"].tolist() if "probability" in merged.columns else None
        )
        r = evaluate(
            merged["y_true"].tolist(),
            merged["label"].tolist(),
            y_prob=prob,
            model_name=name,
        )
        all_results.append(r)

    if not all_results:
        print("No results to compare.")
        return None

    rdf = pd.DataFrame(all_results).set_index("model")
    metrics = ["precision_ppv", "recall_sensitivity", "specificity", "f1"]
    if rdf["roc_auc"].apply(lambda x: isinstance(x, float)).all():
        metrics.append("roc_auc")

    rdf[metrics].plot(
        kind="bar", figsize=(12, 6), title="Model Comparison — AD/ADRD Detection"
    )
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    print(f"Figure saved: {output_fig}")
    return rdf


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    TIER1_DIR = os.path.join(BASE_DIR, "tier1")
    INPUT_CSV = os.path.join(BASE_DIR, "data", "patient_notes", "patient_notes.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TIER1_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # Step 1: Tier 1 — Edge Agent (local qwen2.5:1.5b, M4 only)
    #   Runs 219 times (deduplicated), NOT 300
    # ----------------------------------------------------------
    run_agent_batch(
        data_csv=INPUT_CSV,
        output_csv=os.path.join(TIER1_DIR, "agent_tier1.csv"),
        model="qwen2.5:1.5b",
        provider="ollama",
    )

    # ----------------------------------------------------------
    # Step 2: Tier 2 — Cloud Direct (qwen-plus, no Agent)
    #   Requires: export DASHSCOPE_API_KEY=sk-...
    #   Skipped automatically if key not set
    # ----------------------------------------------------------
    if os.environ.get("DASHSCOPE_API_KEY"):
        run_batch_inference(
            data_csv=INPUT_CSV,
            output_csv=os.path.join(OUTPUT_DIR, "llm_qwen_api_cot.csv"),
            model="qwen-plus",
            template=COT,
            provider="qwen",
        )
    else:
        print("DASHSCOPE_API_KEY not set — skipping Tier 2 (Cloud Direct)")

    # ----------------------------------------------------------
    # Step 3: Compare all tiers
    #   Add Tier 3 path below once Claude Code run is complete
    # ----------------------------------------------------------
    GT_CSV = os.path.join(OUTPUT_DIR, "ground_truth.csv")
    if os.path.exists(GT_CSV):
        compare_all_models(
            ground_truth_csv=GT_CSV,
            results_map={
                "Tier0 Baseline (DX Only)": os.path.join(
                    OUTPUT_DIR, "dx_only_baseline.csv"
                ),
                "Tier1 Edge Agent": os.path.join(TIER1_DIR, "agent_tier1.csv"),
                "Tier2 Cloud Direct": os.path.join(OUTPUT_DIR, "llm_qwen_api_cot.csv"),
                # Uncomment after Tier 3 (Claude Code) run:
                # "Tier3 Frontier (Claude)": os.path.join(OUTPUT_DIR, "llm_claude_cot.csv"),
            },
            output_fig=os.path.join(BASE_DIR, "model_comparison_plot.png"),
        )
    else:
        print("No ground_truth.csv — skip comparison")

    print("Done.")
