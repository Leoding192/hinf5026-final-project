"""
HINF 5026 Final Project — AD/ADRD Identification from EHR
Revised by Codex Review | 2026-03-27
Stack: Python + Ollama (Qwen2.5:7b) + LangGraph
"""

# ============================================================
# 0. SETUP
# pip install ollama langgraph pandas scikit-learn matplotlib seaborn jupyter python-dotenv
# brew install ollama && ollama pull qwen2.5:7b
# ============================================================

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
MAX_CHARS = 6000  # safe range for qwen2.5:7b (32K context)


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
    """Create blank annotation CSV template."""
    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": "",  # 1 / 0 / uncertain
            "evidence_type": "",  # ICD / Medication / CogTest / Note / None
            "evidence_text": "",  # supporting quote from EHR
            "negation": "",  # yes / no
            "confidence": "",  # High / Medium / Low
            "annotator": "",
            "notes": "",
        }
    )
    df.to_csv(output_path, index=False)
    print(f"Template saved: {output_path} ({len(patient_ids)} patients)")


def check_kappa(file_a: str, file_b: str) -> float:
    """Compute Cohen's Kappa between two annotators."""
    a = pd.read_csv(file_a).set_index("patient_id")["label"]
    b = pd.read_csv(file_b).set_index("patient_id")["label"]
    common = a.index.intersection(b.index)
    kappa = cohen_kappa_score(a[common], b[common])
    print(f"Cohen's Kappa = {kappa:.3f}")
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


def evaluate(y_true, y_pred, y_prob=None, model_name: str = "") -> dict:
    """
    Compute Precision/PPV, Recall/Sensitivity, F1, ROC-AUC.
    y_prob: probability scores for AUC (required for ROC-AUC).
    """
    result = {
        "model": model_name,
        "precision_ppv": round(precision_score(y_true, y_pred), 4),
        "recall_sensitivity": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": (
            round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else "N/A"
        ),
    }
    print(f"\n=== {model_name} ===")
    for k, v in result.items():
        if k != "model":
            print(f"  {k}: {v}")
    return result


# ============================================================
# 4. OLLAMA CLIENT (Qwen2.5 / Llama)
# ============================================================

import ollama
import json

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "integer", "enum": [0, 1]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence": {"type": "string"},
        "probability": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["label", "confidence", "evidence", "probability"],
}


def call_llm(prompt: str, model: str = "qwen2.5:7b") -> dict:
    """
    Call local Ollama model with forced JSON output.
    Falls back to error dict if JSON parsing fails.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a clinical NLP expert specializing in "
                "identifying Alzheimer's Disease and Related "
                "Dementias (AD/ADRD) from electronic health records.",
            },
            {"role": "user", "content": prompt},
        ],
        format="json",  # enforce structured output
    )
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse failed: {e}")
        return {
            "label": -1,
            "confidence": "error",
            "evidence": str(e),
            "probability": -1,
        }


# ============================================================
# 5. PROMPT TEMPLATES
# ============================================================

ZERO_SHOT = """
Analyze the EHR text below. Does this patient have AD/ADRD?
Return JSON with: label (0 or 1), confidence (high/medium/low),
evidence (key supporting text), probability (0.0-1.0).

EHR Text:
{ehr_text}
"""

FEW_SHOT = """
Examples:
Input: "G30.9 Alzheimer's disease. Started donepezil 10mg daily."
Output: {{"label":1,"confidence":"high","evidence":"G30.9, donepezil","probability":0.95}}

Input: "Patient reports mild forgetfulness. MMSE 28/30. No dementia diagnosis."
Output: {{"label":0,"confidence":"high","evidence":"MMSE 28, no diagnosis","probability":0.05}}

Now analyze:
{ehr_text}
Return JSON only.
"""

COT = """
Analyze step by step:
Step 1: ICD-10 codes — any G30/G31/F00-F03?
Step 2: Medications — donepezil/memantine/rivastigmine/galantamine?
Step 3: Cognitive scores — MMSE<24? MoCA<26? CDR>=1?
Step 4: Note keywords — dementia/Alzheimer/MCI? Any negations (e.g. "no dementia")?
Step 5: Final judgment based on steps 1-4.

EHR Text:
{ehr_text}

Return JSON: label (0 or 1), confidence (high/medium/low),
evidence (summary), probability (0.0-1.0), reasoning (brief).
"""


# ============================================================
# 6. BATCH LLM INFERENCE
# ============================================================

import time


def run_batch_inference(
    data_csv: str, output_csv: str, model: str = "qwen2.5:7b", template: str = COT
):
    """
    Run LLM inference on all patients in data_csv.
    Saves predictions to output_csv.
    """
    df = pd.read_csv(data_csv)
    results = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] patient {row['patient_id']}")
        ehr = extract_relevant_text(str(row["ehr_text"]))
        prompt = template.format(ehr_text=ehr)
        result = call_llm(prompt, model)
        result["patient_id"] = row["patient_id"]
        results.append(result)
        time.sleep(0.3)  # avoid overloading local model

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv} ({len(out_df)} rows)")
    return out_df


# ============================================================
# 7. AGENT PIPELINE (LangGraph — Parallel Fan-out)
# ============================================================
#
# Architecture:
#   START ──┬── ICD Agent ────┬──> Synthesis ──> END
#           ├── Med Agent ────┤
#           └── Note Agent ───┘

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional


class ADRDState(TypedDict):
    patient_id: str
    ehr_text: str
    icd_score: float
    med_score: float
    note_score: float
    final_label: Optional[int]
    confidence: Optional[str]
    probability: Optional[float]
    evidence_chain: List[str]


_MODEL = "qwen2.5:7b"


def _ask(prompt: str) -> dict:
    r = ollama.chat(
        model=_MODEL, messages=[{"role": "user", "content": prompt}], format="json"
    )
    try:
        return json.loads(r["message"]["content"])
    except json.JSONDecodeError:
        return {"score": 0.0, "evidence": "parse error"}


def icd_agent(state: ADRDState) -> dict:
    result = _ask(
        f"""
Find AD/ADRD ICD-10 codes (G30, G31, F00-F03) in this text.
Return JSON: {{"found": [], "score": 0.0-1.0, "evidence": ""}}
Text: {state['ehr_text'][:2000]}
"""
    )
    score = result.get("score", 0.0)
    chain = list(state["evidence_chain"])
    if score > 0.3 and result.get("evidence"):
        chain.append(f"ICD: {result['evidence']}")
    return {"icd_score": score, "evidence_chain": chain}


def med_agent(state: ADRDState) -> dict:
    result = _ask(
        f"""
Find AD/ADRD medications (donepezil/memantine/rivastigmine/galantamine) in this text.
Return JSON: {{"found": [], "score": 0.0-1.0, "evidence": ""}}
Text: {state['ehr_text'][:2000]}
"""
    )
    score = result.get("score", 0.0)
    chain = list(state["evidence_chain"])
    if score > 0.3 and result.get("evidence"):
        chain.append(f"Med: {result['evidence']}")
    return {"med_score": score, "evidence_chain": chain}


def note_agent(state: ADRDState) -> dict:
    result = _ask(
        f"""
Find cognitive decline evidence (dementia/Alzheimer/MCI/MMSE<24/MoCA<26) in this text.
Handle negations: "no dementia" or "dementia ruled out" → score near 0.
Return JSON: {{"found": [], "score": 0.0-1.0, "evidence": ""}}
Text: {state['ehr_text'][:3000]}
"""
    )
    score = result.get("score", 0.0)
    chain = list(state["evidence_chain"])
    if score > 0.3 and result.get("evidence"):
        chain.append(f"Note: {result['evidence']}")
    return {"note_score": score, "evidence_chain": chain}


def synthesis_agent(state: ADRDState) -> dict:
    """Weighted combination: ICD 50% + Medication 30% + Note 20%."""
    weighted = (
        state["icd_score"] * 0.5 + state["med_score"] * 0.3 + state["note_score"] * 0.2
    )
    if weighted >= 0.6:
        label, conf = 1, "high"
    elif weighted >= 0.3:
        label, conf = 1, "medium"  # consider manual review
    else:
        label, conf = 0, "high"
    return {"final_label": label, "confidence": conf, "probability": round(weighted, 4)}


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
        "evidence": result["evidence_chain"],
    }


def run_agent_batch(data_csv: str, output_csv: str):
    """Run agent pipeline on all patients in data_csv."""
    df = pd.read_csv(data_csv)
    results = []
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] agent: patient {row['patient_id']}")
        r = run_agent(row["patient_id"], str(row["ehr_text"]))
        results.append(r)
    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"\nAgent results saved: {output_csv}")
    return out


# ============================================================
# 8. FINAL COMPARISON (run all models and plot)
# ============================================================

import matplotlib.pyplot as plt


def compare_all_models(
    ground_truth_csv: str, results_map: dict, output_fig: str = "model_comparison.png"
):
    """
    ground_truth_csv: CSV with columns [patient_id, true_label]
    results_map: dict of {model_name: predictions_csv_path}
                 each CSV needs [patient_id, label, probability(optional)]
    """
    gt = pd.read_csv(ground_truth_csv)
    all_results = []

    for name, path in results_map.items():
        df = pd.read_csv(path)
        merged = gt.merge(df, on="patient_id")
        prob = (
            merged["probability"].tolist() if "probability" in merged.columns else None
        )
        r = evaluate(
            merged["true_label"].tolist(),
            merged["label"].tolist(),
            y_prob=prob,
            model_name=name,
        )
        all_results.append(r)

    rdf = pd.DataFrame(all_results).set_index("model")
    metrics = ["precision_ppv", "recall_sensitivity", "f1"]
    if rdf["roc_auc"].dtype != object:
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
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # --- Step 1: create annotation template ---
    # create_annotation_template(
    #     patient_ids=list(range(1, 201)),
    #     output_path="data/annotations/template.csv"
    # )

    # --- Step 2: check inter-annotator agreement ---
    # check_kappa("data/annotations/annotator_a.csv",
    #             "data/annotations/annotator_b.csv")

    # --- Step 3: run LLM inference (CoT) ---
    # run_batch_inference(
    #     data_csv="data/raw/patients.csv",
    #     output_csv="data/results/llm_cot.csv",
    #     model="qwen2.5:7b",
    #     template=COT
    # )

    # --- Step 4: run agent pipeline ---
    # run_agent_batch(
    #     data_csv="data/raw/patients.csv",
    #     output_csv="data/results/agent.csv"
    # )

    # --- Step 5: compare all models ---
    # compare_all_models(
    #     ground_truth_csv="data/annotations/final_labels.csv",
    #     results_map={
    #         "Baseline (HW1)":   "data/results/baseline.csv",
    #         "LLM Zero-shot":    "data/results/llm_zeroshot.csv",
    #         "LLM Few-shot":     "data/results/llm_fewshot.csv",
    #         "LLM CoT":          "data/results/llm_cot.csv",
    #         "Multi-Agent":      "data/results/agent.csv",
    #     },
    #     output_fig="figures/model_comparison.png"
    # )

    print("hinf5026_final_project.py loaded successfully.")
    print("Uncomment the steps in __main__ to run each milestone.")
