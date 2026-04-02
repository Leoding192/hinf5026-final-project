"""
HINF 5026 Final Project — AD/ADRD Identification from EHR
Revised by Codex Review | 2026-03-27
Stack: Python + Ollama (qwen2.5:0.5b) + LangGraph
"""

# ============================================================
# 0. SETUP
# pip install ollama langgraph pandas scikit-learn matplotlib seaborn jupyter python-dotenv
# brew install ollama && ollama pull qwen2.5:0.5b
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
MAX_CHARS = 6000  # safe range for qwen2.5:0.5b (32K context)


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

SYSTEM_PROMPT = (
    "You are a clinical NLP expert specializing in identifying Alzheimer's Disease "
    "and Related Dementias (AD/ADRD) from electronic health records. "
    "Always respond with valid JSON only."
)


def call_llm(
    prompt: str, model: str = "qwen2.5:1.5b", provider: str = "ollama"
) -> dict:
    """
    Call LLM with forced JSON output.
    provider: "ollama" (local) or "qwen" (Qwen API via DashScope).
    Falls back to error dict if JSON parsing fails.
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
        print(f"  [WARN] JSON parse failed: {e}")
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
    data_csv: str,
    output_csv: str,
    model: str = "qwen2.5:1.5b",
    template: str = COT,
    provider: str = "ollama",
):
    """
    Run LLM inference on all patients in data_csv.
    provider: "ollama" (local Ollama) or "qwen" (Qwen API).
    Saves predictions to output_csv.
    """
    df = pd.read_csv(data_csv)
    results = []

    # 自动识别病历文本列名
    text_col = None
    for candidate in [
        "ehr_text",
        "note_text",
        "patient_note",
        "note",
        "text",
        "content",
    ]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(
            f"Could not find EHR text column in {data_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # 自动识别患者ID列名
    id_col = None
    for candidate in ["patient_id", "subject_id", "id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f"Could not find patient id column in {data_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # Qwen API 无需 sleep；Ollama 小模型需要限速
    sleep_time = 0.0 if provider == "qwen" else (1.0 if "0.5b" in model else 0.3)

    for i, row in df.iterrows():
        patient_id = row[id_col]
        print(f"[{i+1}/{len(df)}] patient {patient_id} ({provider})")
        ehr = extract_relevant_text(str(row[text_col]))
        prompt = template.format(ehr_text=ehr)
        result = call_llm(prompt, model, provider)
        result["patient_id"] = patient_id
        results.append(result)
        if sleep_time > 0:
            time.sleep(sleep_time)

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


_MODEL = "qwen2.5:1.5b"
_PROVIDER = "ollama"


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


def run_agent_batch(
    data_csv: str,
    output_csv: str,
    model: str = "qwen2.5:1.5b",
    provider: str = "ollama",
):
    """Run agent pipeline on all patients in data_csv.
    provider: "ollama" (local) or "qwen" (Qwen API).
    """
    global _MODEL, _PROVIDER
    _MODEL = model
    _PROVIDER = provider

    df = pd.read_csv(data_csv)
    results = []

    # 自动识别病历文本列名
    text_col = None
    for candidate in [
        "ehr_text",
        "note_text",
        "patient_note",
        "note",
        "text",
        "content",
    ]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(
            f"Could not find EHR text column in {data_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # 自动识别患者ID列名
    id_col = None
    for candidate in ["patient_id", "subject_id", "id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f"Could not find patient id column in {data_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    sleep_time = 0.0 if provider == "qwen" else (1.0 if "0.5b" in model else 0.3)

    for i, row in df.iterrows():
        patient_id = row[id_col]
        print(f"[{i+1}/{len(df)}] agent: patient {patient_id} ({provider})")
        r = run_agent(patient_id, str(row[text_col]))
        results.append(r)
        if sleep_time > 0:
            time.sleep(sleep_time)

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
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
    INPUT_CSV = os.path.join(BASE_DIR, "data", "patient_notes", "patient_notes.csv")

    os.makedirs(RESULT_DIR, exist_ok=True)

    # --- Step 1: run LLM inference (CoT, local qwen2.5:1.5b on M4) ---
    run_batch_inference(
        data_csv=INPUT_CSV,
        output_csv=os.path.join(RESULT_DIR, "llm_qwen_cot.csv"),
        model="qwen2.5:1.5b",
        template=COT,
        provider="ollama",
    )

    # --- Step 2: run LLM inference (CoT, Qwen API for comparison) ---
    # Requires: pip install openai && export DASHSCOPE_API_KEY=...
    run_batch_inference(
        data_csv=INPUT_CSV,
        output_csv=os.path.join(RESULT_DIR, "llm_qwen_api_cot.csv"),
        model="qwen-plus",
        template=COT,
        provider="qwen",
    )

    # --- Step 3: run agent pipeline (local qwen2.5:1.5b) ---
    run_agent_batch(
        data_csv=INPUT_CSV,
        output_csv=os.path.join(RESULT_DIR, "agent_qwen.csv"),
        model="qwen2.5:1.5b",
        provider="ollama",
    )

    # --- Step 5: compare all models ---
    if os.path.exists(os.path.join(BASE_DIR, "ground_truth.csv")):
        compare_all_models(
            ground_truth_csv=os.path.join(BASE_DIR, "ground_truth.csv"),
            results_map={
                "Baseline (DX Only)": os.path.join(BASE_DIR, "dx_only_baseline.csv"),
                "LLM Zero-shot": os.path.join(RESULT_DIR, "llm_zeroshot.csv"),
                "LLM Few-shot": os.path.join(RESULT_DIR, "llm_fewshot.csv"),
                "LLM CoT": os.path.join(RESULT_DIR, "llm_cot.csv"),
                "Multi-Agent": os.path.join(RESULT_DIR, "agent.csv"),
            },
            output_fig=os.path.join(BASE_DIR, "model_comparison_plot.png"),
        )
    else:
        print("⚠️ 没有 ground_truth.csv，跳过 Step 5")

    print("✅ 全流程运行完成！")
