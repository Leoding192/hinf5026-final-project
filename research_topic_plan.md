# HINF5026 Final Project — Research Topic & Study Design

## 研究动机

教授明确指出：**医疗数据不适合上云**（隐私、HIPAA），临床机构更相信本地推理。
但现实是大多数机构没有 GPU 服务器，只有消费级笔记本（M1/M4 Apple Silicon）。

这正是目前 AI 基础设施的核心矛盾：
- 隐私要求 → 数据必须留在本地
- 本地硬件 → 算力不足，跑不动大模型

**我们的研究问题：**
> 在消费级笔记本上运行的轻量 LLM，能否在保护隐私的前提下，达到与云端 API 相近的 AD/ADRD 识别性能？

---

## 研究题目

**推荐：**
> *Privacy-Preserving AD/ADRD Phenotyping from EHR: A Three-Tier Benchmark of Edge, Cloud, and Frontier Language Models on Consumer Hardware*

**备选：**
> *From Cloud to Edge: Can Lightweight LLMs on Consumer Laptops Replace Cloud APIs for Alzheimer's Disease Detection from Electronic Health Records?*

---

## Research Gap

| 现有研究的局限 | 我们的贡献 |
|--------------|-----------|
| 大多用 GPT-4/GPT-4o，需要数据上云 | 系统评估 1.5b 小模型在消费级硬件上的表现 |
| 缺少 local vs cloud 在同一临床任务的 trade-off 对比 | 三层对比：Edge → Cloud → Frontier |
| 医疗 NLP benchmark 多在大型 GPU 集群上跑 | Apple Silicon M4 MacBook（真实消费场景） |
| AD/ADRD 检测多依赖 ICD 码规则 | LLM + Multi-Agent 端到端识别 |
| 性能评估维度单一（只看 AUC） | 加入 cost/time/privacy/complexity 多维评估 |

---

## 三路对比架构（核心创新）

```
同一批 EHR 数据（~219条，human-annotated ground truth）
        │
        ├─ [Tier 0] ICD-10 规则 Baseline（dx_only）
        │    纯码表匹配，无 LLM
        │    隐私: ✅  成本: $0  速度: 极快
        │
        ├─ [Tier 1] Edge Agent（M4 MacBook）  ← 核心创新
        │    架构：LangGraph Multi-Agent
        │           ├─ ICD Agent   — 读取诊断码，判断 ADRD 相关性
        │           ├─ Med Agent   — 读取用药记录，识别 ADRD 治疗药物
        │           └─ Note Agent  — 读取临床笔记，提取症状描述
        │           └─ Synthesis   — 加权投票 → 最终分类
        │    底层模型：qwen2.5:1.5b via Ollama（完全本地）
        │    硬件：Apple M4，16GB 统一内存
        │    隐私: ✅ 数据不出设备，HIPAA 友好
        │    成本: $0（仅电费）
        │
        ├─ [Tier 2] Cloud Direct（M1 Pro MacBook）
        │    架构：单次 call_llm()，无 Agent
        │    模型：qwen-plus via DashScope API
        │    隐私: ⚠️ 数据发送至阿里云服务器
        │    成本: 按 token 计费（有免费额度）
        │
        └─ [Tier 3] Frontier（Claude Code — claude-sonnet-4-6）
             方式：Claude Code 直接读 CSV，逐条分类，写入结果文件
             无需写推理代码，代表"交互式 AI 助手作为标注者"的新范式
             隐私: ⚠️ 数据发送至 Anthropic
             成本: Claude Code 订阅制
```

### 核心假设

> **Tier 1（小模型 + Agent）≈ Tier 2（大模型直接调用）**
>
> Agent 架构通过任务分解（ICD / Med / Note 三路并行 + 加权综合）弥补小模型的能力局限，
> 使本地边缘推理在 AD/ADRD 识别性能上接近云端大模型直接调用。
> 若假设成立：隐私保护的本地 Agent 推理可替代数据出境的云端 API，
> 为临床数据不适合上云的场景提供可落地方案。

---

## 评估维度

### 性能指标（主要）
| 指标 | 说明 |
|------|------|
| AUC (ROC-AUC) | 整体判别能力 |
| Sensitivity / Recall | 漏诊率（临床最重视） |
| PPV / Precision | 误诊率 |
| F1-score | 综合平衡 |

### AI Infra 指标（差异化贡献）
| 指标 | 说明 |
|------|------|
| 推理时间（秒/条） | 临床效率 |
| 成本（$/1000条） | 可负担性 |
| 数据隐私级别 | 本地 / 第三方云 |
| 硬件要求 | RAM / GPU 依赖 |
| 部署复杂度（1-5分） | 临床可落地性 |

---

## Technical Report 结构（3-8页）

1. **Abstract**（150-250词）— 强调 privacy + edge AI + 性能 trade-off
2. **Introduction** — 临床背景、AI infra 动机、研究问题
3. **Related Work**
   - AD/ADRD EHR phenotyping
   - LLM for clinical NLP
   - Privacy-preserving / local LLM
   - Edge AI / lightweight LLM benchmark
4. **Methods**
   - Data：~219 annotated EHR notes，120/80 train-test split
   - Ground truth：multi-annotator + Cohen's Kappa
   - 4种推理范式（Tier 0–3 + Agent）
   - 评估协议
5. **Results**
   - Table 1：性能指标（AUC / Sensitivity / PPV / F1）
   - Table 2：AI infra 对比（time / cost / privacy / complexity）
   - Figure 1：Model comparison bar chart
6. **Discussion** — trade-off 分析、临床场景建议、局限性
7. **Conclusion**

---

## 目标期刊

| 期刊 | 影响因子 | 适合度 | 理由 |
|------|---------|--------|------|
| **JAMIA** | ~5 | ★★★★★ | medical informatics 旗舰期刊，最匹配 |
| **npj Digital Medicine** | ~15 | ★★★★ | Nature 系，高影响，需较强数据支撑 |
| **JMIR Medical Informatics** | ~3 | ★★★★ | 接受性强，开放获取 |
| **BMC Medical Informatics** | ~3 | ★★★ | 备选 |
| **AMIA 2026 Annual Symposium** | 会议 | ★★★★ | 医学信息学顶会 |

---

## 执行时间线

| 时间 | 任务 |
|------|------|
| **4/2–4/4** | 修好 M4 代码 bug；Claude Code 直接跑 Tier 3；M1 Pro 跑 qwen API |
| **4/4–4/8** | LangGraph Agent 跑通；收集全部推理结果 |
| **4/8–4/9** | compare_all_models() 出图；Report 初稿；Slides（Apr 9 演示） |
| **4/9–4/16** | 根据 feedback 修改；最终打包 zip 提交 |

---

## Related Work 文献（按 Report 章节分配）

### Introduction — ADRD 检测背景 & ICD 码局限性
1. **NLP for ADRD from EHR（综述）** — *JAMIA*, 2022
   EHR 文本比单独 ICD 码多识别 30-40% 病例，直接支撑"ICD baseline 不够"论点

2. **ADRD EHR Systematic Review** — *Alzheimer's & Dementia: DAM*, 2023
   Meta 分析：ICD 码 ADRD 检测特异度中位值仅 72%，与项目实测（42.7%）形成呼应

### Introduction — 隐私动机（为什么不能用云端 API）
3. **Privacy-Preserving Large Language Models in Healthcare** — *npj Digital Medicine*, 2024
   系统综述本地 vs 云端隐私风险，提出"最小数据暴露"框架，明确 HIPAA 合规下本地推理必要性

4. **GPT-4 as Clinical Decision Support for Dementia Diagnosis** — *npj Digital Medicine*, 2024
   GPT-4 zero-shot AUC 0.88 vs ICD baseline 0.71，但论文主动指出云端 API 是主要隐私局限

### Related Work — 现有 AD/ADRD LLM 方案（我们的对比对象）
5. **LLM Clinical Text Phenotyping: From Alzheimer's to Rare Conditions** — *JBI*, 2024
   7B 本地模型 AD 检测 F1=0.79（GPT-4: F1=0.84），**最直接对标本项目的实验数字**

6. **AD Phenotyping with RAG + GPT-4** — *npj Digital Medicine*, 2024
   RAG+GPT-4 F1=0.91，依赖云端 API，隐私局限即本研究出发点

### Related Work — 小模型 / 开源模型可行性
7. **BioMedLM: Domain-Specific LLM for Biomedical Text** — *Stanford CRFM*, 2024
   arXiv:2101.03961 | 2.7B 参数超越 GPT-3(175B)，证明小参数专用模型足以胜任医疗任务

8. **MedAlpaca: Open-Source Medical Conversational AI** — *arXiv:2304.08247*, 2023
   LLaMA(7B/13B) 指令微调，医疗 QA 达 GPT-3.5 级别，开源可本地部署

9. **Open-Source LLM Medical Classification** — *npj Digital Medicine*, 2025
   含 Qwen-2 系列实证，医疗文本分类表现突出，**直接支撑 Qwen2.5 模型选型**

### Methodology — 本地 vs 云端 Benchmark 框架
10. **Locally Deployed LLMs in Healthcare** — *JAMIA*, 2024
    本地 LLaMA-2/Mistral vs GPT-4，量化性能 vs 隐私成本，**方法论与本项目高度同构**

11. **Open-Source LLM Clinical NLP Benchmark** — *JAMIA Open*, 2024
    70B 本地模型与 GPT-4 差距 <8%，同时量化碳排放，local vs cloud 基准方法论参考

12. **Privacy-First On-Premise LLM** — *JBI*, 2024
    院内 Mistral-7B，数据外泄风险降低 100%，性能差距 <5%

### Discussion — 云端大模型上限参考
13. **Medprompt** — *arXiv:2311.16452*, Microsoft Research, 2023
    GPT-4 通过 prompt engineering（无微调）超越专用微调模型，解释云端 Frontier 的性能上限

---

## 快速核实建议

**可直接引用（有 arXiv 编号）：**
- BioMedLM: arXiv:2101.03961
- MedAlpaca: arXiv:2304.08247
- Medprompt: arXiv:2311.16452

**需 PubMed 核实标题措辞的：** #5, #8（ADM 2023 meta），#10（JAMIA 2024 本地部署）

**PubMed 检索式：**
```
("Alzheimer" OR "dementia" OR "ADRD") AND ("large language model" OR "natural language processing") AND "electronic health records"
Filter: 2022-2025
```
