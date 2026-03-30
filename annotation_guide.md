# Annotation Guide — AD/ADRD Identification from EHR

HINF 5026 Final Project | Spring 2026

---

## 1. 任务目标

从 EHR 文本中人工标注每位病人是否患有 **AD/ADRD（阿尔茨海默病 / 相关痴呆）**，产出作为 ground truth 的标注 CSV，供后续监督学习和模型评估使用。

目标人数：**≥ 200 人**（各组 HW1 数据合并 + cross-check）

---

## 2. 数据来源

- 直接使用 **HW1 中已有的 EHR 病人数据**（不另找新数据集）
- 各组 HW1 标注结果汇总后交叉验证，冲突项需重新审阅
- 如需标注更多，在同一数据集中追加（额外标注有加分）

---

## 3. 标注文件格式

保存为 CSV，文件名建议：`annotations_<annotator_initials>.csv`

| 字段 | 类型 | 说明 |
|---|---|---|
| `patient_id` | str | 病人唯一 ID，与 HW1 保持一致 |
| `label` | int/str | **1**（AD/ADRD）/ **0**（非 AD/ADRD）/ **uncertain** |
| `evidence_type` | str | 见第 4 节 |
| `evidence_text` | str | 从 EHR 中摘录的直接支持句（引号内原文） |
| `negation` | str | **yes**（否定表述，如"no dementia"）/ **no** |
| `confidence` | str | **High** / **Medium** / **Low** |
| `annotator` | str | 标注者姓名缩写 |
| `notes` | str | 补充说明、边界情况描述（可空） |

---

## 4. Evidence Type 分类

| 值 | 含义 | 例子 |
|---|---|---|
| `ICD` | ICD 诊断码（F00–F03, G30, G31 等） | "Diagnosis: Alzheimer's disease (F00.1)" |
| `Medication` | AD/ADRD 相关用药 | Donepezil, Memantine, Rivastigmine, Galantamine |
| `CogTest` | 认知测试结果 | MMSE ≤ 23, MoCA ≤ 25, CDR ≥ 0.5 |
| `Note` | 临床 Note 中的描述性证据 | "Patient exhibits progressive memory loss..." |
| `None` | 无任何支持证据，label=0 时使用 | — |

---

## 5. 标注规则

### 5.1 Label = 1（阳性，有 AD/ADRD）

满足以下**任一**条件：
- EHR 中含 AD/ADRD 相关 ICD 码（F00.x, F01.x, F02.x, F03, G30.x, G31.x）
- 处方含 AD/ADRD 一线用药（Donepezil/Aricept, Memantine/Namenda, Rivastigmine/Exelon, Galantamine）
- 认知测试评分明确异常（MMSE ≤ 23 或 MoCA ≤ 25）
- 临床 Note 中医生明确诊断 dementia / Alzheimer's / cognitive impairment

### 5.2 Label = 0（阴性，无 AD/ADRD）

满足以下**所有**条件：
- 无上述 ICD 码
- 无 AD/ADRD 用药
- 认知评分正常或无认知测试记录
- 临床 Note 无 dementia 相关描述

### 5.3 Label = uncertain（不确定）

- 证据矛盾（如有用药但诊断码不符）
- Note 中有认知下降描述但未明确诊断
- 信息不足以判断

> **处理**：uncertain 记录需第二名标注者复查，仍无法确定则排除出训练集。

### 5.4 Negation 处理

若出现否定表述（"no dementia", "rules out Alzheimer's", "not consistent with AD"），则：
- `negation = yes`
- `label = 0`（除非有其他阳性证据）

---

## 6. Confidence 评分标准

| 级别 | 标准 |
|---|---|
| High | 有明确 ICD 码 **且** 有 Note/用药支持，证据一致 |
| Medium | 仅有一类证据（只有 ICD 或只有 Note），或证据略有模糊 |
| Low | 证据推断性强，存在矛盾，或信息极少 |

---

## 7. 标注流程

```
1. 独立标注：每人各自完成分配的病人（建议每人 ≥ 50 人）
2. 交叉验证：对重叠病人计算 Cohen's Kappa（目标 κ ≥ 0.8）
3. 冲突解决：κ < 0.8 时开会讨论边界案例，修订规则
4. 合并：达标后合并为 ground_truth.csv（≥ 200 人）
```

运行 Kappa 检验：
```bash
python hinf5026_final_project.py  # check_kappa(file_a, file_b)
```

---

## 8. 质量要求（对应评分标准）

- `evidence_text` **不能为空**（评分看 reasoning + evidence 质量）
- 每条 uncertain 必须写 `notes` 解释原因
- 至少核心 100 人有 **双人标注**（multiple reviews per patient）
- 最终文件列名必须与模板一致（`patient_id`, `label` 等）

---

## 9. 评估指标（Milestone 1 输出）

| 指标 | 工具 |
|---|---|
| AUC | `sklearn.metrics.roc_auc_score` |
| Sensitivity (Recall) | `sklearn.metrics.recall_score` |
| PPV (Precision) | `sklearn.metrics.precision_score` |
| F1 | `sklearn.metrics.f1_score` |
| 人工标注工时 | 手动记录（总小时数/人） |
| Inter-annotator Agreement | `cohen_kappa_score`（目标 ≥ 0.8） |

参考论文：Blecker et al., JAMA Cardiology, 2016（Slide 参考文献）

---

## 10. 快捷参考：AD/ADRD 相关 ICD-10 码

| ICD-10 | 含义 |
|---|---|
| G30.0 | Alzheimer's disease with early onset |
| G30.1 | Alzheimer's disease with late onset |
| G30.8 | Other Alzheimer's disease |
| G30.9 | Alzheimer's disease, unspecified |
| G31.01 | Pick's disease |
| G31.09 | Other frontotemporal dementia |
| F00.x | Dementia in Alzheimer's disease |
| F01.x | Vascular dementia |
| F02.x | Dementia in other diseases |
| F03 | Unspecified dementia |
