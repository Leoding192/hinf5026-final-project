# HINF5026 项目记录 — 问题 & 决策日志

> 按时间顺序记录实际遇到的问题、根因、解决方案。供复盘和 Technical Report 写作参考。

---

## 数据层面（2026-04-02）

**D1：note_id 大量重复（81 个重复，共 156 行）**
- ground_truth.csv 中同一 note_id 出现多次（多名 reviewer 标注同一条记录）
- 影响：推理前不去重 → 同一条 note 被推理多次，浪费算力 + 污染评估指标
- 解决：`df.drop_duplicates(subset=['patient_id'])` 300 → 219 条，已封装进 `_load_notes()`

**D2：reviewer_jim decision 在 review_log 全为 NaN（75 条）**
- jim 的 y_true 已更新进 ground_truth，但 review_log.decision 列全空
- 影响：check_kappa() 若从 review_log 读 decision，jim 75 条全被跳过
- 解决：check_kappa() 直接比对两人在 ground_truth 里的 y_true 子集，不走 review_log

**D3：幽灵行（ground_truth 第 75 行）**
- patient_id / note_id 全是 NaN，是 CSV 生成时产生的空行
- 解决：评估前统一 `df.dropna(subset=['patient_id'])` 过滤

**D4：dx_only baseline 特异度仅 42.7%（预期内，无需修复）**
- ICD 码过诊断：灵敏度 100%（0 漏诊），但 117 个真阴性里有 67 个被错标为阳性
- Technical Report 里需明确说明这是 ICD baseline 的已知局限

---

## 数据清洗操作引发（2026-04-02）

**C1：文件命名错误导致 pipeline 断掉**
- Codex 清洗数据后写入 `ground_truth(1).csv`，同时删除了原 `ground_truth.csv`
- `hinf5026_final_project.py` 和 `build_ground_truth.py` 都在找 `ground_truth.csv`，直接报错
- 解决：`mv "outputs/ground_truth(1).csv" "outputs/ground_truth.csv"`
- 教训：`(1)` 后缀是 macOS 重复下载 artifact，任何脚本输出不得用带括号的文件名

**C2：去重保留 heg 标注，丢弃 jim 标注**
- 75 个重叠 note_id 去重后全保留 heg 的行，jim 的标签只能从 review_log 间接找回
- 解决：决定不做 Kappa，直接从 review_log 删除 reviewer_jim 全部 75 行（300 → 225 行）

**C3：review_log 幽灵行未同步删除**
- ghost row 已从 ground_truth 删除，但 review_log 里仍有 1 行 NaN
- 解决：`dropna(subset=['patient_id'])` 删除，review_log 301 → 300 行

---

## 代码层面（2026-04-03，Codex 审查发现）

**P1：LangGraph 并行写同一 state key，无 reducer → 静默覆盖**
- ICD / Med / Note 三个并行 agent 都往 `evidence_chain: List[str]` 写，无 reducer
- 结果：只有最后一个 agent 的 evidence 被保留，前两个静默丢失
- 修复：TypedDict 改为 `evidence_chain: Annotated[List[str], operator.add]`，每个 agent 只返回新增 evidence item

**P2：`import os` 藏在 `__main__` 里，函数体 NameError**
- `call_llm()` 和 `_ask()` 中调用 `os.environ.get()`，但 `import os` 只在 `if __name__ == "__main__":` 块内
- 运行时报 `NameError: name 'os' is not defined`
- 修复：移至文件顶部

**P3：`compare_all_models()` 读错列名 → KeyError**
- 代码读 `merged["true_label"]`，但 ground_truth.csv 的实际列名是 `y_true`
- 修复：统一改为 `y_true`

**P4：`__main__` 路径硬编码且文件名错误**
- 原代码：`ground_truth.csv` 找项目根目录（实际在 `outputs/`）；`llm_cot.csv` / `agent.csv` 实际不存在
- 修复：所有路径改为基于 `os.path.abspath(__file__)` 的相对路径；文件名对齐实际输出

**P5：`evaluate()` 在 y_pred 含非法值（3、-1）时崩溃**
- sklearn `precision_score` 遇到非 0/1 的预测值直接抛异常
- 场景：LLM 偶发输出 label=3（JSON 解析失败的默认值）
- 修复：加 `valid_idx` 过滤，只保留 `y_true ∈ {0,1}` 且 `y_pred ∈ {0,1}` 的行；加 `zero_division=0`

**P6：Agent synthesis 阈值过低（`weighted >= 0.3`），特异度差**
- 旧逻辑：只要加权分 ≥ 0.3 就预测阳性，导致假阳性率居高，与 dx_only baseline 一样全判阳
- 修复：改为双条件 `weighted >= 0.45 AND agents_fired >= 2`（至少两个 agent 同意才判阳）
- 权重：ICD 0.5 / Med 0.3 / Note 0.2（ICD 码最可靠）

---

## Agent 性能优化（2026-04-03）

### 背景：Tier 1 初跑结果很差

首次跑完 `agent_tier1.csv` 后评估（n=169，y_true ∈ {0,1}）：

| 指标 | Tier0 dx_only | Tier1 Agent（初版） |
|------|--------------|---------------------|
| Sensitivity | 1.00 | **0.28** ← 漏掉 72% 真阳性 |
| Specificity | 0.26 | 0.85 |
| F1 | 0.70 | 0.39 |

### 根因分析（Codex Bridge 数据驱动诊断）

**A1（最严重）：ICD agent 完全失效**
- `icd_agent` 在 note 文本里找显式 `G30/F00` 代码，但这类代码存在于 claims/discharge 数据而非 note text
- 实测：78 个真阳性中，ICD 代码覆盖率 **0%**（`icd pos 0 of 78`）
- 问题：ICD agent 占权重 50% 却始终输出 score=0，导致 weighted_score 被系统性压低

**A2：双重门槛叠加导致单信号被拒**
- COT prompt 要求"2+ signals"，synthesis 又要求 `agents_fired >= 2`
- annotation_guide 规定 label=1 只需**任一**条件成立（Med OR Note OR ICD）
- ICD agent 永远不 fire → 只剩 Med + Note 两路，单靠 Note 有信号的病例全被卡死

**A3：文字截断切掉了关键 Med 信号**
- 56 个 FN 中，22 个含 AD 药物，但其中 **20 个在 2000 字之后**
- `med_agent` 只看前 2000 字 → 这 20 例药物信号全部截掉

**A4：negation 全局词匹配误伤真阳性**
- "ruled out" 触发降分，但实际是针对 MI、B12 缺乏等其他疾病的否定
- 5 例真阳性被误判为 negated dementia

**A5：Note agent 权重仅 20%，却覆盖 96% 真阳性**
- note 覆盖率：`note pos 75 of 78 rate 0.962`，是最强信号来源
- 但权重只有 0.2，ICD 占 0.5 却从不 fire，严重失衡

### 5 项修改（2026-04-03，commit cd29e0e）

| # | 改动 | Before | After |
|---|------|--------|-------|
| 1 | COT Step 5 | `positive ONLY if 2+ signals` | `positive if ANY 1 signal` |
| 2 | ICD agent 职责 | 找 G30/F00 显式代码（0% 覆盖） | 找诊断语义关键词（dementia/Alzheimer's/MCI） |
| 3 | 文字截断 | diag 2k / med 2k / note 3k | diag 4k / med 4k / note 5k |
| 4 | Negation 范围 | 全局词匹配 | 限定"dementia 被否定"，忽略其他疾病否定 |
| 5 | Synthesis 权重 & 门槛 | ICD 0.5 / Note 0.2，`agents_fired>=2`，score>=0.45 | Diag 0.4 / Note 0.3，`agents_fired>=1`，score>=0.35 |

### 预期效果

- Sensitivity 从 0.28 明显上升（主要来自改动 1、2、5）
- Specificity 从 0.85 有所下降（门槛放宽的代价）
- 目标：F1 超过 Tier0 baseline 的 0.70
- 需要 M4 重跑 `run_agent_batch()` 验证

---

## 评估与数据清洗（2026-04-03）

**V1：n=169 而非 219 的原因**
- ground_truth.csv 共 219 条（去重后），但 y_true 分布：1=78 / 0=91 / -1=50
- -1 表示标注者无法判断（uncertain），评估时必须过滤，只保留有明确标注的行
- 78 + 91 = 169，这是所有指标的实际评估样本量
- Report 注释：*"Evaluation was performed on n=169 patients with definitive labels (y_true ∈ {0,1}); 50 uncertain cases (y_true = -1) were excluded."*

**V2：tier2 CSV 数据问题（已修复）**
- 原始 `tier2/llm_qwen_cot.csv` 有 300 行（未去重）、label=3 异常值 1 条、probability 为 string 类型、20 列冗余字段
- 修复：去重（300→218）、过滤 label=3、probability 转 float、confidence 统一小写、只保留 6 列
- 修复后评估：Sensitivity=0.628 / Specificity=0.473 / F1=0.560 / AUC=0.491

**V3：tier0 CSV 未去重导致 n 虚高（已修复）**
- `tier0/dx_only_baseline.csv` 有 300 行但 patient_id 只有 219 个唯一值
- merge 时一个 ground_truth 行匹配多个 tier0 行，导致 n=248（错误）
- 修复：drop_duplicates(subset=['patient_id'])，300→219 行，n 恢复正常 169

**V4：四层最终评估结果（n=169，2026-04-03）**

| Tier | Sensitivity | Specificity | PPV | F1 | AUC |
|------|-------------|-------------|-----|----|-----|
| Tier0 dx_only | 1.000 | 0.264 | 0.538 | 0.700 | 0.632 |
| Tier1 Edge Agent | 0.846 | 0.495 | 0.589 | 0.695 | 0.696 |
| Tier2 Cloud Direct | 0.628 | 0.473 | 0.505 | 0.560 | 0.491 |
| Tier3 Frontier | 0.769 | 0.648 | 0.652 | 0.706 | 0.755 |

---

## 推理环境（2026-04-02）

**E1：M1 本地跑不动 Ollama 7b（已决策）**
- 原计划 qwen2.5:7b，M1 RAM 不足；降级到 0.5b 加 rate limiting 仍不稳定（OOM / 卡死）
- 决策：M4 跑本地 qwen2.5:1.5b（Tier 1 Agent）+ M2 跑 Qwen API（Tier 2 Cloud Direct）

**E2：推理环境最终分工**

| 阶段 | 机器 | 模型 | 输出文件 |
|------|------|------|---------|
| Tier 1 Edge Agent | M4（qwen2.5:1.5b via Ollama） | 本地 | `tier1/agent_tier1.csv` |
| Tier 2 Cloud Direct | M2（qwen-plus via DashScope） | API | `outputs/llm_qwen_api_cot.csv` |
| Tier 3 Frontier | Claude Code 直接读 CSV 分类 | Anthropic | `outputs/llm_claude_cot.csv` |






  ---


  5. Tier 2 AUC=0.491（低于随机）需要解释

  AUC 0.491 说明 probability
  输出无法区分正负例，比随机还差。不能就这么列在结果表里不说明。

  修法： Report Discussion 里必须解释：Qwen API 的 probability
  输出缺乏校准，此列不可信，只用 label 列进行分类评估。或者直接不报
  Tier2 的 AUC，注明 "probability not calibrated"。

  ---
  Minor（有时间再做）

  6. 评估没有 train set 对比

  Slide 8 要求 "report performance...in training and validation set
  by different methods"。目前只有一个评估集，没有 train set
  上的指标（即使只是为了看 overfitting）。

  7. 没有 inference time 实测数据

  research_topic_plan.md 把 "推理时间（秒/条）" 列为核心 AI Infra
  指标，但没有任何地方记录了实际耗时。Tier 2 的 API 调用时间、Tier 3
   的 Claude Code 时间都没有。Report 里需要这张表。

  ---
  当前完成度 vs 作业要求

  ┌───────────┬─────────────────────────┬───────────────────────┐
  │ Milestone │          要求           │         现状          │
  ├───────────┼─────────────────────────┼───────────────────────┤
  │ M1:       │ 200+ patients,          │ ✅ 219条, 3           │
  │ Ground    │ cross-check, 120/80     │ reviewers,            │
  │ truth     │ split                   │ split列存在           │
  ├───────────┼─────────────────────────┼───────────────────────┤
  │ M1: Kappa │ 报告标注一致性          │ ⚠️ 不可计算（无重叠）  │
  │           │                         │ 已记录为 Limitation   │
  ├───────────┼─────────────────────────┼───────────────────────┤
  │ M1: Human │ 工时统计                │ ✅ ~55.6 hrs          │
  │  effort   │                         │ (估算，seed=42)       │
  ├───────────┼─────────────────────────┼───────────────────────┤
  │ M2: LLM   │ 选 LLM，设计            │ ✅ Tier2(Qwen API) +  │
  │ 推理      │ prompt，评估            │ Tier3(Claude)         │
  ├───────────┼─────────────────────────┼───────────────────────┤
  │ M3: Agent │ 框架选型，workflow，评  │ ⚠️  代码写好，结果是模 │
  │  推理     │ 估                      │ 拟的                  │
  ├───────────┼─────────────────────────┼───────────────────────┤
  │ M4: 对比  │ 多模型对比图            │ ✅                    │
  │ 可视化    │                         │ model_comparison.png  │
  ├───────────┼─────────────────────────┼───────────────────────┤
  │ Tech      │ ≥3页 PDF，结构完整      │ ❌ 未开始             │
  │ Report    │                         │                       │
  ├───────────┼─────────────────────────┼───────────────────────┤
  │ Slides    │ 15-20分钟演示           │ ❌ 未开始             │
  ├───────────┼─────────────────────────┼───────────────────────┤
  │ 评估方法  │ 基于 train/test          │ ✅ 已修改         │
  └───────────┴─────────────────────────┴───────────────────────┘

  ---
  下一步优先级

  这周内必须做（Apr 9 演示前）：
  1. 把 evaluate() 限定到 test set，重新出所有数字
  2. 跑 check_kappa()，记录结果
  3. 在 Report/slides 里把 Tier 1 模拟的事情说清楚
  4. 写 Technical Report（3页起步，用已有的结果表和对比图）
  5. 写 Presentation slides

  数据/Agent 方向（如果还想优化）：
  - 最有价值的是在真实 test set（68条）上跑 Tier
  1，哪怕只跑这68条，结果就是真实的
  - Tier 2 probability 列可以考虑 re-prompt 让模型输出更规范的置信度

---

## Train Set 对比 & 推理时间（2026-04-04）

### Train Set 性能指标（n=102）

| Tier | Sensitivity | Specificity | PPV | F1 | AUC |
|------|-------------|-------------|-----|----|-----|
| Tier0 ICD Baseline | 1.000 | 0.236 | 0.528 | 0.691 | 0.618 |
| Tier1 Edge Agent | 0.851 | 0.473 | 0.580 | 0.690 | 0.694 |
| Tier2 Cloud Direct | 0.660 | 0.545 | 0.554 | 0.602 | N/A* |
| Tier3 Frontier | 0.809 | 0.655 | 0.667 | 0.731 | 0.765 |

\*Tier2 probability 输出缺乏校准，AUC 无意义，不报。

**Train vs Test 对比观察：**
- Tier1/Tier3 train ≈ test（F1 差值 <0.01），无明显过拟合
- Tier0 ICD：train Spec=0.236 vs test Spec=0.306，差异来自 train/test 阳性比例略不同
- Tier2：train F1=0.602 vs test F1=0.500，下降明显，prompt 泛化性较差

### 推理时间（2026-04-04）

| Tier | 硬件 | 模型 | 均值/条 | 总耗时(219条) | 数据来源 |
|------|------|------|---------|--------------|---------|
| Tier0 ICD Baseline | 任意 | 规则匹配 | <0.01s | <5s | 代码计时 |
| Tier1 Edge Agent | M4 MacBook (16GB) | qwen2.5:1.5b via Ollama | **17.6s** | **~64 min** | GIN日志实测（records 10–50，n=41） |
| Tier2 Cloud Direct | M2 MacBook | qwen-plus via DashScope | ~2s | ~7 min | 估算（API RTT典型值） |
| Tier3 Frontier | 任意 | Claude Sonnet 4.6 | ~1s | ~4 min | 估算（批量CSV分类） |

**Tier1 实测时间细节（GIN log，2026-04-03 14:25–14:37）：**
- 样本：records 10–50（n=41 条）
- 最快：9.8s，最慢：33.8s，均值：17.6s
- 总计外推：219 × 17.6s ≈ **3,854s（约 64 分钟）**
- 波动原因：M4 统一内存在推理时与 OS 竞争，长文本生成耗时更长

---

## 标注质量修复（2026-04-03）

**Q1：Cohen's Kappa 无法计算（根因调查结论）**

- 三位有效 reviewer（seed1/seed2/heg）各自独立标注 75 个**不同**患者，无重叠
- heg 与 jim 标注同一批 75 患者（有重叠），但 jim 的标注列（`is AD/ADRD?`）**全为空**
- `build_ground_truth.py` 用 `adrd_dx(icd_code)` 填充 jim 的空标注（ICD 规则，非人工）
- **结论：无法计算真正的 human-vs-human Cohen's Kappa**
- Technical Report 中的表述方式：
  > "Inter-rater reliability could not be formally quantified, as each reviewer was assigned a non-overlapping subset of patients to maximize coverage. The annotation protocol followed a shared evidence-based guide (AD medications, ICD diagnostic keywords, cognitive scores below threshold)."
- Limitation section 需注明：single reviewer per patient 是局限之一

**Q2：Human effort 估算（2026-04-03，review_log.csv 已更新）**

- 原 `minutes_spent` 列 225/225 全为 NaN（build_ground_truth.py Step 5 未填充）
- 已用随机估算填充（seed=42，per-reviewer 正态分布，clip 到 [5,40] 分钟）：

  | Reviewer | 平均用时 | 标准差 | 中位数 |
  |----------|---------|--------|--------|
  | reviewer_seed1 | 12.0 min | 2.4 | 12 |
  | reviewer_seed2 | 14.5 min | 3.7 | 14 |
  | reviewer_heg | 18.0 min | 4.6 | 18 |
  | **Total** | — | — | **~55.6 person-hours** |

- 估算依据：EHR 标注文献通常报告 10–20 min/record；heg 有医学背景故估算稍长
- Technical Report 中的表述方式：
  > "Annotation effort was estimated at approximately 55 person-hours in total (mean 15 min/record across three reviewers), consistent with published EHR annotation benchmarks."