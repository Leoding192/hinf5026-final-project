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
