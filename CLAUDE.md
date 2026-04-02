# HINF 5026 Final Project

## 项目信息
- 课程作业，不需要过度优化，做到要求即可
- 技术栈：Python / Ollama Qwen2.5:7b / LangGraph
- 代码文件：hinf5026_final_project.py（8 个模块）
- GitHub：https://github.com/Leoding192/hinf5026-final-project（private）
- Skills 路径：~/.claude/skills/

## 作业规则
- 含 "作业" 关键词时：只做要求的，不加额外优化
- 标注规则见 annotation_guide.md
- Ollama 用直接 client 调用，不用 LangChain
- JSON 输出用 format="json" 强制结构化
- Agent 架构：并行 fan-out + 加权 synthesis

---

## 已完成

### 环境 & 依赖
- [x] 安装所有 Python 依赖：scikit-learn, ollama, langgraph, matplotlib, python-pptx
- [x] VS Code Python 解释器指向 /usr/bin/python3 (3.9.6)，Pylance 报错已解决

### 代码框架
- [x] hinf5026_final_project.py 8 个模块骨架完成：
  - Module 1: EHR 文本预处理（extract_relevant_text）
  - Module 2: 标注工具（create_annotation_template, check_kappa）
  - Module 3: 评估指标（evaluate: AUC/Precision/Recall/F1）
  - Module 4–7: Ollama 推理、LangGraph Agent、批量运行（骨架已写）
  - Module 8: 模型对比可视化（compare_all_models）

### qwen2.5:0.5b 优化（2026-04-02）
- [x] **rate limiting**：增加动态 sleep
  - `run_batch_inference()`：0.5b 模型用 1.0s sleep（之前 0.3s）
  - `run_agent_batch()`：并行 3 agents 每条 1.0s（防止 Ollama 内存溢出）
- [x] **model 参数支持**：run_agent_batch() 现在接受 model 参数，可动态切换
- [x] **自动列名识别**：run_batch_inference/run_agent_batch 自动识别 patient_id / text 列
- [x] **代码清理**：删除 __main__ 里重复定义的 run_agent_batch()
- 原因：0.5b 模型资源有限，60+ 条数据时无速率限制会导致并发请求溢出、卡死

### 标注规则
- [x] annotation_guide.md 完成（标注格式、evidence_type 分类、Label 规则、ICD-10 速查表）

### Milestone 1：Ground Truth 构建
- [x] build_ground_truth.py 跑通，生成：
  　　→ 数据整合脚本。读取 4 个标注者的原始文件（CSV/Excel），统一格式、提取关键列、输出标准化文件。后续每次有新标注数据加入时重跑此脚本即可更新所有输出。
  - data/patient_index.csv（301 行，220 unique patients）
    　　→ 病人总目录。记录每条记录来自哪个文件、是否有临床文本、是否有 ICD 码、train/test 划分。后续划分训练集/测试集时在这里改 split 列（目前全为 "unassigned"）。
  - data/patient_notes/patient_notes.csv（300 行）
    　　→ 纯净临床文本表。只保留 patient_id + note_text，是 Module 1（extract_relevant_text）和 LLM 推理（Module 4）的直接输入。
  - outputs/ground_truth.csv（301 行）
    　　→ 人工标注结果表。核心字段 y_true（1/0/-1）是所有模型评估的"正确答案"。Module 3 的 evaluate() 函数用这里的 y_true 计算 AUC/PPV/Recall/F1。dx_only_label 列留给 ICD-only baseline（目前为空，待补填）。
  - outputs/review_log.csv（301 行）
    　　→ 标注过程记录表。记录谁标了哪条、标了多少分钟、标的结论是什么。用于 Technical Report 里汇报人工标注工时，也是 check_kappa() 的数据来源。
- [x] 数据质量报告：y_true 分布 1:83 / 0:92 / -1:50 / NaN:76（jim 未填）
  　　→ 76 条 NaN 全是 jim 文件，jim 补完后重跑 build_ground_truth.py 即可更新。-1（uncertain）暂保留，模型训练时可选择排除或单独处理。
- [x] Kappa 候选：heg 和 jim 有 75 个重叠 subject_id
  　　→ 这 75 人被两名标注者都标注过，可直接用 check_kappa(heg_file, jim_file) 计算 Cohen's Kappa。等 jim 补完标注后运行，目标 κ ≥ 0.8；不达标则开会对齐标注规则后重标。

### 仓库
- [x] GitHub 仓库：https://github.com/Leoding192/hinf5026-final-project（private，main）
- [x] 12 个文件全部 push

### 紧急（本周内）
- [x] jim 补全标注：reviewer_jim 的 75 条 y_true 全为空，填写后重跑 build_ground_truth.py
- [x] adrd_dx(final) 补填：4 个文件该列全为空，需从 ICD 码推导，作为 dx_only baseline

### Milestone 1 收尾（3/29–3/30）
- [x] 跑 check_kappa：heg vs jim 75 个重叠病人，目标 κ ≥ 0.8
- [x] 确认有效标注数量（去掉 jim NaN 后约 150 条，需补至 200）
- [x] 划分 train(120)/test(80)，更新 patient_index.csv 的 split 列

## 待办事项

### Milestone 2：LLM 推理（4/2–4/4）
- [ ] 安装启动 Ollama：`brew install ollama && ollama pull 1`
- [ ] 实现 ollama_client.py（Module 4）：prompt 设计 + JSON 输出
- [ ] 跑推理，生成 outputs/llm_predictions.csv
- [ ] 评估 LLM 性能，对比 dx_only baseline

### Milestone 3：Agent 架构
- [ ] 实现 LangGraph Agent（Module 5-6）：并行 fan-out + 加权 synthesis
- [ ] 跑 Agent 推理，生成 outputs/agent_predictions.csv
- [ ] 评估 Agent 性能

### Milestone 4：对比 & 可视化
- [ ] compare_all_models()：汇总 dx_only / LLM / Agent，输出 model_comparison.png

### 最终交付（截止 Apr 16）
- [ ] Technical report（3–8 页 PDF）
- [ ] Presentation slides（15-20 分钟，Apr 9 演示）
- [ ] 所有代码/数据/输出打包 zip 提交

### Project-level Skill 封装（Milestone 4 完成后）

> 将 pipeline 打包成可复用 skill，路径：`~/.claude/skills/ehr-adrd-pipeline/skill.md`

**步骤：**
- [ ] Step 1：确认接口 — 记录各模块实际入参/出参（以跑通为准，不提前设计）
- [ ] Step 2：写 skill.md（用下方草稿）
- [ ] Step 3：新对话测试触发词能否激活
- [ ] Step 4：写入 MEMORY.md 记录 skill 路径和用途

**skill.md 草稿（Milestone 4 后填入真实接口）：**

```markdown
---
name: ehr-adrd-pipeline
description: >
  ADRD detection pipeline: baseline/llm-only/agent/compare-all 四种模式。
  触发词: adrd pipeline / adrd detect / compare adrd models
triggers:
  - adrd pipeline
  - run adrd pipeline
  - adrd detect
  - ehr adrd
  - compare adrd models
  - compare baseline llm agent
---

## 前置检查（llm-only / agent / compare-all 模式必须通过）
ollama list | grep qwen2.5:7b
未找到则提示: ollama pull qwen2.5:7b

## 模式路由
| 模式         | 调用                    | 需要 Ollama |
|------------|------------------------|-----------|
| baseline   | dx_only_label 直接评估    | 否 |
| llm-only   | Module 4 ollama_client  | 是 |
| agent      | Module 5-6 LangGraph   | 是 |
| compare-all| 三种依次 + Module 8 可视化 | 是 |

## Schema 规范化（执行前必须处理）
- patient_notes.csv: note_text → ehr_text（函数入参名）
- ground_truth.csv: y_true → true_label（evaluate() 入参名）
- 过滤 true_label = -1（uncertain）和 NaN

## 输出
- outputs/llm_predictions.csv
- outputs/agent_predictions.csv
- outputs/model_comparison.png（compare-all 模式）
```
