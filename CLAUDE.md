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

---

## 待办事项

### 紧急（本周内）
- [ ] jim 补全标注：reviewer_jim 的 75 条 y_true 全为空，填写后重跑 build_ground_truth.py
- [ ] adrd_dx(final) 补填：4 个文件该列全为空，需从 ICD 码推导，作为 dx_only baseline

### Milestone 1 收尾（3/29–3/30）
- [ ] 跑 check_kappa：heg vs jim 75 个重叠病人，目标 κ ≥ 0.8
- [ ] 确认有效标注数量（去掉 jim NaN 后约 150 条，需补至 200）
- [ ] 划分 train(120)/test(80)，更新 patient_index.csv 的 split 列

### Milestone 2：LLM 推理（4/2–4/4）
- [ ] 安装启动 Ollama：`brew install ollama && ollama pull qwen2.5:7b`
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
