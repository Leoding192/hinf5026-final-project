# HINF 5026 Final Project

## 项目信息
- 课程作业，不需要过度优化，做到要求即可
- 技术栈：Python / Ollama Qwen2.5:7b / LangGraph
- 代码文件：hinf5026_final_project.py（8 个模块）
- GitHub：https://github.com/Leoding192/hinf5026-final-project（private）

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
  - data/patient_index.csv（301 行，220 unique patients）
  - data/patient_notes/patient_notes.csv（300 行）
  - outputs/ground_truth.csv（301 行）
  - outputs/review_log.csv（301 行）
- [x] 数据质量报告：y_true 分布 1:83 / 0:92 / -1:50 / NaN:76（jim 未填）
- [x] Kappa 候选：heg 和 jim 有 75 个重叠 subject_id

### 仓库
- [x] GitHub 仓库：https://github.com/Leoding192/hinf5026-final-project（private，main）
- [x] 12 个文件全部 push

---

## 待办事项

### 紧急（本周内）
- [ ] jim 补全标注：reviewer_jim 的 75 条 y_true 全为空，联系 jim 填写后重跑 build_ground_truth.py
- [ ] 邀请 collaborators：需要 maojiaqi1128 / heg4007 的 GitHub 用户名，在仓库 Settings → Collaborators 添加
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
