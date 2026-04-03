@~/claude-os/CLAUDE.md

# HINF 5026 Final Project

## 项目信息
- 课程作业，不需要过度优化，做到要求即可
- 技术栈：Python / Ollama Qwen2.5:1.5b / LangGraph
- 代码文件：`hinf5026_final_project.py`（8 个模块）
- GitHub：https://github.com/Leoding192/hinf5026-final-project（private）

## 作业规则
- 含 "作业" 关键词时：只做要求的，不加额外优化
- 标注规则见 `annotation_guide.md`
- Ollama 用直接 client 调用，不用 LangChain
- JSON 输出用 `format="json"` 强制结构化
- Agent 架构：ICD/Med/Note 并行 fan-out + 加权 synthesis（ICD 0.5 / Med 0.3 / Note 0.2）

## 关键路径 & 数据
| 文件 | 说明 |
|------|------|
| `data/patient_notes/patient_notes.csv` | 300 行原始文本，`_load_notes()` 去重后 219 条 |
| `outputs/ground_truth.csv` | y_true: 1=83 / 0=92 / -1=50（评估时过滤 -1 和 NaN）|
| `outputs/dx_only_baseline.csv` | ICD 规则 baseline：灵敏度 100%，特异度 42.7% |
| `tier1/agent_tier1.csv` | Tier 1 Agent 结果（M4 跑，待收）|
| `outputs/llm_qwen_api_cot.csv` | Tier 2 Cloud 结果（M2 跑，待收）|

## 推理架构
| Tier | 机器 | 模型 | 隐私 |
|------|------|------|------|
| Tier 0 Baseline | 任意 | ICD-10 规则 | ✅ |
| Tier 1 Edge Agent | M4 | qwen2.5:1.5b via Ollama | ✅ 本地 |
| Tier 2 Cloud Direct | M2 | qwen-plus via DashScope | ⚠️ 阿里云 |
| Tier 3 Frontier | 任意 | Claude Code 直接读 CSV | ⚠️ Anthropic |

环境变量：`DASHSCOPE_API_KEY`（Tier 2 需要，无 key 则自动跳过）

## 问题 & 决策日志
见 [`project_log.md`](project_log.md)

---

## 待办事项

### 进行中（等待结果）
- [ ] 等待 M4 结果：`tier1/agent_tier1.csv`
- [ ] 等待 M2 结果：`outputs/llm_qwen_api_cot.csv`
- [ ] 收到结果后跑 `compare_all_models()`，输出 `model_comparison.png`

### Milestone 4：对比 & 可视化
- [ ] 评估 Tier 1 / Tier 2 性能（AUC / Sensitivity / Specificity / F1）
- [ ] compare_all_models()：Tier 0–2 对比图

### 最终交付（截止 Apr 16）
- [ ] Technical report（3–8 页 PDF）
- [ ] Presentation slides（15-20 分钟，Apr 9 演示）
- [ ] 所有代码/数据/输出打包 zip 提交
