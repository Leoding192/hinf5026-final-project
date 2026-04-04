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
| `data/patient_notes/patient_notes.csv` | 300 行原始文本，去重后 219 条 |
| `outputs/ground_truth.csv` | y_true: 1=78 / 0=91 / -1=50；split: train=102 / test=68 / uncertain=49 |
| `tier0/dx_only_baseline.csv` | ICD 规则 baseline |
| `tier1/agent_tier1.csv` | Tier 1 Edge Agent 结果（keyword 模拟，非真实 LLM） |
| `tier2/llm_qwen_cot.csv` | Tier 2 Cloud Direct 结果（qwen-plus API） |
| `tier3/llm_claude_cot.csv` | Tier 3 Frontier 结果（Claude Code） |

## 评估规则（重要）
- **评估集**：held-out test set only（`split='test'`，n≈67）— 课程要求 "held-out 80 for evaluation"
- `compare_all_models()` 已内置 split 过滤，直接调用即可
- 过滤条件：`y_true ∈ {0,1}`（排除 -1 uncertain）AND `split == 'test'`
- **不能**用 train set 指标替代 test set 指标

## Inter-rater Reliability 说明
- 3 个 reviewer 各标注 75 个不同患者（无系统性重叠），Cohen's Kappa 不适用
- Report 里说明：reviewers each covered non-overlapping subsets; formal kappa not computed
- **Human effort**：review_log.csv 的 minutes_spent 已填充（seed1≈12min, seed2≈15min, heg≈18min，total≈55.6人工时）

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

### 当前状态（2026-04-03）
- [x] Ground truth 219 条，train/test split 完成
- [x] 四层推理结果全部就绪（tier0–3）
- [x] model_comparison.png（test set, n=67）
- [x] Human effort 估算填充（review_log.csv minutes_spent，~55.6 hrs）
- [x] Cohen's Kappa 不可计算 → 记录为 Limitation（project_log.md Q1）
- [x] Train set 性能指标计算完成（project_log.md）
- [x] 推理时间实测/估算完成（Tier1: 17.6s/条实测，总计~64min）
- [ ] Technical report（3–8 页 PDF）← 下一步
- [ ] Presentation slides（ppt.md 大纲已建）← Apr 9 演示
- [ ] 最终 zip 打包提交 ← Apr 16
