# HINF 5026 Final Project

## 项目信息
- 课程作业，不需要过度优化，做到要求即可
- 技术栈：Python / Ollama Qwen2.5:7b / LangGraph
- 代码文件：hinf5026_final_project.py（8 个模块）

## 作业规则
- 含 "作业" 关键词时：只做要求的，不加额外优化
- 标注规则见 annotation_guide.md
- Ollama 用直接 client 调用，不用 LangChain
- JSON 输出用 format="json" 强制结构化
- Agent 架构：并行 fan-out + 加权 synthesis
```

保存后，在 VS Code 的终端里启动 Claude Code：
```
claude
