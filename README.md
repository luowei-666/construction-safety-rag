# 智安查 —— 智慧工地安全规范智能问答与风险预警助手

面向智能建造场景的 RAG（检索增强生成）应用：基于本地建造安全法规标准知识库，实现安全规范自然语言问答、隐患风险结构化分析、安全检查报告一键生成。

> 参赛作品：第八届 AIC 全球校园人工智能算法精英大赛 · 人工智能+场景建造创新大赛 · 创新应用（青创学生组）

## 功能特性

- 📚 **垂直知识库**：内置《建设工程安全生产管理条例》（国务院令393号）、《危险性较大的分部分项工程安全管理规定》（住建部令37号）、《建筑施工安全检查标准》JGJ59-2011 要点、典型安全隐患案例库
- 🔍 **混合检索**：向量检索（text-embedding-v3 + FAISS）与关键词检索（jieba + BM25）RRF 融合，兼顾语义与精确词召回
- 💬 **规范问答**：流式输出、引用来源可追溯、基于对话历史的 Query 改写
- 🚨 **隐患风险结构化分析**：输入现场隐患描述 → 输出风险等级（重大/一般/无隐患）+ 依据条款 + 整改建议
- 📄 **安全检查报告生成**：一键汇总生成《施工安全检查报告》（Markdown）
- ⚙️ **参数可调**：Top-K、距离阈值、上下文窗口、混合权重、检索模式切换
- 📊 **量化评估**：12 题建造领域测试集，Recall@3/5 = 100%，回答相似度 0.887

## 技术栈

| 环节 | 技术 |
| --- | --- |
| 文本向量化 | 通义千问 text-embedding-v3 |
| 关键词检索 | jieba + rank_bm25 |
| 向量库 | FAISS (IndexFlatL2) |
| 大模型 | qwen-turbo（流式） |
| 网页端 | Gradio 6.26 |
| 评估 | 自研 evaluate.py |

## 快速开始

```bash
# 1. 安装依赖
pip install faiss-cpu numpy requests python-dotenv PyPDF2 jieba rank_bm25 gradio

# 2. 配置 API Key（复制 .env.example 为 .env 并填入）
# DASHSCOPE_API_KEY=你的通义千问APIKey

# 3. 构建向量库并启动网页
python rag_webui.py
# 浏览器访问 http://127.0.0.1:7860
```

## 项目结构

```
├── rag_core.py              # 核心引擎：分块/向量化/混合检索/风险分析
├── rag_webui.py             # Gradio 网页端
├── evaluate.py              # 评估脚本（--compare --with-answer）
├── test_set_construction.json  # 建造领域评估测试集
├── demo_scenarios.md        # 演示视频脚本
├── 技术报告大纲.md           # 参赛技术报告大纲（按官方要求）
└── docs/                    # 知识库文档（法规全文/标准要点/案例库）
```

## 运行评估

```bash
python evaluate.py --test-set test_set_construction.json --compare --with-answer
```

## 免责声明

知识库中法规条文来自政府公开渠道，隐患案例为基于公开规范的演示数据；系统分析结果仅供现场检查参考，最终结论须以专业人员复核为准。
