# 智安查 —— 智慧工地安全规范智能问答与风险预警助手

面向智能建造场景的 RAG（检索增强生成）应用：基于本地建造安全法规标准知识库，实现安全规范自然语言问答、现场照片隐患识别、隐患风险结构化分析、含照片检查报告一键生成。

> 参赛作品：第八届 AIC 全球校园人工智能算法精英大赛 · 人工智能+场景建造创新大赛 · 创新应用（青创学生组）

## 功能特性

- 📚 **垂直知识库**：11 份权威规范与真实案例文档——《建设工程安全生产管理条例》（国务院令393号）、《危险性较大的分部分项工程安全管理规定》（住建部令37号）、《危大工程范围清单》、JGJ59/JGJ80/JGJ46/JGJ130/JGJT429 标准要点、典型安全隐患案例库、**政府真实事故调查报告案例库**、**住建厅隐患辨识图集**；语义分块后共 **701 块**，网页端支持增删文档、重建索引
- 🔍 **混合检索 + 重排**：向量检索（text-embedding-v3 + FAISS）与关键词检索（jieba + BM25）RRF 融合，再经 **gte-rerank-v2 二次精排**，兼顾语义、精确词召回与排序质量
- 💬 **规范问答**：流式输出、引用来源可追溯、基于对话历史的 Query 改写
- 📷 **多模态照片识别**：上传现场照片（脚手架、临边防护、电梯井口等），qwen-vl-plus 自动识别隐患现象并接入风险分析链路
- 🚨 **隐患风险结构化分析**：输入隐患描述或照片 → 输出风险等级（重大/一般/无隐患）+ 依据条款 + 整改建议
- 📄 **安全检查报告（PDF）**：一键生成含隐患清单、详情、风险等级、依据条款与**现场照片**的《施工安全检查报告》
- ⚙️ **参数可调**：Top-K、距离阈值、上下文窗口、混合权重、**Rerank 开关**、检索模式切换
- 📊 **量化评估**：25 道建造领域测试题（覆盖 9 部规范与政府真实事故案例），纯向量/混合/混合+Rerank 三组消融 **Recall@1/3/5 全部 100%**

## 技术栈

| 环节 | 技术 |
| --- | --- |
| 文本分块 | 语义分块（按"第X条/一、/数字./#"标题识别） |
| 文本向量化 | 通义千问 text-embedding-v3 |
| 关键词检索 | jieba + rank_bm25 |
| 混合融合 | RRF（BM25 权重 0.25） |
| 重排精排 | gte-rerank-v2（DASHSCOPE Rerank 接口） |
| 多模态识别 | qwen-vl-plus（OpenAI 兼容接口） |
| 向量库 | FAISS (IndexFlatL2) |
| 大模型 | qwen-turbo（流式） |
| 报告生成 | python-docx → Word → PDF（含照片） |
| 网页端 | Gradio 6.26 |
| 评估 | 自研 evaluate.py（三组消融对比） |

## 快速开始

```bash
# 1. 安装依赖
pip install faiss-cpu numpy requests python-dotenv PyPDF2 jieba rank_bm25 gradio python-docx pillow

# 2. 配置 API Key（复制 .env.example 为 .env 并填入）
# DASHSCOPE_API_KEY=你的通义千问APIKey

# 3. 构建向量库并启动网页
python rag_webui.py
# 浏览器访问 http://127.0.0.1:7860
```

> 提示：报告导出 PDF 依赖本机安装 Microsoft Word 或 WPS（docx → PDF 转换）。

## 项目结构

```
├── rag_core.py              # 核心引擎：语义分块/向量化/混合检索+Rerank/风险分析/多模态
├── rag_webui.py             # Gradio 网页端
├── evaluate.py              # 评估脚本（--compare 三组消融）
├── test_set_construction.json  # 建造领域评估测试集（25 题）
├── demo_scenarios.md        # 演示视频脚本
├── 技术报告大纲.md           # 参赛技术报告大纲（按官方要求）
├── 项目介绍.pdf / .docx     # 项目介绍（含系统截图）
├── requirements.txt         # 依赖清单
├── Dockerfile               # 容器化部署
├── docs/                    # 知识库文档（11 份：法规全文/标准要点/案例库/真实事故案例/隐患辨识图集）
└── demo_images/             # 演示素材（真实隐患照片、系统截图）
```

## 运行评估

```bash
# 三组消融对比：纯向量 / 混合 / 混合+Rerank
python evaluate.py --test-set test_set_construction.json --compare
```

## 免责声明

知识库中法规条文与事故案例均来自政府公开渠道（已标注来源），系统分析结果仅供现场检查参考，最终结论须以专业人员复核为准。
