# 智安查 · 智慧工地安全规范智能问答与风险预警助手 — Docker 部署
# 说明：本镜像提供核心 RAG 问答 + 隐患风险分析能力。
#       "生成 PDF 报告"依赖 Microsoft Word COM（仅 Windows），容器内不可用；
#       如需 PDF 导出请在装有 Word/WPS 的 Windows 主机上运行 rag_webui.py。
FROM python:3.11-slim

WORKDIR /app

# 中文字体（网页与日志显示）
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

# 运行前需通过 -e DASHSCOPE_API_KEY=xxx 传入通义千问 API Key
CMD ["python", "rag_webui.py"]
