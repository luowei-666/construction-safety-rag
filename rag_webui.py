"""rag_webui.py — RAG 知识库问答网页界面（Gradio 6.26）

基于 rag_core 公共模块，提供：上传文档、向量库管理、混合检索、流式问答、
上下文窗口限制、对话导出、检索模式切换。
"""
import os
import shutil
import datetime
import gradio as gr

from rag_core import (
    DOC_FOLDER,
    VECTOR_INDEX_PATH,
    CHUNKS_PATH,
    META_PATH,
    build_vector_from_docs,
    hybrid_search,
    rewrite_query,
    truncate_messages,
    dashscope_chat_stream,
    analyze_risk,
    DASHSCOPE_API_KEY,
)

EXPORT_FILE = "chat_history_export.md"
RISK_REPORT_FILE = "safety_check_report.md"

if not os.path.exists(DOC_FOLDER):
    os.mkdir(DOC_FOLDER)

print(f"读取api_key: {DASHSCOPE_API_KEY[:10]}***")

g_index = None
g_chunks_with_source = None
g_bm25 = None
risk_records = []  # 本次会话的风险分析记录，用于生成安全检查报告


def chat_handler(message, chat_history, top_k, dist_threshold, max_rounds, use_hybrid):
    global g_index, g_chunks_with_source, g_bm25
    try:
        if g_index is None:
            new_history = chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "向量库未初始化，请先上传文档并重建向量库。"}
            ]
            yield new_history, ""
            return

        # 完整历史（UI 展示用）
        full_history = chat_history + [{"role": "user", "content": message}]
        # 截断后的历史（LLM 用，节省 token）
        llm_history = truncate_messages(full_history, max_rounds)

        rewritten_q = rewrite_query(llm_history, message)
        print(f"改写query: {rewritten_q}")
        final_hits = hybrid_search(
            g_index, g_chunks_with_source, g_bm25,
            rewritten_q, top_k=top_k,
            dist_threshold=dist_threshold,
            use_hybrid=use_hybrid,
        )
        print(f"召回数量: {len(final_hits)} (模式: {'混合' if use_hybrid else '纯向量'})")

        if len(final_hits) == 0:
            source_text = "\n> 无匹配文档片段"
            sys_content = "没有从文档中检索到相关信息，请更换问题或者补充文档内容。"
        else:
            source_text = ""
            for idx, item in enumerate(final_hits):
                source_text += f"\n---\n**来源：{item['source']}**\n> {item['text'][:250]}"
            context_parts = [item["text"] for item in final_hits]
            context = "\n---\n".join(context_parts)
            sys_content = f"""基于下面参考文档回答用户问题，只使用文档内信息。不知道就直接说不知道，禁止编造。
【参考文档】
{context}
"""

        # 组装大模型消息：system + 截断历史（已含当前用户消息）
        messages = [{"role": "system", "content": sys_content}] + llm_history

        # UI 用完整历史，追加空assistant消息用于流式更新
        display_history = full_history + [{"role": "assistant", "content": ""}]
        yield display_history, ""

        accumulate = ""
        for chunk in dashscope_chat_stream(messages):
            accumulate += chunk
            display_history[-1]["content"] = accumulate
            yield display_history, ""

        display_history[-1]["content"] = accumulate + "\n\n### 引用片段" + source_text
        yield display_history, ""

    except Exception as e:
        err_msg = f"异常: {str(e)}"
        print(err_msg)
        new_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": err_msg}
        ]
        yield new_history, ""


def upload_file_handler(files):
    global g_index, g_chunks_with_source, g_bm25
    if not files:
        return "未选择任何文件"
    success_count = 0
    for tmp_path in files:
        fname = os.path.basename(tmp_path)
        if not fname.lower().endswith((".txt", ".md", ".pdf")):
            continue
        save_path = os.path.join(DOC_FOLDER, fname)
        shutil.copy(tmp_path, save_path)
        success_count += 1
    if success_count == 0:
        return "没有可处理的txt/md/pdf文件"
    return f"成功上传{success_count}个文件，请点击【重建向量库】按钮生效"


def rebuild_btn_click(chunk_size, overlap):
    global g_index, g_chunks_with_source, g_bm25
    g_index, g_chunks_with_source, msg, g_bm25 = build_vector_from_docs(chunk_size, overlap, force_rebuild=True)
    return msg


def clear_vector_btn_click():
    """清除向量库与 BM25 缓存文件，重置索引"""
    global g_index, g_chunks_with_source, g_bm25
    removed = []
    for p in [VECTOR_INDEX_PATH, CHUNKS_PATH, META_PATH]:
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)
    from rag_core import BM25_PATH
    if os.path.exists(BM25_PATH):
        os.remove(BM25_PATH)
        removed.append(BM25_PATH)
    g_index = None
    g_chunks_with_source = None
    g_bm25 = None
    if removed:
        return "已清除缓存：\n" + "\n".join(removed) + "\n请点击【重建向量库】重新构建。"
    return "当前没有缓存文件，无需清除。"


def export_chat_handler(chat_history):
    """导出聊天记录为 Markdown 文件"""
    if not chat_history:
        return None, "暂无聊天记录可导出"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# RAG 问答记录\n\n> 导出时间：{now}\n"]
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"## 用户\n\n{content}\n")
        elif role == "assistant":
            lines.append(f"## 助手\n\n{content}\n")
    export_path = os.path.join(os.getcwd(), EXPORT_FILE)
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return export_path, f"导出成功：{EXPORT_FILE}"


# ---------- 隐患风险结构化分析（决策辅助层） ----------
RISK_LEVEL_BADGE = {
    "重大隐患": "🔴 重大隐患",
    "一般隐患": "🟠 一般隐患",
    "无隐患": "🟢 无隐患",
    "无法判断": "⚪ 无法判断",
    "解析失败": "🔘 解析失败",
}


def risk_analysis_handler(message, top_k, dist_threshold, mode_value):
    global g_index, g_chunks_with_source, g_bm25
    if g_index is None:
        return "⚠️ 向量库未初始化，请先上传文档并重建向量库。"
    if not message or not message.strip():
        return "请先描述施工现场情况，例如：六米深的基坑没有做专项施工方案就直接开挖，坑边堆土很近。"
    use_hybrid = mode_map(mode_value)
    try:
        result = analyze_risk(
            message, g_index, g_chunks_with_source, g_bm25,
            top_k=top_k, dist_threshold=dist_threshold, use_hybrid=use_hybrid,
        )
    except Exception as e:
        print(f"风险分析异常: {e}")
        return f"❌ 分析失败：{e}"

    level = result.get("risk_level", "无法判断")
    badge = RISK_LEVEL_BADGE.get(level, f"⚪ {level}")
    lines = [f"### {badge}"]
    summary = result.get("summary", "")
    if summary:
        lines.append(f"\n**风险分析**：{summary}")
    evidence = result.get("evidence", [])
    if evidence:
        lines.append("\n**依据条款**：")
        for e in evidence:
            lines.append(f"- {e}")
    suggestions = result.get("suggestions", [])
    if suggestions:
        lines.append("\n**整改建议**：")
        for i, s in enumerate(suggestions, 1):
            lines.append(f"{i}. {s}")
    sources = result.get("_sources", [])
    if sources:
        lines.append("\n> 引用来源：" + "、".join(f"`{s}`" for s in sorted(set(sources))))

    # 记录到会话报告（供"生成安全检查报告"使用）
    risk_records.append({
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "desc": message.strip(),
        "level": level,
        "summary": summary,
        "evidence": evidence,
        "suggestions": suggestions,
        "sources": sorted(set(sources)),
    })
    return "\n".join(lines)


def render_risk_report(records, now):
    """把风险分析记录渲染为《施工安全检查报告》Markdown 文本（纯函数，可测试）"""
    lines = []
    lines.append("# 施工安全检查报告（AI 辅助生成）")
    lines.append("")
    lines.append(f"> 报告生成时间：{now}")
    lines.append("> 生成方式：本系统基于规范知识库混合检索 + 大模型结构化分析自动生成，供现场检查参考，最终结论以专业人员复核为准。")
    lines.append("")
    lines.append("## 一、检查基本信息")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("| --- | --- |")
    lines.append("| 项目名称 | （待填写） |")
    lines.append("| 施工单位 | （待填写） |")
    lines.append("| 检查部位/工序 | （待填写） |")
    lines.append(f"| 检查时间 | {now} |")
    lines.append("| 隐患分析条数 | {} 条 |".format(len(records)))
    lines.append("")
    lines.append("## 二、隐患清单")
    lines.append("")
    lines.append("| 序号 | 隐患描述 | 风险等级 | 涉及规范来源 |")
    lines.append("| --- | --- | --- | --- |")
    for i, rec in enumerate(records, 1):
        src = "、".join(rec["sources"]) if rec["sources"] else "-"
        desc = rec["desc"].replace("|", "/")
        lines.append(f"| {i} | {desc} | {rec['level']} | {src} |")
    lines.append("")
    lines.append("## 三、隐患详情与整改要求")
    lines.append("")
    for i, rec in enumerate(records, 1):
        lines.append(f"### 隐患 {i}：{rec['desc']}")
        lines.append("")
        lines.append(f"- **风险等级**：{rec['level']}")
        lines.append(f"- **风险分析**：{rec['summary']}")
        if rec["evidence"]:
            lines.append("- **依据条款**：")
            for e in rec["evidence"]:
                lines.append(f"  - {e}")
        if rec["suggestions"]:
            lines.append("- **整改建议**：")
            for s in rec["suggestions"]:
                lines.append(f"  - {s}")
        lines.append("")
    lines.append("## 四、检查结论")
    lines.append("")
    serious = sum(1 for r in records if r["level"] == "重大隐患")
    if serious:
        lines.append(f"本次检查共发现隐患 {len(records)} 条，其中重大隐患 {serious} 条。重大隐患应立即停工整改，整改完成经验收合格后方可复工；其余隐患应限期整改。")
    else:
        lines.append(f"本次检查共发现隐患 {len(records)} 条，均为一般隐患，应限期整改并复查闭环。")
    lines.append("")
    lines.append("## 五、签字确认")
    lines.append("")
    lines.append("| 角色 | 签字 | 日期 |")
    lines.append("| --- | --- | --- |")
    lines.append("| 检查人 | | |")
    lines.append("| 整改责任人 | | |")
    lines.append("| 复查人 | | |")
    return "\n".join(lines)


def generate_report_handler():
    """把本次会话的风险分析记录汇总为《施工安全检查报告》Markdown"""
    if not risk_records:
        return None, "暂无可生成报告的隐患分析记录，请先进行风险分析。"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = render_risk_report(risk_records, now)
    report_path = os.path.join(os.getcwd(), RISK_REPORT_FILE)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    return report_path, f"安全检查报告已生成：{RISK_REPORT_FILE}"


CUSTOM_CSS = """
#app-header {
  text-align: center;
  padding: 0.8rem 0 0.2rem 0;
}
#app-header h1 {
  font-size: 2.1rem;
  font-weight: 800;
  margin-bottom: 0.2rem;
  background: linear-gradient(90deg, #4f46e5, #9333ea, #db2777);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
#app-header p {
  color: #6b7280;
  font-size: 0.95rem;
  margin-top: 0;
}
.side-card, .chat-card {
  background: rgba(255,255,255,0.88);
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 0.8rem 1rem;
  box-shadow: 0 4px 18px rgba(79,70,229,0.07);
}
.side-card h3 {
  color: #4f46e5;
  font-size: 1.02rem;
  font-weight: 700;
  margin: 0.2rem 0 0.7rem 0;
  padding-bottom: 0.45rem;
  border-bottom: 2px solid #eef2ff;
}
body {
  background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 55%, #fdf4ff 100%) fixed !important;
}
.gr-button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: transform 0.1s ease;
}
.gr-button:hover {
  transform: translateY(-1px);
}
footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Microsoft YaHei", "sans-serif"],
)

with gr.Blocks(title="RAG 知识库问答") as demo:
    gr.HTML("""
    <div id="app-header">
      <h1>📚 RAG 知识库问答</h1>
      <p>多文档检索增强生成 · 混合检索 · 流式输出 · 来源可追溯</p>
    </div>
    """)

    with gr.Row():
        # 左侧参数面板
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes="side-card"):
                gr.Markdown("### ⚙ 检索参数")
                chunk_size_slider = gr.Slider(minimum=100, maximum=1000, value=350, step=50, label="Chunk 大小")
                overlap_slider = gr.Slider(minimum=0, maximum=200, value=60, step=10, label="重叠 Overlap")
                top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top-K 召回")
                dist_slider = gr.Slider(minimum=0.2, maximum=1.5, value=0.85, step=0.05, label="L2 距离阈值")
                max_rounds_slider = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="上下文窗口（轮）")
                mode_dropdown = gr.Dropdown(
                    choices=["混合检索（向量+关键词）", "仅向量检索"],
                    value="混合检索（向量+关键词）",
                    label="检索模式",
                )

            with gr.Group(elem_classes="side-card"):
                gr.Markdown("### 🗂 向量库管理")
                rebuild_btn = gr.Button("重建向量库", variant="primary")
                clear_btn_2 = gr.Button("清除向量库缓存", variant="secondary")
                rebuild_info = gr.Textbox(label="状态", interactive=False)

            with gr.Group(elem_classes="side-card"):
                gr.Markdown("### 📂 上传文档")
                gr.Markdown("支持 `.txt` / `.md` / `.pdf`，上传后点击上方【重建向量库】")
                upload_files = gr.File(file_types=[".txt", ".md", ".pdf"], file_count="multiple")
                upload_info = gr.Textbox(label="上传状态", interactive=False)

        # 右侧聊天区
        with gr.Column(scale=2):
            with gr.Group(elem_classes="chat-card"):
                chatbot = gr.Chatbot(height=520, buttons=["copy"])
                msg_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入你的问题",
                    lines=1
                )
                with gr.Row():
                    clear_btn = gr.Button("清空对话", variant="secondary")
                    export_btn = gr.Button("导出对话记录", variant="secondary")
                    export_file = gr.File(label="下载导出的记录")

            with gr.Accordion("🔍 隐患风险分析（结构化输出）", open=False):
                risk_input = gr.Textbox(
                    label="描述施工现场情况",
                    placeholder="例如：六米深的基坑没有做专项施工方案就直接开挖，坑边堆土很近",
                    lines=2,
                )
                risk_btn = gr.Button("开始风险分析", variant="primary")
                risk_output = gr.Markdown()
                with gr.Row():
                    report_btn = gr.Button("生成安全检查报告", variant="secondary")
                    report_file = gr.File(label="下载安全检查报告")

    # 事件绑定
    def mode_map(x):
        return x == "混合检索（向量+关键词）"

    upload_files.upload(upload_file_handler, inputs=[upload_files], outputs=[upload_info])
    rebuild_btn.click(rebuild_btn_click, inputs=[chunk_size_slider, overlap_slider], outputs=[rebuild_info])
    clear_btn_2.click(clear_vector_btn_click, outputs=[rebuild_info])

    msg_input.submit(
        fn=chat_handler,
        inputs=[msg_input, chatbot, top_k_slider, dist_slider, max_rounds_slider, mode_dropdown],
        outputs=[chatbot, msg_input]
    )
    clear_btn.click(lambda: [], None, chatbot)
    export_btn.click(export_chat_handler, inputs=[chatbot], outputs=[export_file, rebuild_info])
    risk_btn.click(
        risk_analysis_handler,
        inputs=[risk_input, top_k_slider, dist_slider, mode_dropdown],
        outputs=[risk_output],
    )
    report_btn.click(generate_report_handler, outputs=[report_file, risk_output])


g_index, g_chunks_with_source, init_msg, g_bm25 = build_vector_from_docs(chunk_size=350, overlap=60)
print(init_msg)

print("网页服务启动，浏览器访问: http://127.0.0.1:7860")
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    show_error=True,
    theme=theme,
    css=CUSTOM_CSS,
)
