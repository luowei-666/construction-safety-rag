"""rag_webui.py — RAG 知识库问答网页界面（Gradio 6.26）

基于 rag_core 公共模块，提供：上传文档、向量库管理、混合检索、流式问答、
上下文窗口限制、对话导出、检索模式切换。
"""
import os
import re
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
    analyze_image_risk,
    dashscope_asr,
    dashscope_tts,
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


MODEL_MAP = {
    "qwen-turbo（默认，快）": "qwen-turbo",
    "qwen-plus（更准）": "qwen-plus",
    "qwen-max（最强）": "qwen-max",
}


def resolve_model(x):
    return MODEL_MAP.get(x, x)


def extract_clause(text):
    """从条款文本中提取'第X条'，用于引用标题展示"""
    m = re.search(r"第[一二三四五六七八九十百零0-9.]+条", text)
    return m.group(0) if m else ""


def highlight_keywords(text, query, max_len=250):
    """在检索片段中高亮 query 的关键词（Markdown 加粗），返回截断后的高亮文本"""
    import jieba

    stop = set("的了是在和与及等个这那为对于或并且被把从向以我你他她它们之按上")
    words = [w for w in jieba.lcut(query) if len(w) >= 2 and w not in stop]
    words = sorted(set(words), key=len, reverse=True)
    snippet = text if len(text) <= max_len else text[:max_len]
    for w in words:
        if f"**{w}**" in snippet:
            continue  # 已高亮过，避免嵌套
        snippet = snippet.replace(w, f"**{w}**")
    return snippet


def chat_handler(message, chat_history, top_k, dist_threshold, max_rounds, use_hybrid, use_rerank, model_name):
    model_name = resolve_model(model_name)
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
            use_rerank=use_rerank,
        )
        print(f"召回数量: {len(final_hits)} (模式: {'混合' if use_hybrid else '纯向量'}{'+Rerank' if use_rerank else ''})")

        if len(final_hits) == 0:
            source_text = "\n> 无匹配文档片段"
            sys_content = "没有从文档中检索到相关信息，请更换问题或者补充文档内容。"
        else:
            source_text = ""
            for idx, item in enumerate(final_hits):
                clause = extract_clause(item["text"])
                title = f"`{item['source']}`" + (f" · {clause}" if clause else "")
                snippet = highlight_keywords(item["text"], rewritten_q)
                source_text += f"\n---\n📌 **依据条款**：{title}\n> {snippet}"
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
        for chunk in dashscope_chat_stream(messages, model=model_name):
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


# ---------- 知识库文件管理 ----------
def list_docs_handler():
    """列出 docs 目录下的知识库文档"""
    if not os.path.exists(DOC_FOLDER):
        return gr.Dropdown(choices=[], value=None)
    files = sorted(f for f in os.listdir(DOC_FOLDER)
                   if f.lower().endswith((".txt", ".md", ".pdf")))
    return gr.Dropdown(choices=files, value=files[0] if files else None)


def delete_doc_handler(fname):
    """删除选中的知识库文档（提示重建索引生效）"""
    if not fname:
        return list_docs_handler(), "请先在下拉框选择要删除的文档。"
    path = os.path.join(DOC_FOLDER, fname)
    if os.path.exists(path):
        os.remove(path)
        return list_docs_handler(), f"已删除：{fname}，请点击【重建索引】使索引更新。"
    return list_docs_handler(), f"文件不存在：{fname}"


def risk_analysis_handler(message, image_path, owner, deadline, top_k, dist_threshold, mode_value, use_rerank, model_name):
    model_name = resolve_model(model_name)
    global g_index, g_chunks_with_source, g_bm25
    if g_index is None:
        return "⚠️ 向量库未初始化，请先上传文档并重建向量库。"
    if (not message or not message.strip()) and not image_path:
        return "请描述施工现场情况，或上传现场照片。"
    use_hybrid = mode_map(mode_value)
    try:
        if image_path:
            result = analyze_image_risk(
                image_path, g_index, g_chunks_with_source, g_bm25,
                top_k=top_k, dist_threshold=dist_threshold, use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                model=model_name,
            )
            image_desc = result.get("_image_desc", "")
            message = f"{message.strip()}\n【照片识别】{image_desc}".strip()
        else:
            result = analyze_risk(
                message, g_index, g_chunks_with_source, g_bm25,
                top_k=top_k, dist_threshold=dist_threshold, use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                model=model_name,
            )
    except Exception as e:
        print(f"风险分析异常: {e}")
        return f"❌ 分析失败：{e}"

    level = result.get("risk_level", "无法判断")
    badge = RISK_LEVEL_BADGE.get(level, f"⚪ {level}")
    lines = [f"### {badge}"]
    if image_path and result.get("_image_desc"):
        lines.append(f"\n**照片识别**：{result['_image_desc']}")
    summary = result.get("summary", "")
    if summary:
        lines.append(f"\n**风险分析**：{summary}")
    evidence = result.get("evidence", [])
    verified_map = {v.get("text"): v.get("verified", False) for v in result.get("_verified", [])}
    if evidence:
        lines.append("\n**依据条款**：")
        for e in evidence:
            mark = "✅ 已核验" if verified_map.get(e) else "⚠️ 待复核"
            lines.append(f"- {mark} {highlight_keywords(e, message)}")
    suggestions = result.get("suggestions", [])
    if suggestions:
        lines.append("\n**整改建议**：")
        for i, s in enumerate(suggestions, 1):
            lines.append(f"{i}. {s}")
    sources = result.get("_sources", [])
    if sources:
        lines.append("\n> 引用来源：" + "、".join(f"`{s}`" for s in sorted(set(sources))))

    # 记录到会话台账（供生成安全检查报告/整改通知单/统计使用）
    risk_records.append({
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "desc": message.strip(),
        "level": level,
        "summary": summary,
        "evidence": evidence,
        "suggestions": suggestions,
        "sources": sorted(set(sources)),
        "image_path": image_path if image_path else None,
        "status": "待整改",
        "owner": (owner or "").strip() or "未指定",
        "deadline": (deadline or "").strip() or "未指定",
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
    """把本次会话的风险分析记录汇总为《施工安全检查报告》，输出 PDF"""
    if not risk_records:
        return None, "暂无可生成报告的隐患分析记录，请先进行风险分析。"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        docx_path = render_risk_report_docx(risk_records, now)
        pdf_path = docx_to_pdf(docx_path)
        if os.path.exists(docx_path):
            os.remove(docx_path)  # 仅保留 PDF
        return pdf_path, f"安全检查报告（PDF）已生成：{os.path.basename(pdf_path)}"
    except Exception as e:
        print(f"报告生成失败: {e}")
        return None, f"报告生成失败：{e}"


def render_risk_report_docx(records, now):
    """把风险分析记录渲染为《施工安全检查报告》Word 文档（python-docx）"""
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    # 页面边距
    for sec in doc.sections:
        sec.top_margin = Cm(2.2)
        sec.bottom_margin = Cm(2.2)
        sec.left_margin = Cm(2.5)
        sec.right_margin = Cm(2.5)

    # 标题
    title = doc.add_heading("施工安全检查报告（AI 辅助生成）", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 说明
    p = doc.add_paragraph()
    r = p.add_run(f"报告生成时间：{now}\n生成方式：本系统基于规范知识库混合检索 + 大模型结构化分析自动生成，供现场检查参考，最终结论以专业人员复核为准。")
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # 一、检查基本信息
    doc.add_heading("一、检查基本信息", level=1)
    base_rows = [
        ("项目名称", "（待填写）"),
        ("施工单位", "（待填写）"),
        ("检查部位/工序", "（待填写）"),
        ("检查时间", now),
        ("隐患分析条数", f"{len(records)} 条"),
    ]
    t1 = doc.add_table(rows=len(base_rows), cols=2)
    t1.style = "Table Grid"
    for i, (k, v) in enumerate(base_rows):
        t1.cell(i, 0).text = k
        t1.cell(i, 1).text = v

    # 二、隐患清单
    doc.add_heading("二、隐患清单", level=1)
    t2 = doc.add_table(rows=1 + len(records), cols=4)
    t2.style = "Table Grid"
    hdr = t2.rows[0].cells
    for j, htext in enumerate(["序号", "隐患描述", "风险等级", "涉及规范来源"]):
        hdr[j].text = htext
    for i, rec in enumerate(records, 1):
        cells = t2.rows[i].cells
        cells[0].text = str(i)
        cells[1].text = rec["desc"]
        cells[2].text = rec["level"]
        cells[3].text = "、".join(rec["sources"]) if rec["sources"] else "-"

    # 三、隐患详情与整改要求
    doc.add_heading("三、隐患详情与整改要求", level=1)
    for i, rec in enumerate(records, 1):
        doc.add_heading(f"隐患 {i}：{rec['desc']}", level=2)
        # 现场照片（如有，先用 PIL 转码为标准 JPEG 再嵌入）
        img = rec.get("image_path")
        if img and os.path.exists(img):
            try:
                from PIL import Image as PILImage
                import tempfile
                tmp = os.path.join(tempfile.gettempdir(), f"report_img_{i}.jpg")
                PILImage.open(img).convert("RGB").save(tmp, "JPEG", quality=90)
                doc.add_picture(tmp, width=Cm(14))
                if os.path.exists(tmp):
                    os.remove(tmp)
                doc.add_paragraph("▲ 现场照片")
            except Exception as e:
                print(f"图片嵌入跳过: {e}")
        doc.add_paragraph(f"风险等级：{rec['level']}")
        doc.add_paragraph(f"风险分析：{rec['summary']}")
        if rec["evidence"]:
            doc.add_paragraph("依据条款：")
            for e in rec["evidence"]:
                doc.add_paragraph(e, style="List Bullet")
        if rec["suggestions"]:
            doc.add_paragraph("整改建议：")
            for s in rec["suggestions"]:
                doc.add_paragraph(s, style="List Bullet")

    # 四、检查结论
    doc.add_heading("四、检查结论", level=1)
    serious = sum(1 for r in records if r["level"] == "重大隐患")
    if serious:
        doc.add_paragraph(
            f"本次检查共发现隐患 {len(records)} 条，其中重大隐患 {serious} 条。"
            "重大隐患应立即停工整改，整改完成经验收合格后方可复工；其余隐患应限期整改。")
    else:
        doc.add_paragraph(f"本次检查共发现隐患 {len(records)} 条，均为一般隐患，应限期整改并复查闭环。")

    # 五、签字确认
    doc.add_heading("五、签字确认", level=1)
    t3 = doc.add_table(rows=4, cols=3)
    t3.style = "Table Grid"
    for j, htext in enumerate(["角色", "签字", "日期"]):
        t3.rows[0].cells[j].text = htext
    for i, role in enumerate(["检查人", "整改责任人", "复查人"], 1):
        t3.rows[i].cells[0].text = role

    docx_path = os.path.join(os.getcwd(), "safety_check_report.docx")
    doc.save(docx_path)
    return docx_path



# ---------- 报告模板扩展（监理通知单 / 项目周报） ----------
def render_supervision_docx(records, now):
    """监理工程师通知单 Word 文档"""
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.2); sec.left_margin = Cm(2.5); sec.right_margin = Cm(2.5)
    title = doc.add_heading("监理工程师通知单（AI 辅助生成）", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run("编号：JL-%s-%s%s\n签发时间：%s\n致（施工单位）：%s" % (
        now[:10].replace("-", ""), now[11:13], now[14:16], now, records[0].get("owner", "（待填写）")))
    r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    doc.add_heading("一、通知事由", level=1)
    doc.add_paragraph("经现场检查，发现以下安全隐患，请贵单位立即组织整改：")
    doc.add_heading("二、隐患明细", level=1)
    t = doc.add_table(rows=1 + len(records), cols=4); t.style = "Table Grid"
    for j, h in enumerate(["序号", "隐患描述", "风险等级", "整改要求"]):
        t.rows[0].cells[j].text = h
    for i, rec in enumerate(records, 1):
        c = t.rows[i].cells
        c[0].text = str(i); c[1].text = rec["desc"]; c[2].text = rec["level"]
        c[3].text = "；".join(rec["suggestions"]) if rec["suggestions"] else "按相关规范整改"
    doc.add_heading("三、整改要求与复查", level=1)
    doc.add_paragraph("请于 %s 前完成整改，整改完成后报监理复查验收；逾期未整改或整改不合格的，将按合同约定及有关规定处理。" % records[0].get("deadline", "规定期限"))
    doc.add_heading("四、签字", level=1)
    t3 = doc.add_table(rows=4, cols=3); t3.style = "Table Grid"
    for j, h in enumerate(["角色", "签字", "日期"]):
        t3.rows[0].cells[j].text = h
    for i, role in enumerate(["总监理工程师/监理工程师", "施工单位负责人", "复查人"], 1):
        t3.rows[i].cells[0].text = role
    path = os.path.join(os.getcwd(), "supervision_notice.docx")
    doc.save(path); return path


def render_weekly_docx(records, now):
    """项目安全周报 Word 文档"""
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.2); sec.left_margin = Cm(2.5); sec.right_margin = Cm(2.5)
    title = doc.add_heading("项目施工安全周报（AI 辅助生成）", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run("统计截止：%s（第 %s 周）\n生成方式：本系统自动汇总本周隐患分析记录" % (now[:10], now[5:7]))
    r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    doc.add_heading("一、本周隐患总览", level=1)
    serious = sum(1 for x in records if x["level"] == "重大隐患")
    normal = sum(1 for x in records if x["level"] == "一般隐患")
    closed = sum(1 for x in records if x.get("status") == "已整改")
    t1 = doc.add_table(rows=5, cols=2); t1.style = "Table Grid"
    for i, (k, v) in enumerate([("隐患总数", "%d 条" % len(records)), ("重大隐患", "%d 条" % serious),
                                ("一般隐患", "%d 条" % normal), ("已闭环", "%d 条" % closed),
                                ("待整改", "%d 条" % (len(records) - closed))]):
        t1.cell(i, 0).text = k; t1.cell(i, 1).text = v
    doc.add_heading("二、隐患明细", level=1)
    t2 = doc.add_table(rows=1 + len(records), cols=5); t2.style = "Table Grid"
    for j, h in enumerate(["序号", "时间", "隐患描述", "风险等级", "状态"]):
        t2.rows[0].cells[j].text = h
    for i, rec in enumerate(records, 1):
        c = t2.rows[i].cells
        c[0].text = str(i); c[1].text = rec["time"][5:16]; c[2].text = rec["desc"]
        c[3].text = rec["level"]; c[4].text = rec.get("status", "待整改")
    doc.add_heading("三、本周小结", level=1)
    if serious:
        doc.add_paragraph("本周共发现重大隐患 %d 条，须立即停工整改并组织专项验收，严防群死群伤事故发生。" % serious)
    else:
        doc.add_paragraph("本周隐患均为一般隐患，已督促限期整改并复查闭环。")
    doc.add_heading("四、签发", level=1)
    t3 = doc.add_table(rows=4, cols=3); t3.style = "Table Grid"
    for j, h in enumerate(["角色", "签字", "日期"]):
        t3.rows[0].cells[j].text = h
    for i, role in enumerate(["安全员", "项目负责人", "监理"], 1):
        t3.rows[i].cells[0].text = role
    path = os.path.join(os.getcwd(), "weekly_report.docx")
    doc.save(path); return path


def generate_template_handler(template_name):
    """按模板生成 PDF：整改通知单 / 监理工程师通知单 / 项目安全周报"""
    if not risk_records:
        return None, "暂无隐患记录，请先进行风险分析。"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if template_name == "监理工程师通知单":
            docx_path = render_supervision_docx(risk_records, now)
        elif template_name == "项目安全周报":
            docx_path = render_weekly_docx(risk_records, now)
        else:  # 整改通知单
            recs = [r for r in risk_records if r.get("status", "待整改") == "待整改"] or risk_records
            docx_path = render_notice_docx(recs, now)
        pdf_path = docx_to_pdf(docx_path)
        if os.path.exists(docx_path):
            os.remove(docx_path)
        return pdf_path, "%s（PDF）已生成：%s" % (template_name, os.path.basename(pdf_path))
    except Exception as e:
        print("模板报告生成失败: %s" % e)
        return None, "模板报告生成失败：%s" % e


# ---------- 语音问答（ASR 输入 / TTS 播报） ----------
def asr_question_handler(audio_path):
    """语音转文字：识别结果同时填入识别框与风险分析输入框"""
    if not audio_path:
        return "", ""
    try:
        text = dashscope_asr(audio_path)
        return text, text
    except Exception as e:
        print("语音识别失败: %s" % e)
        return "语音识别失败：%s" % e, "语音识别失败：%s" % e


def tts_handler(text):
    """朗读风险分析结果（返回音频文件）"""
    if not text or not text.strip():
        return None
    try:
        return dashscope_tts(text[:500])
    except Exception as e:
        print("语音合成失败: %s" % e)
        return None


def docx_to_pdf(docx_path):
    """用本机 Word 将 docx 转为 PDF（Word COM）"""
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        word.DisplayAlerts = 0
        doc = word.Documents.Open(docx_path, ReadOnly=True)
        pdf_path = os.path.splitext(docx_path)[0] + ".pdf"
        doc.SaveAs2(pdf_path, FileFormat=17)  # 17 = wdFormatPDF
        doc.Close(False)
        return pdf_path
    finally:
        word.Quit()


# ---------- 隐患闭环管理（台账 / 状态流转 / 整改通知单） ----------
LEDGER_HEADERS = ["序号", "时间", "隐患描述", "风险等级", "状态", "责任人", "期限"]


def ledger_df():
    """把风险记录渲染为台账 DataFrame"""
    import pandas as pd
    if not risk_records:
        return pd.DataFrame([], columns=LEDGER_HEADERS)
    return pd.DataFrame([{
        "序号": i + 1,
        "时间": r["time"][5:16],
        "隐患描述": (r["desc"] or "")[:36],
        "风险等级": r["level"],
        "状态": r.get("status", "待整改"),
        "责任人": r.get("owner") or "-",
        "期限": r.get("deadline") or "-",
    } for i, r in enumerate(risk_records)])


def record_choices():
    return [f"#{i + 1} {(r['desc'] or '')[:30]}" for i, r in enumerate(risk_records)]


def refresh_ledger_handler():
    return ledger_df(), gr.Dropdown(choices=record_choices(), value=None)


def update_status_handler(sel, new_status):
    """隐患台账状态流转：待整改 → 复查中 → 已整改"""
    if sel is None:
        return ledger_df(), "请先在台账下拉框选择要更新的隐患记录。"
    idx = int(str(sel).split(" ")[0].lstrip("#"))
    if 1 <= idx <= len(risk_records):
        risk_records[idx - 1]["status"] = new_status
        return ledger_df(), f"✅ 记录 #{idx} 状态已更新为：{new_status}"
    return ledger_df(), "记录不存在，请刷新台账。"


def render_notice_docx(records, now):
    """把待整改隐患渲染为《施工安全隐患整改通知单》Word 文档"""
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.2)
        sec.bottom_margin = Cm(2.2)
        sec.left_margin = Cm(2.5)
        sec.right_margin = Cm(2.5)

    title = doc.add_heading("施工安全隐患整改通知单（AI 辅助生成）", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(f"通知单编号：TZ-{now[:10].replace('-', '')}-{now[11:13]}{now[14:16]}{now[17:19]}\n"
                  f"下发时间：{now}\n生成方式：本系统基于规范知识库自动生成，供现场检查参考。")
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.add_heading("一、整改基本信息", level=1)
    t1 = doc.add_table(rows=5, cols=2)
    t1.style = "Table Grid"
    rows = [
        ("项目名称", "（待填写）"),
        ("施工单位", "（待填写）"),
        ("整改责任人", records[0].get("owner", "未指定")),
        ("整改期限", records[0].get("deadline", "未指定")),
        ("下发隐患条数", f"{len(records)} 条"),
    ]
    for i, (k, v) in enumerate(rows):
        t1.cell(i, 0).text = k
        t1.cell(i, 1).text = v

    doc.add_heading("二、需整改隐患清单", level=1)
    t2 = doc.add_table(rows=1 + len(records), cols=4)
    t2.style = "Table Grid"
    for j, htext in enumerate(["序号", "隐患描述", "风险等级", "整改要求"]):
        t2.rows[0].cells[j].text = htext
    for i, rec in enumerate(records, 1):
        cells = t2.rows[i].cells
        cells[0].text = str(i)
        cells[1].text = rec["desc"]
        cells[2].text = rec["level"]
        sug = rec["suggestions"]
        cells[3].text = "；".join(sug) if sug else "按相关规范整改"

    doc.add_heading("三、整改要求", level=1)
    serious = [r for r in records if r["level"] == "重大隐患"]
    if serious:
        doc.add_paragraph(
            "本次通知共 {n} 条隐患，其中重大隐患 {s} 条。重大隐患必须立即停止相关作业，"
            "制定专项整改方案并组织整改，整改完成经验收合格后方可恢复施工；其余隐患应在整改期限内完成整改。".format(
                n=len(records), s=len(serious)))
    else:
        doc.add_paragraph(f"本次通知共 {len(records)} 条隐患，均应在整改期限内完成整改，整改完成后报请复查验收。")

    doc.add_heading("四、签字确认", level=1)
    t3 = doc.add_table(rows=4, cols=3)
    t3.style = "Table Grid"
    for j, htext in enumerate(["角色", "签字", "日期"]):
        t3.rows[0].cells[j].text = htext
    for i, role in enumerate(["签发人", "整改责任人", "复查人"], 1):
        t3.rows[i].cells[0].text = role

    docx_path = os.path.join(os.getcwd(), "safety_notice.docx")
    doc.save(docx_path)
    return docx_path


def generate_notice_handler(sel):
    """对选中隐患（或全部待整改隐患）生成整改通知单 PDF"""
    if not risk_records:
        return None, "暂无可生成通知单的隐患记录，请先进行风险分析。"
    if sel is not None:
        idx = int(str(sel).split(" ")[0].lstrip("#"))
        recs = [risk_records[idx - 1]]
    else:
        recs = [r for r in risk_records if r.get("status", "待整改") == "待整改"]
        if not recs:
            return None, "没有待整改的隐患记录。"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        docx_path = render_notice_docx(recs, now)
        pdf_path = docx_to_pdf(docx_path)
        if os.path.exists(docx_path):
            os.remove(docx_path)
        return pdf_path, f"整改通知单（PDF）已生成：{os.path.basename(pdf_path)}"
    except Exception as e:
        print(f"通知单生成失败: {e}")
        return None, f"通知单生成失败：{e}"


# ---------- 统计仪表盘 ----------
def dashboard_handler():
    """基于风险台账生成统计图：风险等级分布 / 高频隐患关键词 Top10 / 隐患趋势"""
    import pandas as pd
    from collections import Counter
    import jieba
    import plotly.express as px
    import plotly.graph_objects as go

    empty_fig = go.Figure()
    if not risk_records:
        return empty_fig, empty_fig, empty_fig, "暂无隐患数据，请先进行风险分析或视频巡检。"
    df = pd.DataFrame(risk_records)
    lv = df["level"].value_counts().reset_index()
    lv.columns = ["风险等级", "数量"]
    lv_fig = px.pie(lv, names="风险等级", values="数量", title="风险等级分布",
                    color_discrete_sequence=px.colors.qualitative.Set1)
    cnt = Counter()
    for d in df["desc"]:
        for w in jieba.lcut(str(d)):
            if len(w) >= 2:
                cnt[w] += 1
    words = pd.DataFrame(cnt.most_common(10), columns=["隐患关键词", "出现次数"])
    word_fig = px.bar(words, x="隐患关键词", y="出现次数", title="高频隐患关键词 Top10",
                      color_discrete_sequence=["#1d4ed8"])
    tr = df.copy()
    tr["日期"] = tr["time"].str[:10]
    trd = tr.groupby("日期").size().cumsum().reset_index()
    trd.columns = ["日期", "累计隐患数"]
    trend_fig = px.line(trd, x="日期", y="累计隐患数", title="隐患累计趋势",
                        markers=True, color_discrete_sequence=["#1d4ed8"])
    return lv_fig, word_fig, trend_fig, (
        f"统计已更新：共 {len(risk_records)} 条隐患记录"
        f"（重大 {int((df['level'] == '重大隐患').sum())} / 一般 {int((df['level'] == '一般隐患').sum())}）。")


# ---------- 视频抽帧隐患识别 ----------
def extract_video_frames(video_path, max_frames=6):
    """从视频均匀抽取 max_frames 帧，返回 [(帧numpy, 时间点秒)]"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0
    step = max(1, total // max_frames) if total > 0 else 1
    frames = []
    idx = 0
    while idx < total and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((frame, idx / fps))
        idx += step
    cap.release()
    return frames


def video_analysis_handler(video_path, top_k, dist_threshold, mode_value, use_rerank, model_name):
    model_name = resolve_model(model_name)
    """上传现场视频 → 抽帧 → 逐帧 AI 隐患识别 → 视频巡检记录并入台账"""
    global g_index, g_chunks_with_source, g_bm25
    if g_index is None:
        return "⚠️ 向量库未初始化，请先上传文档并重建向量库。"
    if not video_path:
        return "请先上传现场视频（mp4/avi/mov 等）。"
    import cv2
    import time as _time
    frames = extract_video_frames(video_path)
    if not frames:
        return "无法读取视频，请确认文件格式与编码（建议 H.264 mp4）。"
    os.makedirs("demo_images/frames", exist_ok=True)
    use_hybrid = mode_map(mode_value)
    lines = ["# 🎥 视频巡检记录", ""]
    lines.append(f"> 视频：{os.path.basename(video_path)}　抽帧 {len(frames)} 帧　逐帧 AI 隐患识别", "")
    for i, (frame, t) in enumerate(frames, 1):
        tmp = f"demo_images/frames/vf_{int(_time.time())}_{i}.jpg"
        cv2.imwrite(tmp, frame)
        try:
            result = analyze_image_risk(
                tmp, g_index, g_chunks_with_source, g_bm25,
                top_k=top_k, dist_threshold=dist_threshold, use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                model=model_name,
            )
        except Exception as e:
            print(f"帧{i}识别异常: {e}")
            lines.append(f"### 第 {i} 帧（⏱ {t:.1f}s）识别失败：{e}")
            continue
        level = result.get("risk_level", "无法判断")
        badge = RISK_LEVEL_BADGE.get(level, f"⚪ {level}")
        lines.append(f"### 第 {i} 帧（⏱ {t:.1f}s）{badge}")
        if result.get("_image_desc"):
            lines.append(f"\n**识别结果**：{result['_image_desc']}")
        if result.get("summary"):
            lines.append(f"\n**风险分析**：{result['summary']}")
        ev = result.get("evidence", [])
        ev_map = {v.get("text"): v.get("verified", False) for v in result.get("_verified", [])}
        if ev:
            lines.append("\n**依据条款**：")
            for e in ev:
                mark = "✅ 已核验" if ev_map.get(e) else "⚠️ 待复核"
                lines.append(f"- {mark} {highlight_keywords(e, str(result.get('_image_desc', '')))}")
        sg = result.get("suggestions", [])
        if sg:
            lines.append("\n**整改建议**：")
            for j, s in enumerate(sg, 1):
                lines.append(f"{j}. {s}")
        lines.append(f"\n![第{i}帧画面]({tmp})")
        desc = f"[视频巡检·{t:.1f}s] {result.get('_image_desc', '') or f'第{i}帧画面'}"
        risk_records.append({
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "desc": desc,
            "level": level,
            "summary": result.get("summary", ""),
            "evidence": ev,
            "suggestions": sg,
            "sources": sorted(set(result.get("_sources", []))),
            "image_path": tmp,
            "status": "待整改",
            "owner": "未指定",
            "deadline": "未指定",
        })
        lines.append("")
    lines.append("\n> ✅ 全部帧识别完毕，隐患已自动加入台账，可一键生成带图安全检查报告或整改通知单。")
    return "\n".join(lines)


CUSTOM_CSS = """
#app-header {
  text-align: center;
  padding: 1.4rem 1rem;
  background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
  border-radius: 12px;
  margin-bottom: 0.8rem;
  box-shadow: 0 2px 10px rgba(30, 64, 175, 0.25);
}
#app-header h1 {
  font-size: 1.9rem;
  font-weight: 800;
  margin-bottom: 0.2rem;
  color: #ffffff;
  letter-spacing: 0.12em;
}
#app-header p {
  color: #dbeafe;
  font-size: 0.9rem;
  margin-top: 0;
}
.side-card, .chat-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 0.7rem 0.9rem;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
.side-card h3 {
  color: #ffffff;
  background: linear-gradient(90deg, #1d4ed8, #2563eb);
  font-size: 0.98rem;
  font-weight: 700;
  margin: 0 0 0.6rem 0;
  padding: 0.5rem 0.8rem;
  border-radius: 8px;
  border-bottom: none;
}
body {
  background: #f8fafc !important;
}
.gr-button {
  border-radius: 8px !important;
  font-weight: 500 !important;
}
footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Microsoft YaHei", "sans-serif"],
)

with gr.Blocks(title="智安查 · 建造安全智能问答") as demo:
    gr.HTML("""
    <div id="app-header">
      <h1>智安查</h1>
      <p>智慧工地安全规范问答 · 隐患风险分析 · 检查报告</p>
    </div>
    """)

    with gr.Row():
        # 左侧：文档与索引 + 高级参数
        with gr.Column(scale=1, min_width=280):
            with gr.Group(elem_classes="side-card"):
                gr.Markdown("### 文档与索引")
                upload_files = gr.File(file_types=[".txt", ".md", ".pdf"], file_count="multiple")
                upload_info = gr.Textbox(label="上传状态", interactive=False)
                doc_list = gr.Dropdown(label="知识库文档", choices=[], interactive=True)
                with gr.Row():
                    refresh_docs_btn = gr.Button("刷新列表", variant="secondary")
                    delete_doc_btn = gr.Button("删除所选", variant="secondary")
                with gr.Row():
                    rebuild_btn = gr.Button("重建索引", variant="primary")
                    clear_btn_2 = gr.Button("清空缓存", variant="secondary")
                rebuild_info = gr.Textbox(label="索引状态", interactive=False)

            with gr.Accordion("⚙ 高级参数", open=False):
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
                rerank_checkbox = gr.Checkbox(
                    label="启用 Rerank 重排（更精准，稍慢）",
                    value=True,
                )
                model_dropdown = gr.Dropdown(
                    choices=["qwen-turbo（默认，快）", "qwen-plus（更准）", "qwen-max（最强）"],
                    value="qwen-turbo（默认，快）",
                    label="问答模型",
                )

            with gr.Accordion("📊 隐患台账与统计", open=False):
                ledger_table = gr.Dataframe(
                    headers=LEDGER_HEADERS,
                    value=ledger_df(),
                    interactive=False,
                    wrap=True,
                    label="隐患台账（本会话）",
                )
                with gr.Row():
                    record_sel = gr.Dropdown(choices=record_choices(), label="选择记录")
                    status_dropdown = gr.Dropdown(
                        choices=["待整改", "复查中", "已整改"],
                        value="复查中",
                        label="更新为",
                    )
                with gr.Row():
                    update_status_btn = gr.Button("更新状态", variant="secondary")
                    refresh_ledger_btn = gr.Button("刷新台账", variant="secondary")
                with gr.Row():
                    notice_btn = gr.Button("生成整改通知单", variant="secondary")
                    notice_file = gr.File(label="下载通知单（PDF）")
                dashboard_btn = gr.Button("刷新统计", variant="secondary")
                level_pie = gr.Plot(label="风险等级分布")
                word_bar = gr.Plot(label="高频隐患关键词 Top10")
                trend_line = gr.Plot(label="隐患累计趋势")
                dashboard_info = gr.Textbox(label="统计状态", interactive=False)

        # 右侧：聊天 + 风险分析
        with gr.Column(scale=2):
            with gr.Group(elem_classes="chat-card"):
                chatbot = gr.Chatbot(height=520, buttons=["copy"])
                msg_input = gr.Textbox(
                    label="输入问题",
                    placeholder="输入问题，例如：特种作业人员上岗有什么要求？",
                    lines=1
                )
                with gr.Row():
                    clear_btn = gr.Button("清空对话", variant="secondary")
                    export_btn = gr.Button("导出对话记录", variant="secondary")
                    export_file = gr.File(label="下载导出的记录")

            with gr.Accordion("🔍 隐患风险分析（结构化输出）", open=False):
                risk_input = gr.Textbox(
                    label="描述施工现场情况（文字或照片二选一/可并用）",
                    placeholder="例如：六米深的基坑没有做专项施工方案就直接开挖，坑边堆土很近",
                    lines=2,
                )
                risk_image = gr.Image(type="filepath", label="现场照片（可选，自动识别隐患）")
                with gr.Row():
                    risk_owner = gr.Textbox(label="整改责任人（可选）", placeholder="如：张工")
                    risk_deadline = gr.Textbox(label="整改期限（可选）", placeholder="如：2026-09-20")
                risk_btn = gr.Button("开始风险分析", variant="primary")
                risk_output = gr.Markdown()
                with gr.Row():
                    report_btn = gr.Button("生成安全检查报告", variant="secondary")
                    report_file = gr.File(label="下载报告（PDF）")
                with gr.Row():
                    report_template = gr.Dropdown(
                        choices=["整改通知单", "监理工程师通知单", "项目安全周报"],
                        value="整改通知单",
                        label="模板报告",
                    )
                    template_btn = gr.Button("生成模板报告", variant="secondary")
                    template_file = gr.File(label="下载模板报告（PDF）")

            with gr.Accordion("🎥 视频巡检（抽帧隐患识别）", open=False):
                video_input = gr.Video(label="现场视频（mp4/avi/mov）")
                video_btn = gr.Button("开始视频巡检", variant="primary")

            with gr.Accordion("🎤 语音问答", open=False):
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="点击录音，说出施工现场情况")
                asr_btn = gr.Button("语音转文字", variant="primary")
                asr_output = gr.Textbox(label="识别结果（可编辑后用于风险分析）", interactive=True, lines=2)
                with gr.Row():
                    tts_btn = gr.Button("朗读风险分析结果", variant="secondary")
                    tts_audio = gr.Audio(label="语音播报", type="filepath")

    # 事件绑定
    def mode_map(x):
        return x == "混合检索（向量+关键词）"

    upload_files.upload(upload_file_handler, inputs=[upload_files], outputs=[upload_info])
    rebuild_btn.click(rebuild_btn_click, inputs=[chunk_size_slider, overlap_slider], outputs=[rebuild_info])
    clear_btn_2.click(clear_vector_btn_click, outputs=[rebuild_info])

    msg_input.submit(
        fn=chat_handler,
        inputs=[msg_input, chatbot, top_k_slider, dist_slider, max_rounds_slider, mode_dropdown, rerank_checkbox, model_dropdown],
        outputs=[chatbot, msg_input]
    )
    clear_btn.click(lambda: [], None, chatbot)
    export_btn.click(export_chat_handler, inputs=[chatbot], outputs=[export_file, rebuild_info])
    refresh_docs_btn.click(list_docs_handler, outputs=doc_list)
    delete_doc_btn.click(delete_doc_handler, inputs=doc_list, outputs=[doc_list, rebuild_info])
    demo.load(list_docs_handler, outputs=doc_list)
    risk_btn.click(
        risk_analysis_handler,
        inputs=[risk_input, risk_image, risk_owner, risk_deadline, top_k_slider, dist_slider, mode_dropdown, rerank_checkbox, model_dropdown],
        outputs=[risk_output, ledger_table],
    )
    report_btn.click(generate_report_handler, outputs=[report_file, risk_output])
    video_btn.click(
        video_analysis_handler,
        inputs=[video_input, top_k_slider, dist_slider, mode_dropdown, rerank_checkbox, model_dropdown],
        outputs=[risk_output, ledger_table],
    )
    refresh_ledger_btn.click(refresh_ledger_handler, outputs=[ledger_table, record_sel])
    update_status_btn.click(update_status_handler, inputs=[record_sel, status_dropdown], outputs=[ledger_table, dashboard_info])
    notice_btn.click(generate_notice_handler, inputs=[record_sel], outputs=[notice_file, dashboard_info])
    dashboard_btn.click(dashboard_handler, outputs=[level_pie, word_bar, trend_line, dashboard_info])
    template_btn.click(generate_template_handler, inputs=[report_template], outputs=[template_file, dashboard_info])
    asr_btn.click(asr_question_handler, inputs=[audio_input], outputs=[asr_output, risk_input])
    tts_btn.click(tts_handler, inputs=[risk_output], outputs=[tts_audio])


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
