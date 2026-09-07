"""rag_core.py — RAG 核心公共模块

包含：向量库读写、BM25 索引构建、混合检索(RRF 融合)、Embedding/大模型调用、文档解析分块。
供 rag_webui.py（网页）与 evaluate.py（评估）共同复用。
"""
import os
import re
import faiss
import numpy as np
import pickle
import requests
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import jieba
from rank_bm25 import BM25Okapi

load_dotenv(override=True)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

VECTOR_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"
META_PATH = "meta.pkl"
BM25_PATH = "bm25_index.pkl"
DOC_FOLDER = "./docs"

jieba.setLogLevel(20)  # 关闭 jieba 初始化日志

if not os.path.exists(DOC_FOLDER):
    os.mkdir(DOC_FOLDER)


# ---------- 分词 ----------
def tokenize(text: str):
    """中文分词，去除空白与标点"""
    tokens = [w.strip() for w in jieba.lcut(text)]
    stop = set(" .,;:!?，。；：！？、()（）\"''\n\t《》")
    return [w for w in tokens if w and w not in stop]


# ---------- 通义 API ----------
def dashscope_embedding(text: str):
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-v3",
        "input": {"texts": [text]}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["output"]["embeddings"][0]["embedding"]


def dashscope_chat(messages):
    """非流式调用，返回完整回答"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-turbo",
        "input": {"messages": messages},
        "parameters": {"result_format": "message"}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    return j["output"]["choices"][0]["message"]["content"].strip()


def dashscope_chat_stream(messages):
    """流式调用，逐字返回"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-turbo",
        "input": {"messages": messages},
        "parameters": {"result_format": "message", "incremental_output": True}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60, stream=True)
    for line in resp.iter_lines():
        if not line:
            continue
        raw = line.decode("utf-8").removeprefix("data:")
        try:
            j = json.loads(raw)
            chunk = j["output"]["choices"][0]["message"]["content"]
            yield chunk
        except Exception:
            continue


# ---------- 文档解析与分块 ----------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text


def scan_all_docs(folder):
    file_list = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        mtime = os.path.getmtime(fpath)
        content = ""
        if fname.lower().endswith((".txt", ".md")):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        elif fname.lower().endswith(".pdf"):
            content = read_pdf(fpath)
        else:
            continue
        file_list.append((fname, fpath, mtime, content))
    return file_list


# 标题行识别：条款号 / 中文序数章节 / 数字编号 / Markdown 标题
# （不匹配"（一）（二）"等列表项，避免条款被过度切碎）
TITLE_RE = re.compile(
    r"^(第[一二三四五六七八九十百零〇\d]+条"
    r"|[一二三四五六七八九十]+、"
    r"|\d+[\.、]"
    r"|#{1,4}\s)"
)


def _split_long_block(block, chunk_size, overlap):
    """长块细切：优先在句号/分号后断句，保持语义完整；过短尾片并入前块"""
    out = []
    n = len(block)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = block.rfind("。", start + chunk_size // 2, end + 1)
            if cut == -1:
                cut = block.rfind("；", start + chunk_size // 2, end + 1)
            if cut != -1:
                end = cut + 1
        out.append(block[start:end])
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    if len(out) > 1 and len(out[-1]) < 80:
        out[-2] += out[-1]
        out.pop()
    return out


def split_text(text, chunk_size=350, overlap=60):
    """语义分块：以条款号/章节标题/空行为块边界，标题保留在块首；长块按句号断句。

    相比纯字符切块，避免切断条款原文，使检索命中的引用更完整。
    """
    text = text.strip()
    if not text:
        return []
    # 第一遍：按标题行/空行分组为逻辑块
    logical = []
    current = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            if current:
                logical.append("\n".join(current))
                current = []
            continue
        if TITLE_RE.match(s) and current:
            logical.append("\n".join(current))
            current = []
        current.append(ln)
    if current:
        logical.append("\n".join(current))

    chunks = []
    for block in logical:
        if not block.strip():
            continue
        if len(block) <= chunk_size:
            chunks.append(block)
        else:
            chunks.extend(_split_long_block(block, chunk_size, overlap))

    # 合并过小碎片（<60字符且非标题行）到前一块
    merged = []
    for c in chunks:
        if merged and len(c) < 60 and not TITLE_RE.match(c.strip()):
            merged[-1] += "\n" + c
        else:
            merged.append(c)
    return [c for c in merged if c.strip()]


# ---------- 向量库持久化 ----------
def save_vector_store(index, chunks_with_source, file_mtime_dict):
    faiss.write_index(index, VECTOR_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks_with_source, f)
    meta = {"file_mtime": file_mtime_dict}
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)


def load_vector_store():
    if not (os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(META_PATH)):
        return None, None, None
    index = faiss.read_index(VECTOR_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks_with_source = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, chunks_with_source, meta


# ---------- BM25 索引 ----------
def build_bm25_index(chunks_with_source):
    """基于分块文本构建 BM25，返回 (bm25对象, 分词文档列表)"""
    tokenized_docs = [tokenize(c["text"]) for c in chunks_with_source]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs


def save_bm25_index(tokenized_docs):
    with open(BM25_PATH, "wb") as f:
        pickle.dump(tokenized_docs, f)


def load_bm25_index():
    if not os.path.exists(BM25_PATH):
        return None, None
    with open(BM25_PATH, "rb") as f:
        tokenized_docs = pickle.load(f)
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs


# ---------- 向量库构建 ----------
def check_any_file_changed(file_list, old_meta):
    old_dict = old_meta.get("file_mtime", {})
    new_dict = {fname: mtime for fname, _, mtime, _ in file_list}
    if set(old_dict.keys()) != set(new_dict.keys()):
        return True
    for fname in new_dict:
        if abs(new_dict[fname] - old_dict[fname]) > 0.001:
            return True
    return False


def build_vector_from_docs(chunk_size=350, overlap=60, force_rebuild=False):
    """构建（或加载）向量库与 BM25 索引。

    返回: (index, chunks_with_source, msg, bm25)
    """
    file_list = scan_all_docs(DOC_FOLDER)
    if len(file_list) == 0:
        return None, None, "docs文件夹无文档", None

    index, chunks_with_source, meta = load_vector_store()
    need_rebuild = force_rebuild
    if not force_rebuild:
        if index is not None and meta is not None:
            if not check_any_file_changed(file_list, meta):
                need_rebuild = False

    if need_rebuild:
        print("重新构建向量库...")
        file_mtime_dict = {fname: mtime for fname, _, mtime, _ in file_list}
        chunks_with_source = []
        for fname, _, _, content in file_list:
            sub_chunks = split_text(content, chunk_size=chunk_size, overlap=overlap)
            for ck in sub_chunks:
                chunks_with_source.append({"source": fname, "text": ck})
        emb_list = []
        for idx, item in enumerate(chunks_with_source):
            emb = dashscope_embedding(item["text"])
            emb_list.append(emb)
        emb_np = np.array(emb_list, dtype=np.float32)
        index = faiss.IndexFlatL2(emb_np.shape[1])
        index.add(emb_np)
        save_vector_store(index, chunks_with_source, file_mtime_dict)
        # 同步构建 BM25 索引
        bm25, tokenized_docs = build_bm25_index(chunks_with_source)
        save_bm25_index(tokenized_docs)
        msg = f"向量库构建完成，共 {len(chunks_with_source)} 个文本块"
    else:
        msg = f"加载缓存向量库，共 {len(chunks_with_source)} 个文本块"

    bm25, _ = load_bm25_index()
    return index, chunks_with_source, msg, bm25


# ---------- 混合检索 ----------
def hybrid_search(index, chunks_with_source, bm25, query, top_k=3,
                  dist_threshold=0.85, use_hybrid=True, weight_bm25=0.25):
    """混合检索：向量召回 + BM25 关键词召回，RRF 融合排序。

    use_hybrid=False 时仅用向量检索。
    返回: [{source, text, score}, ...]，score 为 RRF 融合分（混合）或 L2 距离（纯向量）。
    """
    # 向量路
    q_emb = np.array([dashscope_embedding(query)], dtype=np.float32)
    distances, indices = index.search(q_emb, max(top_k * 3, 10))
    vec_hits = []
    for dist, i in zip(distances[0], indices[0]):
        if dist <= dist_threshold:
            vec_hits.append((int(i), float(dist)))

    if (not use_hybrid) or bm25 is None:
        result = []
        for i, d in vec_hits[:top_k]:
            item = dict(chunks_with_source[i])
            item["score"] = d
            result.append(item)
        return result

    # BM25 路
    tokens = tokenize(query)
    bm25_scores = bm25.get_scores(tokens)
    bm25_hits = [(i, float(s)) for i, s in enumerate(bm25_scores) if s > 0]
    bm25_hits.sort(key=lambda x: x[1], reverse=True)
    bm25_hits = bm25_hits[:top_k * 3]

    # RRF 融合
    K = 60.0
    rrf = {}
    for rank, (i, _) in enumerate(vec_hits):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (K + rank + 1)
    for rank, (i, _) in enumerate(bm25_hits):
        rrf[i] = rrf.get(i, 0.0) + weight_bm25 / (K + rank + 1)

    merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for i, s in merged:
        item = dict(chunks_with_source[i])
        item["score"] = s
        result.append(item)
    return result


# ---------- 上下文截断 ----------
def truncate_messages(history_msgs, max_rounds):
    """只保留最近 max_rounds 轮对话（每轮=1条user+1条assistant）"""
    if not history_msgs:
        return []
    current = history_msgs[-1:]          # 最后一条是当前用户消息，必须保留
    prev = history_msgs[:-1]
    if max_rounds and max_rounds > 0:
        limit = max_rounds * 2
        if len(prev) > limit:
            prev = prev[-limit:]
    return prev + current


# ---------- 多模态（现场照片隐患识别） ----------
def _image_to_data_uri(image_path):
    """本地图片转 base64 Data URI（供多模态接口使用）"""
    import base64
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp",
            "bmp": "bmp", "gif": "gif"}.get(ext, "jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def dashscope_vl(messages, model="qwen-vl-plus", timeout=90):
    """通义千问视觉模型（OpenAI 兼容接口），返回完整回答"""
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def describe_image_hazard(image_path):
    """识别现场照片中的安全隐患，返回文字描述（供检索与风险分析使用）"""
    data_uri = _image_to_data_uri(image_path)
    prompt = (
        "你是施工现场安全检查专家。请仔细观察这张现场照片，用简洁的中文列出其中存在的"
        "安全隐患（如高处作业不系安全带、临边无防护、基坑边坡失稳、用电不规范、物体堆放"
        "危险等）。要求：1）只描述照片中可见的事实，不要臆测；2）若有多处隐患逐条列出；"
        "3）若未发现明显隐患，明确说明'照片中未发现明显安全隐患'。"
    )
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": prompt},
    ]}]
    return dashscope_vl(messages)


def analyze_image_risk(image_path, index, chunks_with_source, bm25, top_k=5,
                       dist_threshold=0.85, use_hybrid=True, weight_bm25=0.25):
    """现场照片隐患分析：先视觉识别隐患描述，再走规范检索 + 结构化风险分析。

    返回 analyze_risk 的结果，并附带 _image_desc（视觉识别出的隐患描述）。
    """
    desc = describe_image_hazard(image_path)
    result = analyze_risk(desc, index, chunks_with_source, bm25,
                          top_k=top_k, dist_threshold=dist_threshold,
                          use_hybrid=use_hybrid, weight_bm25=weight_bm25)
    result["_image_desc"] = desc
    return result


# ---------- 隐患风险结构化分析 ----------
def analyze_risk(question, index, chunks_with_source, bm25, top_k=5,
                 dist_threshold=0.85, use_hybrid=True, weight_bm25=0.25):
    """对施工现场隐患描述做结构化风险分析（决策辅助层）。

    流程：混合检索规范条款 → LLM 按固定 JSON 结构输出风险等级/依据/整改建议。
    返回 dict: {risk_level, summary, evidence: [], suggestions: [], _sources: []}
    """
    hits = hybrid_search(index, chunks_with_source, bm25, question,
                         top_k=top_k, dist_threshold=dist_threshold,
                         use_hybrid=use_hybrid, weight_bm25=weight_bm25)
    if not hits:
        return {
            "risk_level": "无法判断",
            "summary": "知识库中未检索到与描述相关的规范条款，无法给出风险判定。",
            "evidence": [],
            "suggestions": [],
            "_sources": [],
        }

    context_parts = [f"【来源：{h['source']}】\n{h['text']}" for h in hits]
    context = "\n---\n".join(context_parts)
    prompt = f"""你是建筑施工安全领域的专家助手。请根据下面检索到的规范/标准条文，对用户描述的施工现场情况进行安全隐患风险分析，并严格按 JSON 格式输出，不要输出 JSON 以外的任何内容。

【检索到的参考条文】
{context}

【施工现场情况描述】
{question}

请输出如下 JSON：
{{
  "risk_level": "重大隐患 / 一般隐患 / 无隐患 / 无法判断（四选一，重大隐患指可能造成群死群伤或重大经济损失、须立即停工整改的情形）",
  "summary": "用2-4句话概括该情况存在的安全风险及定性依据",
  "evidence": ["命中的规范条款原文摘录，可多条，必须来自上面的参考条文"],
  "suggestions": ["整改建议，2-5条，须与规范要求对应"]
}}
注意：只依据检索到的条文判断；若条文与描述不相关，risk_level 填"无法判断"。"""
    msgs = [{"role": "user", "content": prompt}]
    raw = dashscope_chat(msgs)

    data = None
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        data = {"risk_level": "解析失败", "summary": raw, "evidence": [], "suggestions": []}

    data.setdefault("risk_level", "无法判断")
    data.setdefault("summary", "")
    data.setdefault("evidence", [])
    data.setdefault("suggestions", [])
    if not isinstance(data["evidence"], list):
        data["evidence"] = [str(data["evidence"])]
    if not isinstance(data["suggestions"], list):
        data["suggestions"] = [str(data["suggestions"])]
    # 证据条款去重（保序）
    seen, unique_ev = set(), []
    for e in data["evidence"]:
        key = re.sub(r"\s+", "", str(e))
        if key and key not in seen:
            seen.add(key)
            unique_ev.append(str(e))
    data["evidence"] = unique_ev
    data["_sources"] = [h["source"] for h in hits]
    return data


# ---------- Query 改写 ----------
def rewrite_query(history, current_question):
    if len(history) == 0:
        return current_question
    history_text = ""
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"
    prompt = f"""根据下面对话历史，把用户当前问题改写成一个完整、独立、不带代词的检索查询语句，只输出改写后的查询，不要多余解释。
对话历史：{history_text}
用户当前问题：{current_question}
"""
    msgs = [{"role": "user", "content": prompt}]
    return dashscope_chat(msgs)
