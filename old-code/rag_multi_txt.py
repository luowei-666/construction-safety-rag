import os
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=10.0
)

VECTOR_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"
META_PATH = "meta.pkl"
DOC_FOLDER = "./docs"  # 存放所有txt文档的文件夹

def scan_all_txt(folder):
    """扫描文件夹下全部txt，返回 [(文件名, 文件完整路径, 文件修改时间, 文件内容)]"""
    if not os.path.exists(folder):
        os.mkdir(folder)
        print(f"📁自动创建文件夹 {folder}，请把txt文档放入此目录")
        return []
    file_list = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".txt"):
            fpath = os.path.join(folder, fname)
            mtime = os.path.getmtime(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            file_list.append((fname, fpath, mtime, content))
    return file_list

def split_text(text, chunk_size=350, overlap=60):
    """带重叠分块"""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    total_len = len(text)
    while start < total_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-v3",
        input=text
    )
    return resp.data[0].embedding

def save_vector_store(index, chunks_with_source, file_mtime_dict):
    faiss.write_index(index, VECTOR_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks_with_source, f)
    meta = {"file_mtime": file_mtime_dict}
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print("💾多文档向量库与元信息保存完成")

def load_vector_store():
    if not (os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(META_PATH)):
        return None, None, None
    index = faiss.read_index(VECTOR_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks_with_source = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, chunks_with_source, meta

def check_any_file_changed(file_list, old_meta):
    """对比：任意txt新增/删除/修改 返回True需要重建"""
    old_dict = old_meta.get("file_mtime", {})
    new_dict = {fname: mtime for fname, _, mtime, _ in file_list}
    # 文件数量不一样
    if set(old_dict.keys()) != set(new_dict.keys()):
        return True
    # 文件mtime发生变化
    for fname in new_dict:
        if abs(new_dict[fname] - old_dict[fname]) > 0.001:
            return True
    return False

def search_k(index, chunks_with_source, query, top_k=3):
    q_emb = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(q_emb, top_k)
    hit = []
    for i in indices[0]:
        hit.append(chunks_with_source[i])
    return hit

def rag_ask(index, chunks_with_source, question):
    hit_items = search_k(index, chunks_with_source, question, top_k=3)
    print("\n--------检索召回片段--------")
    context_parts = []
    for idx, item in enumerate(hit_items):
        src_name, seg_text = item["source"], item["text"]
        print(f"【片段{idx+1}】来源:{src_name} 内容:{seg_text[:130]}...")
        context_parts.append(f"[文档来源:{src_name}]\n{seg_text}")
    print("--------------------------------")
    context = "\n---\n".join(context_parts)

    prompt = f"""基于下面参考文档回答用户问题，只使用文档内信息。
【参考文档】
{context}

【用户问题】
{question}
"""
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role":"system","content":"你是文档问答助手，严格依据提供的参考文档回答，不要编造文档不存在的内容。回答可以标注信息来自哪个文档。"},
            {"role":"user","content": prompt}
        ]
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    file_list = scan_all_txt(DOC_FOLDER)
    if len(file_list) == 0:
        print(f"docs文件夹下没有找到任何txt文档，请放入txt文件后重新运行！")
        exit(0)
    print(f"✅扫描到 {len(file_list)} 个txt文档：{[f[0] for f in file_list]}")

    index, chunks_with_source, meta = load_vector_store()
    need_rebuild = True
    if index is not None and meta is not None:
        if not check_any_file_changed(file_list, meta):
            need_rebuild = False

    if need_rebuild:
        print("⚠检测到文档改动/无缓存，开始构建多文档向量库……")
        file_mtime_dict = {fname: mtime for fname, _, mtime, _ in file_list}
        chunks_with_source = []
        for fname, _, _, content in file_list:
            sub_chunks = split_text(content, chunk_size=350, overlap=60)
            for ck in sub_chunks:
                chunks_with_source.append({"source": fname, "text": ck})
        print(f"全部文档切分完成，总块数：{len(chunks_with_source)}")
        if len(chunks_with_source) == 0:
            print("错误：所有文档内容为空，退出")
            exit(1)

        emb_list = []
        for idx, item in enumerate(chunks_with_source):
            print(f"生成向量 {idx+1}/{len(chunks_with_source)}")
            emb = get_embedding(item["text"])
            emb_list.append(emb)

        emb_np = np.array(emb_list, dtype=np.float32)
        dim = emb_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(emb_np)
        save_vector_store(index, chunks_with_source, file_mtime_dict)
    else:
        print("📂文档无改动，直接加载本地缓存向量库")
        print(f"总文本块数量：{len(chunks_with_source)}")

    print("\n====多文档RAG交互问答，输入exit退出====")
    while True:
        user_input = input("\n请输入你的问题：").strip()
        if user_input.lower() == "exit":
            print("程序退出")
            break
        if not user_input:
            continue
        ans = rag_ask(index, chunks_with_source, user_input)
        print(f"\n回答：{ans}")