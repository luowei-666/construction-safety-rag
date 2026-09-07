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

def load_txt_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, chunk_size=350):
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
        start = end
    return chunks

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-v3",
        input=text
    )
    return resp.data[0].embedding

# 将向量库、分块保存到本地
def save_vector_store(index, chunks):
    faiss.write_index(index, VECTOR_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print("💾向量库已持久化保存到本地")

# 从磁盘加载向量库、分块
def load_vector_store():
    if not os.path.exists(VECTOR_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None
    index = faiss.read_index(VECTOR_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print("📂从本地加载已存在的向量库")
    return index, chunks

def search_k(index, chunks, query, top_k=2):
    q_emb = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(q_emb, top_k)
    hit_chunks = [chunks[i] for i in indices[0]]
    return hit_chunks

def rag_ask(index, chunks, question):
    related_texts = search_k(index, chunks, question, top_k=2)
    context = "\n---\n".join(related_texts)
    prompt = f"""基于下面参考文档回答用户问题，只使用文档内信息。
【参考文档】
{context}

【用户问题】
{question}
"""
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role":"system","content":"你是财务分析助手，严格依据提供的参考文档回答，不要编造文档不存在的内容。"},
            {"role":"user","content": prompt}
        ]
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    index, chunks = load_vector_store()

    # 如果本地没有缓存向量库，则重新构建
    if index is None or chunks is None:
        print("⚠本地无缓存向量库，开始构建……")
        doc_content = load_txt_file("test.txt")
        print(f"✅读取test.txt，文本长度：{len(doc_content)}")
        chunks = split_text(doc_content)
        print(f"文档切分完成，共 {len(chunks)} 个文本块")
        if len(chunks) == 0:
            print("错误：文档为空，退出")
            exit(1)

        emb_list = []
        for idx, chunk in enumerate(chunks):
            print(f"生成向量 {idx+1}/{len(chunks)}")
            emb = get_embedding(chunk)
            emb_list.append(emb)

        emb_np = np.array(emb_list, dtype=np.float32)
        dim = emb_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(emb_np)
        save_vector_store(index, chunks)
    else:
        print(f"向量库加载成功，块数量：{len(chunks)}")

    print("\n====交互式RAG问答，输入exit退出程序====")
    while True:
        user_input = input("\n请输入你的问题：").strip()
        if user_input.lower() == "exit":
            print("程序退出")
            break
        if not user_input:
            continue
        answer = rag_ask(index, chunks, user_input)
        print(f"\n回答：{answer}")