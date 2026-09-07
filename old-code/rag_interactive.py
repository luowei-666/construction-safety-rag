import os
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=10.0
)

# ==========拓展1：读取本地txt文件============
def load_txt_file(file_path):
    """读取本地txt文档"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}，请把txt放到程序同目录")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 修复版分块：按字符切割，不怕没有换行
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

# 3.获取embedding向量
def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-v3",
        input=text
    )
    return resp.data[0].embedding

# 4.检索函数
def search_k(index, chunks, query, top_k=2):
    q_emb = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(q_emb, top_k)
    hit_chunks = [chunks[i] for i in indices[0]]
    return hit_chunks

# 5.RAG问答
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
    # 加载同目录下test.txt
    doc_content = load_txt_file("test.txt")
    print("✅成功读取test.txt")
    print(f"txt原始文本长度：{len(doc_content)}")

    chunks = split_text(doc_content)
    print(f"文档切分完成，共 {len(chunks)} 个文本块")
    if len(chunks) == 0:
        print("错误：没有切分出文本块，txt内容为空！")
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
    print("✅向量库构建完成\n")

    # ==========拓展2：循环交互问答============
    print("====交互式RAG问答，输入exit退出程序====")
    while True:
        user_input = input("\n请输入你的问题：").strip()
        if user_input.lower() == "exit":
            print("程序退出")
            break
        if not user_input:
            continue
        answer = rag_ask(index, chunks, user_input)
        print(f"\n回答：{answer}")