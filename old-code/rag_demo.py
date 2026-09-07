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

# 1.原始文档
doc_content = """企业短期偿债能力分析

流动比率是企业财务管理中核心的偿债指标，计算公式为流动资产除以流动负债。流动资产包含货币资金、应收票据、应收账款、存货以及其他可以在一年内变现的资产，流动负债主要涵盖短期借款、应付账款、应付票据、应交税费等一年内需要偿还的债务。通常认为流动比率维持在2左右相对合理，比率越高代表企业短期偿还债务的缓冲空间越大，但过高的流动比率也可能说明企业存货积压、资金闲置，资产利用效率偏低。

速动比率剔除了存货资产的影响，计算方式为速动资产除以流动负债，速动资产等于流动资产减去存货。存货存在滞销贬值的风险，无法保证快速变现，速动比率可以更真实反映企业即时还债能力，一般参考标准为1。

现金比率是更为严苛的短期偿债指标，仅使用货币资金与交易性金融资产除以流动负债，只看企业手上可立刻动用的现金类资产，该指标数值不宜过高，过高意味着大量现金没有投入经营，盈利能力被拉低。

在财务分析过程中不能孤立看待单一指标，需要结合行业特性横向对比同类型企业，同时结合企业历年报表数据纵向观察变化趋势。零售行业存货周转快，流动比率普遍偏低；制造业存货占比高，流动比率会相对更高。如果企业三项偿债指标持续下降，说明短期财务风险在上升，存在到期债务无法按时兑付的隐患。
"""

# 2.简单文本切分（按段落分块）
def split_text(text, chunk_size=350):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) > chunk_size:
            if buf:
                chunks.append(buf)
            buf = para
        else:
            buf += "\n" + para
    if buf:
        chunks.append(buf)
    return chunks

chunks = split_text(doc_content)
print(f"文档切分完成，共 {len(chunks)} 个文本块")

# 3.获取embedding向量
def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-v3",
        input=text
    )
    return resp.data[0].embedding

emb_list = []
for idx, chunk in enumerate(chunks):
    print(f"生成向量 {idx+1}/{len(chunks)}")
    emb = get_embedding(chunk)
    emb_list.append(emb)

emb_np = np.array(emb_list, dtype=np.float32)
dim = emb_np.shape[1]

# 4.构建FAISS本地向量库
index = faiss.IndexFlatL2(dim)
index.add(emb_np)
print("向量库构建完成")

# 5.检索函数
def search_k(query, top_k=2):
    q_emb = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(q_emb, top_k)
    hit_chunks = [chunks[i] for i in indices[0]]
    return hit_chunks

# 6.RAG问答
def rag_ask(question):
    related_texts = search_k(question, top_k=2)
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
    print("\n====RAG问答测试====")
    q1 = "速动比率怎么计算，参考值是多少？"
    ans1 = rag_ask(q1)
    print(f"Q:{q1}\nA:{ans1}\n")

    q2 = "零售和制造业流动比率有什么区别？"
    ans2 = rag_ask(q2)
    print(f"Q:{q2}\nA:{ans2}\n")

    q3 = "现金比率过高会带来什么问题？"
    ans3 = rag_ask(q3)
    print(f"Q:{q3}\nA:{ans3}\n")