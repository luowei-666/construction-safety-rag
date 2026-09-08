"""evaluate.py — RAG 检索与回答质量评估脚本

用法:
  python evaluate.py                          # 混合检索评估
  python evaluate.py --mode vector            # 纯向量检索评估
  python evaluate.py --compare                # 一键对比混合 vs 纯向量（推荐）
  python evaluate.py --compare --with-answer  # 额外做回答质量评估（消耗 API）
  python evaluate.py --test-set test_set.json --top-k 5 --threshold 5.0
"""
import argparse
import json
import numpy as np

from rag_core import (
    load_vector_store,
    load_bm25_index,
    hybrid_search,
    build_vector_from_docs,
    dashscope_embedding,
    dashscope_chat,
)


def cosine(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run_eval(index, chunks, bm25, test_set, top_k, use_hybrid, dist_threshold, use_rerank=False):
    """对测试集跑一次检索评估，返回 (hit_at, per_item)"""
    total = len(test_set)
    hit_at = {k: 0 for k in (1, 3, 5)}
    per_item = []
    for item in test_set:
        q = item["question"]
        expected = set(item.get("expected_sources", []))
        hits = hybrid_search(
            index, chunks, bm25, q,
            top_k=top_k, dist_threshold=dist_threshold, use_hybrid=use_hybrid,
            use_rerank=use_rerank,
        )
        hit_sources = [h["source"] for h in hits]
        for k in (1, 3, 5):
            if expected & set(hit_sources[:k]):
                hit_at[k] += 1
        per_item.append({
            "question": q,
            "expected": sorted(expected),
            "retrieved": hit_sources[:top_k],
            "hit": bool(expected & set(hit_sources)),
        })
    return hit_at, per_item, total


def print_report(title, hit_at, total, per_item):
    print("\n" + "=" * 56)
    print(f"  {title}")
    print("=" * 56)
    print(f"测试题数: {total}")
    for k in (1, 3, 5):
        print(f"  Recall@{k}: {hit_at[k] / total:.2%}  ({hit_at[k]}/{total})")
    print("\n逐题明细:")
    for p in per_item:
        mark = "✓" if p["hit"] else "✗"
        print(f"  [{mark}] Q: {p['question']}")
        print(f"       期望来源: {p['expected']}")
        print(f"       实际召回: {p['retrieved']}")


def answer_quality(index, chunks, bm25, test_set, dist_threshold, use_hybrid=True):
    """回答质量评估：回答与标准答案 embedding 余弦相似度"""
    print("\n" + "=" * 56)
    print("  回答质量评估（回答与标准答案 embedding 余弦相似度）")
    print("=" * 56)
    sims = []
    for item in test_set:
        q = item["question"]
        gt = item.get("ground_truth", "")
        if not gt:
            continue
        hits = hybrid_search(
            index, chunks, bm25, q,
            top_k=3, dist_threshold=dist_threshold, use_hybrid=use_hybrid,
        )
        context = "\n---\n".join(h["text"] for h in hits)
        sys_prompt = (
            "基于下面参考文档回答用户问题，只使用文档内信息，尽量简洁准确。\n"
            f"【参考文档】\n{context}"
        )
        try:
            ans = dashscope_chat([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ])
            sim = cosine(dashscope_embedding(ans), dashscope_embedding(gt))
            sims.append(sim)
            print(f"  Q: {q}")
            print(f"    相似度: {sim:.3f} | 回答: {ans[:60]}...")
        except Exception as e:
            print(f"  Q: {q} 评估失败: {e}")
    if sims:
        print(f"\n  回答相似度均值: {np.mean(sims):.3f}  (n={len(sims)})")
    return sims


def main():
    ap = argparse.ArgumentParser(description="RAG 检索与回答质量评估")
    ap.add_argument("--test-set", default="test_set.json")
    ap.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
    ap.add_argument("--compare", action="store_true", help="同时跑混合与纯向量并输出对比")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--with-answer", action="store_true")
    args = ap.parse_args()

    index, chunks, meta = load_vector_store()
    bm25, _ = load_bm25_index()
    if index is None or bm25 is None:
        index, chunks, msg, bm25 = build_vector_from_docs(350, 60, force_rebuild=True)
        print(msg)

    with open(args.test_set, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    if args.compare:
        print("\n>>> 模式对比：纯向量 / 混合检索 / 混合+Rerank（消融实验）")
        hit_hybrid, per_hybrid, total = run_eval(
            index, chunks, bm25, test_set, args.top_k, True, args.threshold)
        hit_vector, per_vector, _ = run_eval(
            index, chunks, bm25, test_set, args.top_k, False, args.threshold)
        hit_rerank, per_rerank, _ = run_eval(
            index, chunks, bm25, test_set, args.top_k, True, args.threshold,
            use_rerank=True)

        print("\n" + "=" * 56)
        print("  Recall@k 对比")
        print("=" * 56)
        print(f"  {'指标':<10}{'纯向量':>12}{'混合':>12}{'混合+Rerank':>16}")
        for k in (1, 3, 5):
            a = hit_vector[k] / total
            b = hit_hybrid[k] / total
            c = hit_rerank[k] / total
            print(f"  {'Recall@' + str(k):<10}{a:>12.2%}{b:>12.2%}{c:>16.2%}")

        print("\n  逐题命中对比（✓=命中 ✗=未命中）:")
        print(f"  {'#':<4}{'向量':<6}{'混合':<6}{'+Rerank':<10} 问题")
        for i, (pv, ph, pr) in enumerate(zip(per_vector, per_hybrid, per_rerank), 1):
            mv = "✓" if pv["hit"] else "✗"
            mh = "✓" if ph["hit"] else "✗"
            mr = "✓" if pr["hit"] else "✗"
            print(f"  {i:<4}{mv:<6}{mh:<6}{mr:<10} {ph['question']}")
            if len({mv, mh, mr}) > 1:
                print(f"       向量召回: {pv['retrieved']}")
                print(f"       混合召回: {ph['retrieved']}")
                print(f"       Rerank召回: {pr['retrieved']}")

        if args.with_answer:
            answer_quality(index, chunks, bm25, test_set, args.threshold, use_hybrid=True)
    else:
        use_hybrid = (args.mode == "hybrid")
        title = "混合检索评估" if use_hybrid else "纯向量检索评估"
        hit_at, per_item, total = run_eval(
            index, chunks, bm25, test_set, args.top_k, use_hybrid, args.threshold)
        print_report(title, hit_at, total, per_item)
        if args.with_answer:
            answer_quality(index, chunks, bm25, test_set, args.threshold, use_hybrid=use_hybrid)


if __name__ == "__main__":
    main()
