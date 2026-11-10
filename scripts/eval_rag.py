"""
eval_rag.py - RAG 检索准确率评测脚本
用法（在 D:\\tcm-ai 目录下，(.venv) 已激活）：
    python scripts\\eval_rag.py
"""
import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer

# ===== 配置 =====
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "tcm_corpus_v4"
EVAL_FILE = "data/eval/eval_set.jsonl"
TOP_K = 3


def main():
    # 加载模型
    print("正在加载 embedding 模型 ...")
    embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # 连接向量库
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = chroma_client.get_collection(COLLECTION_NAME)

    # 读评测集
    eval_items = []
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_items.append(json.loads(line))

    print(f"\n共 {len(eval_items)} 条评测问题，开始测试...\n")
    print("=" * 70)

    hit_count = 0
    results = []

    for item in eval_items:
        qid = item["id"]
        question = item["question"]
        expected = item["expected_chunk"]

        # 检索 top-K
        q_vec = embedder.encode([question], normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=q_vec, n_results=TOP_K)
        retrieved_ids = res["ids"][0]

        # 检查是否命中
        hit = expected in retrieved_ids
        if hit:
            hit_count += 1

        status = "✅ 命中" if hit else "❌ 未命中"
        results.append({
            "id": qid,
            "question": question,
            "expected": expected,
            "retrieved": retrieved_ids,
            "hit": hit,
        })

        print(f"{qid} [{item['category']}/{item['difficulty']}] {status}")
        print(f"  问题：{question}")
        print(f"  期望块：{expected}")
        print(f"  检索到：{retrieved_ids}")
        print()

    # 输出汇总
    print("=" * 70)
    accuracy = hit_count / len(eval_items) * 100
    print(f"\n📊 评测结果：{hit_count}/{len(eval_items)} 命中，准确率 {accuracy:.1f}%")

    # 按分类统计
    by_category = {}
    for r, item in zip(results, eval_items):
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = {"hit": 0, "total": 0}
        by_category[cat]["total"] += 1
        if r["hit"]:
            by_category[cat]["hit"] += 1

    print("\n按分类统计：")
    for cat, stats in by_category.items():
        acc = stats["hit"] / stats["total"] * 100
        print(f"  {cat}: {stats['hit']}/{stats['total']} ({acc:.0f}%)")


if __name__ == "__main__":
    main()
