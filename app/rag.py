"""
rag.py - RAG 问答核心
用法（在 D:\\tcm-ai 目录下，(.venv) 已激活）：
    python app\\rag.py
"""
import json
import os

# 国内镜像，加速 huggingface 模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# ===== 配置 =====
CHROMA_DIR = "data/chroma"
CHUNKS_FILE = "data/processed/chunks.jsonl"
COLLECTION_NAME = "tcm_corpus_v4"
TOP_K = 3
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


# 加载中文语义模型（首次运行自动从国内镜像下载，约 100MB）
print(f"正在加载 embedding 模型 {EMBED_MODEL} ...")
embedder = SentenceTransformer(EMBED_MODEL)
print("模型加载完成")


def embed_texts(texts):
    """把文本列表转成向量列表"""
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# DeepSeek 客户端（兼容 OpenAI 格式）
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# ChromaDB 本地持久化客户端
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def build_index():
    """从 chunks.jsonl 建/重建向量库"""
    col = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # 清空旧数据
    if col.count() > 0:
        all_ids = col.get()["ids"]
        col.delete(ids=all_ids)

    docs, ids, metas = [], [], []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(rec["text"])
            ids.append(rec["id"])
            metas.append({
                "source_file": rec["source_file"],
                "category": rec["category"],
                "title": rec["title"],
            })

    embeddings = embed_texts(docs)
    col.add(embeddings=embeddings, documents=docs, ids=ids, metadatas=metas)
    print(f"已入库 {col.count()} 块语料")


def ask(question: str) -> str:
    """检索相关语料 + 调 DeepSeek 生成回答"""
    col = chroma_client.get_collection(COLLECTION_NAME)

    # 1. 向量检索 top-K
    query_vec = embed_texts([question])[0]
    results = col.query(query_embeddings=[query_vec], n_results=TOP_K)
    contexts = results["documents"][0]
    sources = sorted(set(m["source_file"] for m in results["metadatas"][0]))

    # 2. 拼 prompt
    context_text = "\n\n".join(
        [f"[资料{i+1}]\n{c}" for i, c in enumerate(contexts)]
    )
    prompt = f"""你是一位中医养生科普助手。请严格依据下面提供的资料回答用户问题。
规则：
1. 【最重要】只能使用资料里明确提到的内容，绝对不能自己补充、推测或扩展资料里没有的信息
2. 资料里没有的内容，直接说"资料中未提及"，不要编造
3. 不做医疗诊断，不承诺疗效，必要时提示就医
4. 回答用简洁的中文，口语化一点，分点列出

【参考资料】
{context_text}

【用户问题】{question}"""

    # 3. 调 DeepSeek 生成回答
    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = resp.choices[0].message.content

    # 4. 附上来源
    answer += f"\n\n—— 参考资料：{', '.join(sources)}"
    return answer


if __name__ == "__main__":
    build_index()
    print("\nRAG 已就绪！输入问题开始对话（输入 q 退出）")
    while True:
        q = input("\n你问：").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        try:
            print(f"\nAI 老中医：\n{ask(q)}")
        except Exception as e:
            print(f"\n出错了：{e}")
