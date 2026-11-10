"""
web_app.py - AI 老中医养生助手 Web 界面
用法（在 D:\\tcm-ai 目录下，(.venv) 已激活）：
    streamlit run app\\web_app.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ===== 配置 =====
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "tcm_corpus_v4"
TOP_K = 3


@st.cache_resource
def load_models():
    """加载模型和数据库（只加载一次，缓存复用）"""
    embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = chroma_client.get_collection(COLLECTION_NAME)
    llm = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    return embedder, col, llm


def ask(question: str, embedder, col, llm) -> str:
    """RAG 问答"""
    query_vec = embedder.encode([question], normalize_embeddings=True).tolist()
    results = col.query(query_embeddings=query_vec, n_results=TOP_K)
    contexts = results["documents"][0]
    sources = sorted(set(m["source_file"] for m in results["metadatas"][0]))

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

    resp = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = resp.choices[0].message.content
    answer += f"\n\n—— 参考资料：{', '.join(sources)}"
    return answer


# ===== 页面 =====
st.set_page_config(page_title="AI 老中医养生助手", page_icon="🌿")
st.title("🌿 AI 老中医养生助手")
st.caption("基于中医养生知识库的 RAG 问答系统 · 仅供养生参考，不构成医疗建议")

# 加载模型
with st.spinner("正在加载知识库..."):
    embedder, col, llm = load_models()

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是 AI 老中医养生助手，有什么养生问题可以问我～"}
    ]

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
if prompt := st.chat_input("输入你的养生问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                answer = ask(prompt, embedder, col, llm)
            except Exception as e:
                answer = f"出错了：{e}"
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
