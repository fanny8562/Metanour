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

    # 如果数据库为空，自动建库
    col = chroma_client.get_or_create_collection(COLLECTION_NAME)
    if col.count() == 0:
        import json
        with open("data/processed/chunks.jsonl", encoding="utf-8") as f:
            docs, ids, metas = [], [], []
            for line in f:
                rec = json.loads(line)
                docs.append(rec["text"])
                ids.append(rec["id"])
                metas.append({
                    "source_file": rec["source_file"],
                    "category": rec["category"],
                    "title": rec["title"],
                })
        embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
        col.add(embeddings=embeddings, documents=docs, ids=ids, metadatas=metas)

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


# ===== 体质自测题目 =====
QUIZ_QUESTIONS = [
    # 气虚质
    {"q": "您容易疲乏、没力气吗？", "type": "气虚"},
    {"q": "您容易气短（呼吸短促、接不上气）吗？", "type": "气虚"},
    {"q": "您说话声音低弱无力吗？", "type": "气虚"},
    {"q": "您稍微活动就容易出汗吗？", "type": "气虚"},
    # 阳虚质
    {"q": "您手脚发凉吗？", "type": "阳虚"},
    {"q": "您比一般人耐受不了寒冷吗？", "type": "阳虚"},
    # 阴虚质
    {"q": "您手脚心发热吗？", "type": "阴虚"},
    {"q": "您皮肤或口唇干燥吗？", "type": "阴虚"},
    {"q": "您容易失眠、做梦多吗？", "type": "阴虚"},
]

# 体质对应功法推荐
CONSTITUTION_GUIDE = {
    "气虚质": {
        "desc": "容易疲乏、气短、自汗，性格偏内向",
        "exercise": "八段锦、太极拳",
        "diet": "宜食山药、黄芪、党参、大枣，少食生冷",
    },
    "阳虚质": {
        "desc": "畏寒怕冷、手脚冰凉，喜热饮",
        "exercise": "五禽戏、八段锦",
        "diet": "宜食羊肉、韭菜、生姜、桂圆，少食西瓜、苦瓜",
    },
    "阴虚质": {
        "desc": "口燥咽干、手脚心热，容易失眠",
        "exercise": "六字诀、站桩",
        "diet": "宜食百合、梨、银耳，少食辛辣燥热",
    },
    "平和质": {
        "desc": "形体匀称、精力充沛，很少生病",
        "exercise": "五禽戏、八段锦都可以",
        "diet": "饮食均衡即可，无需特别忌口",
    },
}


def run_quiz():
    """体质自测页面"""
    st.subheader("📋 体质自测")
    st.caption("回答下面 9 个问题，帮你判断体质倾向，推荐适合的养生功法")

    answers = {}
    for i, item in enumerate(QUIZ_QUESTIONS):
        answers[i] = st.radio(
            f"{i+1}. {item['q']}",
            ["是", "否"],
            key=f"q_{i}",
            horizontal=True,
        )

    if st.button("提交，看我的体质结果", type="primary"):
        # 统计得分
        scores = {"气虚": 0, "阳虚": 0, "阴虚": 0}
        for i, item in enumerate(QUIZ_QUESTIONS):
            if answers[i] == "是":
                scores[item["type"]] += 1

        # 判断体质
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        if max_score <= 1:
            result = "平和质"
        else:
            result = f"{max_type}质"

        # 输出结果
        st.divider()
        st.subheader(f"你的体质倾向：{result}")
        guide = CONSTITUTION_GUIDE[result]

        st.markdown(f"**特点：** {guide['desc']}")
        st.markdown(f"**推荐功法：** {guide['exercise']}")
        st.markdown(f"**饮食建议：** {guide['diet']}")

        st.info("以上结果仅供养生参考，不能替代专业医生诊断。如有明显不适，请及时就医。")


# ===== 页面 =====
st.set_page_config(page_title="AI 老中医养生助手", page_icon="🌿")
st.title("🌿 AI 老中医养生助手")
st.caption("基于中医养生知识库的 RAG 问答系统 · 仅供养生参考，不构成医疗建议")

# 加载模型
with st.spinner("正在加载知识库..."):
    embedder, col, llm = load_models()

# 两个标签页
tab1, tab2, tab3 = st.tabs(["💬 养生问答", "📋 体质自测", "🩺 老中医问诊"])

with tab1:
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

with tab2:
    run_quiz()

with tab3:
    st.subheader("🩺 老中医问诊")
    st.caption("我来问你几个问题，帮你判断体质倾向，最后给养生建议")

    # 问诊问题列表
    CONSULT_QUESTIONS = [
        "你平时容易疲乏、没力气吗？",
        "你手脚容易发凉、怕冷吗？",
        "你手脚心容易发热、口干吗？",
        "你睡眠怎么样？容易失眠吗？",
        "你平时容易出汗吗？（不活动也出汗）",
    ]

    # 初始化问诊状态
    if "consult_step" not in st.session_state:
        st.session_state.consult_step = 0
        st.session_state.consult_answers = []
        st.session_state.consult_done = False

    # 显示问诊历史
    if st.session_state.consult_step == 0 and not st.session_state.consult_done:
        st.write("您好，我是 AI 老中医。接下来问您几个问题，请如实回答～")

    for i, ans in enumerate(st.session_state.consult_answers):
        with st.chat_message("assistant"):
            st.write(CONSULT_QUESTIONS[i])
        with st.chat_message("user"):
            st.write(ans)

    # 问诊未结束时显示当前问题
    if not st.session_state.consult_done and st.session_state.consult_step < len(CONSULT_QUESTIONS):
        current_q = CONSULT_QUESTIONS[st.session_state.consult_step]
        with st.chat_message("assistant"):
            st.write(current_q)

        user_ans = st.chat_input("请回答...", key="consult_input")
        if user_ans:
            st.session_state.consult_answers.append(user_ans)
            st.session_state.consult_step += 1
            st.rerun()

    # 问诊结束，生成结果
    if st.session_state.consult_step >= len(CONSULT_QUESTIONS) and not st.session_state.consult_done:
        with st.spinner("正在辨证分析..."):
            answers_text = "\n".join([
                f"问题{i+1}：{q}\n回答：{a}"
                for i, (q, a) in enumerate(zip(CONSULT_QUESTIONS, st.session_state.consult_answers))
            ])
            prompt = f"""你是一位中医老中医。根据下面用户的回答，判断他的体质倾向（气虚/阳虚/阴虚/平和），并给出养生建议。

用户回答：
{answers_text}

请按以下格式回答：
1. 体质判断：你的体质倾向是XX质
2. 特点：用2-3句话描述这个体质的特点
3. 推荐功法：推荐适合的养生功法
4. 饮食建议：简单说一下饮食宜忌
5. 日常调理：1-2个日常小建议

注意：仅供养生参考，不构成医疗诊断。"""

            resp = llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            result = resp.choices[0].message.content

        st.session_state.consult_done = True
        st.session_state.consult_result = result

    # 显示问诊结果
    if st.session_state.consult_done:
        st.divider()
        st.subheader("📝 辨证结果")
        st.write(st.session_state.get("consult_result", ""))
        st.info("以上结果仅供养生参考，不能替代专业医生诊断。如有明显不适，请及时就医。")

        if st.button("重新问诊"):
            st.session_state.consult_step = 0
            st.session_state.consult_answers = []
            st.session_state.consult_done = False
            st.rerun()
