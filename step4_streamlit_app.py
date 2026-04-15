"""
RAG 챗봇 4단계: Streamlit UI
==============================
실행 방법:
    pip install streamlit
    streamlit run step4_streamlit_app.py

[구성]
 - 사이드바: 프로젝트 소개, 기술 스택, 빠른 질문 버튼
 - 메인: 대화형 챗봇 (참고 문서 출처 표시)
 - 세션 기반 대화 히스토리 유지
"""

import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import cohere

# ── 설정 ──────────────────────────────────────────
CHROMA_DIR     = "./chroma_db"
COLLECTION     = "shopping_rag"
# 로컬: os.environ / 배포: Streamlit Cloud st.secrets
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY") or st.secrets.get("COHERE_API_KEY", "")
cohere_client = cohere.ClientV2(COHERE_API_KEY)
# ──────────────────────────────────────────────────

st.set_page_config(
    page_title="쇼핑몰 AI 고객센터",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'Noto Sans KR', sans-serif; }

/* 전체 배경 */
.stApp {
    background-color: #F7F8FA;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background-color: #1A1F2E;
    border-right: 1px solid #2D3346;
}
section[data-testid="stSidebar"] * {
    color: #E8EAF0 !important;
}

/* 사이드바 구분선 */
section[data-testid="stSidebar"] hr {
    border-color: #2D3346;
}

/* 메인 헤더 */
.main-header {
    background: linear-gradient(135deg, #1A1F2E 0%, #2D3346 100%);
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.main-header h1 {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.3px;
}
.main-header p {
    color: #8B93A8;
    font-size: 13px;
    margin: 4px 0 0 0;
}
.status-dot {
    width: 8px;
    height: 8px;
    background: #4ADE80;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* 사용자 메시지 */
.user-message {
    background: #1A1F2E;
    color: #FFFFFF;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px auto;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
    width: fit-content;
    margin-left: auto;
}

/* 봇 메시지 */
.bot-message {
    background: #FFFFFF;
    color: #1A1F2E;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px auto 8px 0;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid #E8EAF0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    width: fit-content;
}

/* 출처 태그 */
.source-tag {
    display: inline-block;
    background: #F0F4FF;
    color: #4A6CF7;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    padding: 3px 8px;
    border-radius: 4px;
    margin: 2px 3px 2px 0;
    border: 1px solid #D4DEFF;
}

/* 출처 컨테이너 */
.sources-container {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #E8EAF0;
}
.sources-label {
    font-size: 11px;
    color: #8B93A8;
    margin-bottom: 6px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* 빠른 질문 버튼 */
.stButton > button {
    background: #2D3346 !important;
    color: #E8EAF0 !important;
    border: 1px solid #3D4560 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    padding: 6px 12px !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3D4560 !important;
    border-color: #4A6CF7 !important;
}

/* 입력창 */
.stChatInput input {
    background: #FFFFFF !important;
    border: 1px solid #E8EAF0 !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    color: #1A1F2E !important;
}

/* 기술 스택 배지 */
.tech-badge {
    display: inline-block;
    background: #2D3346;
    color: #8B93A8;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    padding: 3px 8px;
    border-radius: 4px;
    margin: 2px;
    border: 1px solid #3D4560;
}

/* 채팅 컨테이너 */
.chat-container {
    background: #F7F8FA;
    border-radius: 16px;
    padding: 16px;
    min-height: 400px;
}

/* 환영 메시지 */
.welcome-box {
    background: #FFFFFF;
    border: 1px solid #E8EAF0;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 20px auto;
    max-width: 480px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.welcome-box h3 {
    color: #1A1F2E;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
}
.welcome-box p {
    color: #8B93A8;
    font-size: 13px;
    line-height: 1.6;
}

/* 스크롤바 */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2D3346; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── RAG 초기화 (캐시) ──────────────────────────────
@st.cache_resource
def init_rag():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    return vectorstore, llm

vectorstore, llm = init_rag()

# [5단계 추가] 멀티턴 대화 지원
# 이전 대화 히스토리를 프롬프트에 포함시켜 문맥 기반 답변 가능
# 예: "방수 등산화 추천해줘" → "그 중 제일 가벼운 거 뭐야?" 연속 질문 처리
PROMPT_TEMPLATE = """You are a Korean shopping mall customer service chatbot.
Answer ONLY using the exact information from the reference documents below.
Rules:
- Answer in Korean only.
- NEVER invent product names, prices, or any information not in the documents.
- If documents contain relevant product info, quote it directly.
- If the answer is not in the documents, respond ONLY with: "해당 내용은 고객센터(1588-0000)로 문의해 주세요."
- Do NOT mix other languages into Korean sentences.
- Use the conversation history to understand follow-up questions in context.

[Reference Documents]
{context}

[Conversation History]
{history}

[Customer Question]
{question}

[Answer]"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "history", "question"]
)

KO_TO_EN = {
    "등산화": "hiking boots", "방수": "waterproof",
    "신발": "shoes", "자켓": "jacket", "부츠": "boots",
    "운동화": "sneakers", "샌들": "sandals",
    "농구화": "basketball shoes", "런닝화": "running shoes",
    "트레킹화": "trekking boots", "남성": "men", "여성": "women",
    "겨울": "winter", "방한": "insulated", "경량": "lightweight",
}

def translate_query(q):
    for ko, en in KO_TO_EN.items():
        q = q.replace(ko, en)
    return q

def build_history(messages: list, max_turns: int = 3) -> str:
    """
    최근 N턴의 대화 히스토리를 문자열로 변환
    - max_turns: 포함할 최대 대화 턴 수 (너무 많으면 토큰 낭비)
    - 현재 질문 제외, 이전 대화만 포함
    """
    if not messages:
        return "없음"
    recent = messages[-(max_turns * 2):]  # user/assistant 쌍으로 계산
    history_lines = []
    for msg in recent:
        role = "고객" if msg["role"] == "user" else "챗봇"
        history_lines.append(f"{role}: {msg['content']}")
    return "\n".join(history_lines) if history_lines else "없음"


def get_answer(question: str, chat_history: list = None):
    # 상품 추천 관련 키워드
    product_keywords = [
        "추천", "상품", "등산화", "신발", "자켓", "부츠", "옷", "의류",
        "농구화", "운동화", "런닝화", "트레킹화", "스니커즈",
        "boots", "shoes", "jacket", "hiking", "waterproof", "sneakers"
    ]

    # 리뷰/후기 관련 키워드 (번역된 리뷰 데이터 검색용)
    # 번역 전에는 한국어 질문으로 영어 리뷰를 찾지 못했던 문제를 해결
    review_keywords = [
        "후기", "리뷰", "사용기", "사용 후기", "평가", "어때", "어떤가요",
        "써봤어", "써봤나요", "만족", "추천해", "어떤지"
    ]

    # 후속 질문 감지
    followup_keywords = ["그거", "그중", "그 중", "거기서", "그것", "위에서", "방금"]
    is_followup = any(kw in question for kw in followup_keywords)

    search_question = question
    if is_followup and chat_history:
        recent_user_msgs = [m["content"] for m in chat_history if m["role"] == "user"][-3:]
        search_question = " ".join(recent_user_msgs) + " " + question

    is_product_query = any(kw in search_question.lower() for kw in product_keywords)
    is_review_query  = any(kw in search_question for kw in review_keywords)
    search_query = translate_query(search_question) if (is_product_query or is_review_query) else question

    all_docs = vectorstore.similarity_search(search_query, k=20)

    # 질문 유형에 따라 검색 소스 결정
    # "등산화 후기"처럼 상품+리뷰 키워드가 동시에 감지되면 세 소스 모두 검색
    if is_review_query and is_product_query:
        allowed = {"review", "product", "faq"}
    elif is_review_query:
        allowed = {"review", "faq"}
    elif is_product_query:
        allowed = {"product", "faq"}
    else:
        allowed = {"faq"}

    filtered_docs = [d for d in all_docs if d.metadata.get("source") in allowed]
    if not filtered_docs:
        filtered_docs = all_docs

    # Re-ranking: Cohere Rerank API로 관련성 높은 문서 재정렬
    # 벡터 유사도만으로는 순서가 부정확할 수 있어 Cross-Encoder 방식으로 재정렬
    if len(filtered_docs) > 1:
        try:
            rerank_docs = filtered_docs[:20]
            response = cohere_client.rerank(
                model="rerank-v3.5",
                query=question,
                documents=[d.page_content for d in rerank_docs],
                top_n=5,
            )
            docs = [rerank_docs[r.index] for r in response.results]
        except Exception:
            docs = filtered_docs[:5]
    else:
        docs = filtered_docs[:5]

    context = "\n\n".join([
        f"[문서{i+1} | {d.metadata.get('source')} | {d.metadata.get('category')}]\n{d.page_content}"
        for i, d in enumerate(docs)
    ])

    # 히스토리 포함하여 프롬프트 구성
    history_str = build_history(chat_history or [])

    chain = (
        {
            "context": lambda _: context,
            "history": lambda _: history_str,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    sources = [{"source": d.metadata.get("source"), "category": d.metadata.get("category")} for d in docs]
    return answer, sources


# ── 세션 초기화 ────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quick_question" not in st.session_state:
    st.session_state.quick_question = None


# ── 사이드바 ───────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛍️ 쇼핑몰 AI 고객센터")
    st.markdown("---")

    st.markdown("**프로젝트 소개**")
    st.markdown("""
    <div style='font-size:12px; color:#8B93A8; line-height:1.7;'>
    FAQ, 상품 정보, 고객 리뷰를 기반으로<br>
    고객 질문에 자동 답변하는 RAG 챗봇입니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**기술 스택**")
    techs = ["GPT-4o-mini", "text-embedding-3-small", "ChromaDB", "LangChain", "Streamlit"]
    for t in techs:
        st.markdown(f"<span class='tech-badge'>{t}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 데이터 현황**")
    st.markdown("""
    <div style='font-size:12px; color:#8B93A8; line-height:2;'>
    🗂️ FAQ: 80개<br>
    👟 상품: 247개<br>
    ⭐ 리뷰: 300개<br>
    🔢 총 벡터: 1,784개
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚡ 빠른 질문**")

    quick_questions = [
        "📦 배송 며칠 걸려요?",
        "🔄 환불 어떻게 하나요?",
        "🎟️ 쿠폰 적립금 같이 쓸 수 있나요?",
        "🥾 방수 등산화 추천해줘",
        "💳 결제 수단은 어떻게 되나요?",
        "👟 여성 하이킹 부츠 추천해줘",
    ]
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}"):
            st.session_state.quick_question = q.split(" ", 1)[1]  # 이모지 제거

    st.markdown("---")
    if st.button("🗑️ 대화 초기화", key="clear"):
        st.session_state.messages = []
        st.rerun()


# ── 메인 영역 ──────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div>
        <h1>🛍️ 쇼핑몰 AI 고객센터 <span class='status-dot'></span></h1>
        <p>RAG 기반 자동 답변 시스템 · FAQ + 상품정보 + 고객리뷰 통합 검색</p>
    </div>
</div>
""", unsafe_allow_html=True)

# 대화 없을 때 환영 메시지
if not st.session_state.messages:
    st.markdown("""
    <div class='welcome-box'>
        <h3>👋 무엇을 도와드릴까요?</h3>
        <p>배송, 환불, 상품 추천 등<br>궁금한 점을 자유롭게 물어보세요.<br>
        왼쪽 빠른 질문 버튼도 이용하실 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)

# 대화 히스토리 렌더링
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            tags = "".join([
                f"<span class='source-tag'>{s['source']} · {s['category']}</span>"
                for s in msg["sources"]
            ])
            sources_html = f"""
            <div class='sources-container'>
                <div class='sources-label'>📎 참고 문서</div>
                {tags}
            </div>"""
        st.markdown(
            f"<div class='bot-message'>{msg['content']}{sources_html}</div>",
            unsafe_allow_html=True
        )

# 빠른 질문 처리
if st.session_state.quick_question:
    question = st.session_state.quick_question
    st.session_state.quick_question = None
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("답변 생성 중..."):
        answer, sources = get_answer(question, st.session_state.messages)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.rerun()

# 채팅 입력
if user_input := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("답변 생성 중..."):
        answer, sources = get_answer(user_input, st.session_state.messages)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.rerun()
