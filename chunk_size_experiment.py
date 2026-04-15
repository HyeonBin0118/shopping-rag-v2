"""
청크 사이즈 실험 v2: 300 / 500 / 700 비교
==========================================
실험 배경:
  chunk_size=500은 임의로 설정한 값으로, 최적값인지 검증된 적 없음.
  청크가 너무 크면 -> 관련 없는 내용이 섞여 검색 정확도 하락
  청크가 너무 작으면 -> 문맥이 잘려 답변 품질 하락
  -> 300/500/700 세 가지로 실험하여 RAGAS 수치로 최적값 도출

실험 설계:
  - FAQ(80개): 짧아서 청크 사이즈 무관 -> 모든 실험에 동일하게 포함
  - product(643개) + review(1359개): 청크 사이즈별로 재청킹
  - 통제 변수: overlap=50, text-embedding-3-small, GPT-4o-mini

실행:
  set OPENAI_API_KEY=sk-...
  python chunk_size_experiment.py
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

import chromadb
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# ── 설정 ──────────────────────────────────────────
CHUNKS_PATH    = "chunks.jsonl"
CHROMA_DIR     = "./chroma_db_experiment"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
CHUNK_SIZES    = [300, 500, 700]
CHUNK_OVERLAP  = 50
# ──────────────────────────────────────────────────

TEST_SET = [
    {"question": "배송은 보통 며칠 걸려요?",
     "ground_truth": "일반적으로 결제 완료 후 2~3 영업일 내에 배송됩니다."},
    {"question": "환불하고 싶은데 어떻게 해야 하나요?",
     "ground_truth": "마이페이지 > 반품/교환 신청 후 상품을 반송하시면 확인 후 환불됩니다."},
    {"question": "쿠폰이랑 적립금 같이 쓸 수 있나요?",
     "ground_truth": "쿠폰과 적립금은 동시에 사용하실 수 있습니다."},
    {"question": "주말에도 배송되나요?",
     "ground_truth": "주말 및 공휴일에는 배송이 이루어지지 않습니다."},
    {"question": "환불 기간은 얼마나 걸리나요?",
     "ground_truth": "상품 회수 후 2~5 영업일 내에 환불 처리됩니다."},
    {"question": "배송비는 얼마예요?",
     "ground_truth": "기본 배송비는 3,000원이며, 일부 상품은 무료배송이 적용됩니다."},
    {"question": "적립금은 어떻게 사용하나요?",
     "ground_truth": "결제 페이지에서 사용할 적립금 금액을 입력하시면 됩니다."},
    {"question": "회원가입 없이 주문할 수 있나요?",
     "ground_truth": "비회원 주문도 가능합니다. 주문 시 이메일을 입력하시면 주문 내역을 확인할 수 있습니다."},
]

PROMPT_TEMPLATE = """You are a Korean shopping mall customer service chatbot.
Answer ONLY using the exact information from the reference documents below.
Rules:
- Answer in Korean only.
- NEVER invent information not in the documents.
- If the answer is not in the documents, respond ONLY with: "해당 내용은 고객센터(1588-0000)로 문의해 주세요."

[Reference Documents]
{context}

[Customer Question]
{question}

[Answer]"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def load_chunks(path):
    """FAQ / product+review 분리 로드"""
    faq_chunks = []
    long_chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item["source"] == "faq":
                faq_chunks.append(item)
            else:
                long_chunks.append(item)
    print(f"FAQ: {len(faq_chunks)}개 (고정)")
    print(f"product+review: {len(long_chunks)}개 (재청킹 대상)")
    return faq_chunks, long_chunks


def rechunk_long(long_chunks, chunk_size, overlap):
    """product/review를 새 chunk_size로 재청킹"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    result = []
    seen = set()
    for item in long_chunks:
        for i, text in enumerate(splitter.split_text(item["text"])):
            doc_id = f"{item['doc_id']}_{chunk_size}_{i}"
            if doc_id not in seen:
                seen.add(doc_id)
                result.append({
                    "doc_id": doc_id,
                    "text": text,
                    "source": item["source"],
                    "category": item["category"]
                })
    return result


def build_vectordb(all_chunks, chunk_size, embeddings):
    """청크 사이즈별 컬렉션 구축"""
    collection_name = f"rag_{chunk_size}"
    oai_client = OpenAI(api_key=OPENAI_API_KEY)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing:
        chroma_client.delete_collection(collection_name)

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    for i in tqdm(range(0, len(all_chunks), 100), desc=f"  임베딩 중 (chunk={chunk_size})"):
        batch = all_chunks[i:i+100]
        texts = [c["text"] for c in batch]
        ids   = [c["doc_id"] for c in batch]
        metas = [{"source": c["source"], "category": c["category"]} for c in batch]
        resp  = oai_client.embeddings.create(model="text-embedding-3-small", input=texts)
        embs  = [d.embedding for d in resp.data]
        collection.add(ids=ids, documents=texts, embeddings=embs, metadatas=metas)

    print(f"  벡터DB 구축 완료: {collection.count()}개 벡터")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def run_rag(question, vectorstore, llm):
    all_docs = vectorstore.similarity_search(question, k=20)
    docs = [d for d in all_docs if d.metadata.get("source") == "faq"][:5]
    if not docs:
        docs = all_docs[:5]
    context = "\n\n".join([f"[문서{i+1}]\n{d.page_content}" for i, d in enumerate(docs)])
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain.invoke(question), [d.page_content for d in docs]


def evaluate_chunk(chunk_size, vectorstore, llm, eval_llm, eval_emb):
    print(f"\n  RAGAS 평가 중 (chunk_size={chunk_size})...")
    questions, answers, contexts, ground_truths = [], [], [], []
    for item in TEST_SET:
        answer, ctx = run_rag(item["question"], vectorstore, llm)
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(item["ground_truth"])
        print(f"    Q: {item['question'][:20]}... -> {answer[:50]}...")

    dataset = Dataset.from_dict({
        "question": questions, "answer": answers,
        "contexts": contexts, "ground_truth": ground_truths,
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm, embeddings=eval_emb,
    )

    def score(key):
        val = result[key]
        if isinstance(val, list):
            val = [v for v in val if v is not None]
            return round(sum(val)/len(val), 4) if val else 0.0
        return round(float(val), 4)

    return {
        "chunk_size": chunk_size,
        "Faithfulness": score("faithfulness"),
        "Answer Relevancy": score("answer_relevancy"),
        "Context Precision": score("context_precision"),
        "Context Recall": score("context_recall"),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("청크 사이즈 실험: 300 / 500 / 700")
    print("실험 목적: chunk_size=500이 최적값인지 데이터로 검증")
    print("통제 변수: overlap=50, text-embedding-3-small, GPT-4o-mini")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    llm      = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    eval_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

    faq_chunks, long_chunks = load_chunks(CHUNKS_PATH)
    results = []

    for chunk_size in CHUNK_SIZES:
        print(f"\n{'='*40}")
        print(f"[{CHUNK_SIZES.index(chunk_size)+1}/3] chunk_size = {chunk_size}")
        print(f"{'='*40}")

        rechunked  = rechunk_long(long_chunks, chunk_size, CHUNK_OVERLAP)
        all_chunks = faq_chunks + rechunked
        print(f"  전체 청크: {len(all_chunks)}개 (FAQ {len(faq_chunks)} + 재청킹 {len(rechunked)})")

        vectorstore = build_vectordb(all_chunks, chunk_size, embeddings)
        scores = evaluate_chunk(chunk_size, vectorstore, llm, eval_llm, embeddings)
        results.append(scores)

    print("\n" + "=" * 60)
    print("실험 결과")
    print("=" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    best = df.loc[df["Context Precision"].idxmax()]
    print(f"\n최적 chunk_size: {int(best['chunk_size'])} (Context Precision 기준)")

    df.to_csv("chunk_experiment_results.csv", index=False, encoding="utf-8-sig")
    print("\n결과 저장 완료: chunk_experiment_results.csv")
    print("이 수치를 README v2에 넣어!")
