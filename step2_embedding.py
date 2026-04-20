"""
RAG 챗봇 2단계: 임베딩 + ChromaDB 벡터 저장
=============================================
[변경 이력]
 v1 (Ollama 버전): paraphrase-multilingual-MiniLM-L12-v2 (HuggingFace 무료)
 v2 (OpenAI 버전): text-embedding-3-small (OpenAI) ← 현재

[변경 이유]
 - LLM을 GPT-4o-mini로 교체함에 따라 임베딩도 같은 OpenAI 생태계로 통일
 - text-embedding-3-small은 MiniLM 대비 한국어/영어 혼합 검색 정확도가 높음
 - 1,784개 기준 임베딩 비용 약 $0.003 

실행 전 설치:
    pip install openai chromadb tqdm
"""
# -*- coding: utf-8 -*-
import json
import os
import chromadb
from tqdm import tqdm
from openai import OpenAI
# -*- coding: utf-8 -*-
# ── 설정 ──────────────────────────────────────────
CHUNKS_PATH = "chunks_translated.jsonl"
CHROMA_DIR     = "./chroma_db"       # 벡터DB 저장 폴더
COLLECTION     = "shopping_rag"
BATCH_SIZE     = 100                 # OpenAI API 배치 크기 (최대 2048)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
EMBED_MODEL    = "text-embedding-3-small"
# ──────────────────────────────────────────────────


# ── 임베딩 클라이언트 초기화 ───────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    OpenAI text-embedding-3-small로 임베딩 생성
    - 1536차원 벡터 반환
    - 한국어/영어 혼합 입력 모두 지원
    """
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]


# ── 청크 로드 (중복 doc_id 제거 포함) ─────────────
def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))

    # 중복 doc_id 제거 (리뷰 데이터 동일 ProductId 문제)
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            unique_chunks.append(c)

    print(f"청크 로드 완료: {len(chunks)}개 → 중복 제거 후 {len(unique_chunks)}개")
    return unique_chunks


# ── ChromaDB 구축 ──────────────────────────────────
def build_vectordb(chunks: list[dict]):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 기존 컬렉션 삭제 후 재생성 (재실행 시 중복 방지)
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION in existing:
        chroma_client.delete_collection(COLLECTION)
        print(f"기존 컬렉션 '{COLLECTION}' 삭제 (재구축)")

    collection = chroma_client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
    )

    # 배치 단위로 임베딩 + 저장
    total = len(chunks)
    for i in tqdm(range(0, total, BATCH_SIZE), desc="OpenAI 임베딩 저장 중"):
        batch     = chunks[i : i + BATCH_SIZE]
        texts     = [c["text"]   for c in batch]
        ids       = [c["doc_id"] for c in batch]
        metadatas = [{"source": c["source"], "category": c["category"]} for c in batch]

        embeddings = get_embeddings(texts)

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"\n✅ 저장 완료! 총 {collection.count()}개 벡터 → {CHROMA_DIR}/")
    print(f"   임베딩 모델: {EMBED_MODEL} (1536차원)")
    return collection


# ── 검색 테스트 ────────────────────────────────────
def test_search(collection, query: str, n: int = 3):
    """임베딩 품질 확인용 검색 테스트"""
    print(f"\n검색 테스트: '{query}'")
    print("-" * 50)

    query_emb = get_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        score = round(1 - dist, 4)  # 코사인 유사도로 변환
        print(f"\n[{i+1}] 유사도: {score} | source: {meta['source']} | category: {meta['category']}")
        print(f"    {doc[:150]}...")


if __name__ == "__main__":
    # 1. 청크 로드
    chunks = load_chunks(CHUNKS_PATH)

    # 2. 벡터DB 구축 (OpenAI 임베딩으로 재구축)
    collection = build_vectordb(chunks)

    # 3. 검색 테스트 — Ollama 버전과 유사도 점수 비교
    test_search(collection, "배송 며칠 걸려요?")
    test_search(collection, "환불 정책이 어떻게 돼요?")
    test_search(collection, "hiking boots waterproof")
