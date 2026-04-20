"""
리뷰 데이터 한국어 번역
========================
실험 배경:
    기존 리뷰 데이터(300개)가 영어로 구성되어 있어 한국어 질문으로
    검색 시 의미적 유사도가 낮게 측정되는 문제가 있었다.
    예: "등산화 후기 알려줘" 쿼리로 영어 리뷰를 찾지 못하는 현상.
    GPT-4o-mini로 번역하여 한국어 쿼리와의 매칭 품질을 향상시킨다.

번역 비용:
    리뷰 300개 기준 약 $0.05 (GPT-4o-mini)

실행:
    set OPENAI_API_KEY=sk-...
    python translate_reviews.py
"""

import os
import re
import json
import time
from tqdm import tqdm
from openai import OpenAI

# ── 설정 ──────────────────────────────────────────
CHUNKS_PATH    = "chunks.jsonl"
OUTPUT_PATH    = "chunks_translated.jsonl"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BATCH_SIZE     = 5
SLEEP_BETWEEN  = 0.5
# ──────────────────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)


def clean_text(text: str) -> str:
    """HTML 태그 제거 및 공백 정리"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_english_content(text: str) -> str:
    """
    "리뷰 요약: XXX 내용: YYY" 형태에서 영어 본문만 추출
    한국어 접두사(리뷰 요약:, 내용:)는 번역 후 다시 붙임
    """
    # "내용:" 이후 텍스트 추출
    if "내용:" in text:
        content = text.split("내용:", 1)[1].strip()
    elif "리뷰 요약:" in text:
        content = text.split("리뷰 요약:", 1)[1].strip()
    else:
        content = text
    return clean_text(content)


def translate_batch(texts: list) -> list:
    """GPT-4o-mini로 영어 텍스트 배치 번역"""
    numbered = "\n---\n".join([f"[{i+1}] {t}" for i, t in enumerate(texts)])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional Korean translator specializing in product reviews. "
                    "Translate each numbered English text to natural, colloquial Korean. "
                    "Rules:\n"
                    "- Keep product names and brand names in English\n"
                    "- Use natural Korean expressions, not literal translations\n"
                    "- Return ONLY the translated texts in the same [N] format\n"
                    "- Each text is separated by ---"
                )
            },
            {
                "role": "user",
                "content": f"Translate these product reviews to Korean:\n\n{numbered}"
            }
        ]
    )

    result = response.choices[0].message.content.strip()

    # [1], [2] 형태로 파싱
    translated = []
    parts = re.split(r'\[\d+\]', result)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == len(texts):
        return parts
    return texts  # 파싱 실패 시 원본 반환


def load_chunks(path: str) -> tuple:
    non_review, review_chunks = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item["source"] == "review":
                review_chunks.append(item)
            else:
                non_review.append(item)
    print(f"FAQ + 상품: {len(non_review)}개 (번역 불필요)")
    print(f"리뷰: {len(review_chunks)}개 (번역 대상)")
    return non_review, review_chunks


def translate_reviews(review_chunks: list) -> list:
    translated_chunks = []
    success, fail = 0, 0

    for i in tqdm(range(0, len(review_chunks), BATCH_SIZE), desc="리뷰 번역 중"):
        batch = review_chunks[i:i+BATCH_SIZE]

        # 영어 본문만 추출해서 번역
        english_texts = [extract_english_content(c["text"]) for c in batch]

        try:
            translated_texts = translate_batch(english_texts)
            for chunk, translated_text in zip(batch, translated_texts):
                translated_chunks.append({
                    "doc_id": chunk["doc_id"],
                    "text": f"리뷰 내용: {translated_text}",
                    "text_original": chunk["text"],
                    "source": chunk["source"],
                    "category": chunk["category"],
                    "translated": True
                })
                success += 1
        except Exception as e:
            print(f"\n번역 실패 (batch {i}): {e} -> 원본 사용")
            for chunk in batch:
                translated_chunks.append({**chunk, "translated": False})
                fail += 1

        time.sleep(SLEEP_BETWEEN)

    print(f"\n번역 완료 - 성공: {success}개 / 실패: {fail}개")
    return translated_chunks


def preview_translation(translated_reviews: list, n: int = 3):
    print("\n번역 샘플 확인:")
    print("-" * 60)
    count = 0
    for item in translated_reviews:
        if item.get("translated") and count < n:
            print(f"[원본] {item['text_original'][:100]}...")
            print(f"[번역] {item['text'][:100]}...")
            print()
            count += 1


if __name__ == "__main__":
    print("=" * 60)
    print("리뷰 데이터 한국어 번역")
    print("목적: 영어 리뷰를 한국어로 변환하여 검색 품질 향상")
    print("모델: GPT-4o-mini")
    print("=" * 60)

    non_review, review_chunks = load_chunks(CHUNKS_PATH)

    print(f"\n총 {len(review_chunks)}개 리뷰 번역 시작...")
    translated_reviews = translate_reviews(review_chunks)

    preview_translation(translated_reviews)

    all_chunks = non_review + translated_reviews
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n저장 완료: {OUTPUT_PATH} ({len(all_chunks)}개 청크)")
    
