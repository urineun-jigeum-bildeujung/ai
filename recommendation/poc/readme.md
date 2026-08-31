# 상품 추천 시스템 POC

## 목적
백엔드 목데이터 제공 전, 더미 데이터 + 룰베이스 구현으로 전체 파이프라인이
스키마대로 엔드투엔드로 연결되는지 검증.

## 실행 방법
```
cd poc
python3 main.py
```

## 구조
| 파일 | 역할 | 검증 범위 밖(나중에 교체) |
|---|---|---|
| `dummy_data.py` | pet_profile / product_master / reviews 더미 데이터 | 실제 목데이터/합성데이터로 교체 |
| `sentiment.py` | 룰베이스 감성분석 (KcELECTRA 대체) | KcELECTRA 파인튜닝 모델로 교체 |
| `aspect_tagging.py` | aspect 키워드 태깅 | 필요 시 모델 기반으로 고도화 |
| `allergy_filter.py` | 알러지 매칭(역추천) | 그대로 유지 (규칙 기반이 맞는 영역) |
| `recommend.py` | 룰베이스 추천 스코어링 (DeepFM 대체) | DeepFM 학습 모델로 교체 |
| `main.py` | 전체 파이프라인 실행 | 구조는 유지, 내부 호출만 교체 |

## 교체 시 유의사항
- `sentiment.py`의 `analyze_sentiment(review_text)` 함수 시그니처만 유지하면
  `main.py`는 수정할 필요 없음 (입력: 텍스트, 출력: sentiment_label/sentiment_score)
- `recommend.py`의 `recommend_products(pet, products, product_review_summary)` 함수도
  동일하게 시그니처만 유지하면 DeepFM 결과로 교체 가능
- 알러지 필터(`allergy_filter.py`)는 스키마 논의 때 "규칙 기반으로 가야 한다"고 정했던
  부분이라, 모델 교체 대상이 아님 — 계속 유지

## 한계 (POC 범위)
- 감성분석/추천 로직 모두 정확도 검증 대상 아님 (키워드 매칭 수준)
- 더미 데이터 8개 리뷰, 5개 상품, 3개 반려동물로만 검증
- 생애주기(`age_group`)/체구(`breed_size`) 계산도 단순화된 임시 로직