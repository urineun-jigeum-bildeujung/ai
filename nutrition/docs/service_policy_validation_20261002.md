# Nutrition 정책 수정 검증 (2026-10-02)

## 변경

- by-service-id에 optional allergy_profile_status (UNKNOWN / KNOWN_NONE / KNOWN_LIST)를 추가했다. FE 상태를 실제 SELECT pet_allergy 목록과 대조한다. 미전달/null 또는 상태-목록 모순은 UNKNOWN으로 fail-close한다. DB 쓰기 및 신규 컬럼은 없다.
- birth_date를 age와 기존 stage보다 우선한다. KST 현재 날짜의 달력 개월로 <12개월 GROWTH_REPRODUCTION, >=12개월 ADULT_MAINTENANCE를 계산한다. 잘못된 날짜와 미래 날짜는 거절한다. 임신/수유를 추론하지 않는다.
- birth_date 없는 production은 기존 stage 누락 fail-close를 유지한다. 내부 명시 stage 입력 및 별도 Mock Nutrition AGE_RULE 호환 경로는 기존 정책을 유지한다. Mock AGE_RULE을 Feeding 정책으로 승격하지 않는다.
- SALMON/TUNA/BONITO/ANCHOVY/MACKEREL/HERRING/SARDINE/WHITEFISH는 동명의 v3 canonical entry가 있을 때만 매핑한다. 현재 v3에는 없으므로 UNRESOLVED다. parent FISH로 확대하지 않는다.
- Toxic/Caution code는 Allergy 매핑/flag 교집합에서 제외하고 기존 product_cautions 독성 정책으로 독립 평가한다.
- 한글 ingredient exact alias 매핑과 기존 Feeding MER 미확정/null 정책을 유지한다.

## 검증 결과

- 전체 Nutrition pytest: **441 passed**, 의존성 deprecation warning 1개.
- Toxic / Allergen / Feeding 지정 회귀: **145 passed**.
- legacy projection: **286/286 동일**. 기존 projection에서 feeding과 product_allergen_refs를 제외하는 기준을 그대로 사용했다.
- 기존 archived 성공 응답 Nutrition projection: **22/22 동일**.
- Mock 상태: READY/PARTIAL/UNAVAILABLE **107/13/166**, caution 상품 **17**.
- Toxic 합성: **5/5 PASS**.
- Python scripts/tests **47파일 compile PASS**.
- git diff --check 및 신규 파일 포함 whitespace 검사 **PASS**.
- 초기 pytest 1건은 psycopg2 누락으로 실패했다. 기존 로컬 psycopg2-binary 2.9.9 경로를 사용하여 전체 재실행 후 통과했다. 의존성 설치 및 테스트 skip은 없다.

## 재현

원본: `/private/tmp/ai-nutrition-feeding/nutrition`.
검토용 복사본: `nutrition-implementation-20261002/nutrition` (변경 파일 byte 일치 확인).

```sh
PYTHONPATH=/private/tmp/ai-nutrition-test-deps ai-nutrition-curated/.venv/bin/python -B -m pytest /private/tmp/ai-nutrition-feeding/nutrition/tests -q -p no:cacheprovider
ai-nutrition-curated/.venv/bin/python -B nutrition-implementation-20261002/regression.py /private/tmp/ai-nutrition-feeding/nutrition /private/tmp/nutrition-policy-after.json
git -C /private/tmp/ai-nutrition-feeding diff --check -- nutrition
```

로컬 archived SELECT replay 및 DB double/HTTP TestClient 검증이다. 새 live FE/Gateway/DB E2E, 배포, GitHub/Notion 변경은 수행하지 않았다. 수정은 Nutrition 내부로 한정했으며 다른 파트와 Git 기록은 변경하지 않았다.
