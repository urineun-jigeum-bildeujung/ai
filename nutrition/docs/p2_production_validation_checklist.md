# P2 실제 서비스 데이터 검증 체크리스트

## 적용 조건

현재는 로컬 dump 및 내부 evidence 검증만 수행했다. 아래는 **향후 실제 Service Product 표본을 전달받은 뒤** 사용할 절차이며 운영 검증 완료 기록이 아니다. 실제 DB 접근 및 쓰기 승인이 아니다. 계수는 승인된 로컬 export로 먼저 계산하고 실제 인프라 작업은 별도 승인을 받는다.

현재 확인된 snapshot:

```text
Nutrition internal GTIN clusters: 323
Ambiguous internal GTIN clusters: 62
Service products sampled: 0
Service SKU → Nutrition GTIN coverage: NOT MEASURABLE
```

내부 복수 source GTIN cluster 62개와 향후 서비스 SKU의 AMBIGUOUS 상품 수는 다른 지표다. 서비스 0행에 대한 운영 coverage percentage는 계산하지 않는다.

## 입력과 권한 체크

- [ ] export 버전/관측일/생성 주체 및 파일 hash 기록. 비밀값·개인정보 원문은 보고서에 넣지 않는다.
- [ ] source 데이터 조회/집계 책임과 ownership 검증 경계가 합의됐는지 확인.
- [ ] `service_source_input_contract.md`의 실제 공급 필드와 단위를 대조. 누락을 추정으로 채우지 않는다.
- [ ] source/allergen flags의 생성 주체와 provenance 확인.
- [ ] evidence/reference/dictionary 원본은 read-only 유지.

## 실측 기록 표 (미수령 값은 TBD)

아래 값은 향후 표본을 위한 빈 기록란이다. 현재 dump 0행 집계를 여기 복사해서 검증 완료로 취급하지 않는다.

| 항목 | 값 | 계수 기준 |
|---|---|---|
| products checked | TBD | 중복 PK를 검사한 서로 다른 서비스 product ID |
| sku present | TBD | SKU가 있는 상품 수; 유효 GTIN과 별도 |
| strict valid GTIN | TBD | 길이 8/12/13/14, ASCII, check digit 통과; 공백/하이픈 보정 금지 |
| exact Nutrition identity MATCHED | TBD | exact 후보 1개, 후속 종/분류 충돌 없음 |
| UNMATCHED | TBD | 유효 GTIN이나 내부 exact 후보 없음 |
| AMBIGUOUS | TBD | 복수 후보 또는 명시적 종/분류 충돌. 원인별 하위 계수 별도 |
| INVALID_SKU | TBD | SKU 없음/형식/체크디지트 오류 |
| ingredient codes checked | TBD | (product_id, code) 중복 제거 후 code 쌍 수 |
| structured exact mapped | TBD | dictionary exact 단일 매핑 |
| unresolved | TBD | 미등록/복수 매핑, 원인 구분 |
| pet allergy profiles checked | TBD | 중복 Pet ID 및 상태-목록 일관성 확인한 profile 수 |
| KNOWN_LIST | TBD | 명시 상태 + 비어 있지 않은 목록 |
| KNOWN_NONE | TBD | 명시 상태 + 빈 목록 |
| UNKNOWN | TBD | 명시 UNKNOWN, 상태 누락/미지원, 모순 |

- [ ] 최종 identity 4개 상태 합계 = products checked. adapter 후 종/분류 conflict를 반영한 최종 상태와 최초 GTIN lookup 상태를 구분한다.
- [ ] valid GTIN = MATCHED + UNMATCHED + AMBIGUOUS. UNKNOWN 데이터는 임의 MATCHED로 변경하지 않는다.
- [ ] ingredient checked = exact mapped + unresolved. 공백/부분 문자열/fuzzy/name matching 금지.
- [ ] profiles checked = KNOWN_LIST + KNOWN_NONE + UNKNOWN. 상태 일관성과 알레르겐 코드 지원 여부는 별도 검증한다.
- [ ] 표본 수가 0이면 coverage는 NOT MEASURABLE. 비율 보고 시 분자/분모·표본 범위 명시.

## Runtime 검증

- [ ] target_age_group은 마케팅 metadata로만 유지. AAFCO evidence 없음 → UNKNOWN; exact GTIN만으로 label 검증 주장 금지.
- [ ] 보증성분 value/unit/basis/source 및 identity binding을 확인. 결측값 생성 금지.
- [ ] Pet profile SoT 유지, Product override 금지.
- [ ] exact service allergen intersection → SAFETY_BLOCKED / excluded=true.
- [ ] intersection 없음 + 근거 부족 → SAFETY_DATA_INSUFFICIENT / excluded=true.
- [ ] Safety blocker 없는 Nutrition-only 부족 → INSUFFICIENT_DATA / excluded=false.
- [ ] 실제 source/auth 미구성 상태 → by-service-id 503 / SERVICE_SOURCE_NOT_CONFIGURED, local fallback 없음.
- [ ] source/auth/result store 미연결이면 Actual Service DB validation / Actual E2E를 YES로 기록하지 않는다.

## 로컬 재현 및 실행 기록

`nutrition/`에서 기존 환경을 사용한다. 새로운 DB나 데이터를 생성하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest tests -q -p no:cacheprovider
python -B scripts/nutrition/audit_service_identity.py \
  --dump ../../petflow-db-dump/data/product_db.dump \
  --pg-restore /opt/homebrew/opt/libpq/bin/pg_restore
git diff --check
```

compile은 `scripts/**/*.py`, `tests/**/*.py`의 bytes를 Python `compile(..., 'exec')`에 전달하여 파일을 생성하지 않고 확인한다. dump audit은 `pg_restore --file=-`로 로컬 파일만 읽고 DB에 연결하지 않는다.

2026-09-29 로컬 실행 결과:

- 수정 전: 203 passed, 1 warning. 신규 회귀 5건의 수정 전 실패를 확인했다.
- 수정 후: **208 passed, 1 warning**. warning은 기존 Starlette/AnyIO deprecated alias다.
- compile: scripts/tests Python **33개 PASS**, 메모리 compile이므로 pyc 생성 없음.
- `git diff --check`: PASS.
- 로컬 dump 재계수: products 0, 내부 GTIN clusters 323, 내부 ambiguous clusters 62. 실제 DB 접속 없음.
- 작업 전 SHA256 snapshot 1,105개와 비교하여 삭제 0, 범위 밖 기존 파일 변경 0을 확인했다. 원본 evidence/reference/dictionary/Rule Engine/구형 및 curated Runtime은 변경하지 않았다.
- 최종 변경 경로 15개는 모두 `nutrition/**`다. Git 쓰기, AWS 접근, 실제 DB 쓰기/DDL 없음.

이 문서의 체크박스는 미실시 운영 검증이므로 체크하지 않는다. 로컬 테스트 통과는 실제 Service DB validation 또는 운영 E2E 성공이 아니다.
