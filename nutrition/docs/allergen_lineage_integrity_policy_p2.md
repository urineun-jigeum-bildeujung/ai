# 알레르기 Evidence Lineage 무결성 정책 (P2)

본 문서는 영양성분 분석 AI 전체 중 알레르기 안전성 판단 서브시스템의 검증 문서이다. NIAS 영양 기준 비교, AAFCO 판정, 보증성분 정규화 등 영양성분 전체 기능의 완료를 의미하지 않는다.

`product_allergen_refs`는 기존 SQLite 호환성을 위해 유지하며, P2는 schema-level FK 대신 application-level parent integrity 검사를 사용한다.

- 모든 catalog evidence에는 `component_occurrence_id`가 있어야 한다.
- ID는 terminal `processing_status`를 가진 `product_ingredient_components` 행으로 해소되어야 한다.
- `scripts/nutrition/allergen_repository.py:get_refs`는 parent 누락, component table 누락, 비정상 terminal status, serialization 손상을 `LINEAGE_INTEGRITY_ERROR`로 처리하고 evidence를 반환하지 않는다.
- `scripts/api_nutrition.py:_allergy_gate`는 해당 trace를 공통 안전 서비스에 전달한다. KNOWN_LIST allergy profile에서는 `SAFETY_DATA_INSUFFICIENT`로 fail-close되며 runtime parsing으로 우회하지 않는다.

이는 application-level FK 정책이며 schema-level SQLite foreign key가 아니다. 향후 명시적 destructive migration을 수행할 때만 `FOREIGN KEY(component_occurrence_id) REFERENCES product_ingredient_components(component_occurrence_id)` 도입을 검토한다.

기존 52건 reconciliation은 component 정의가 다른 historical audit이다. P1.1의 persisted component universe 2,814개와 같은 metric으로 비교하지 않는다.
