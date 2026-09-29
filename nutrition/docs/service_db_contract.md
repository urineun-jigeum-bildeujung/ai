# Nutrition Service DB Contract

## 목적과 현재 상태

2026-09-28 보충: 아래 미확인 목록은 최초 baseline 당시 기록이다. 현재 로컬 PostgreSQL dump schema 및 사용자 확정 계약으로 순수 Adapter를 준비했다. 실제 조회 연결, 인증 신뢰 경계, 결과 저장소는 미연결이다. 최신 확인/차단 구분은 [P0~P3 보고서](service_integration_preparation_p0_p3.md)를 따른다.

이 문서는 AWS Service DB와 Nutrition AI 사이의 logical contract를 정의한다. 최초 baseline에는 AWS schema가 없었으며, 그 당시 실제 table명, column명, SQL은 정의하거나 구현하지 않았다.

## 데이터 책임

아래 표는 후속 운영 설계의 책임 구분이지 현재 DB 접근·쓰기 승인이 아니다. 현재 단계에서는 로컬 dump 분석과 adapter/test만 허용한다. 실제 인프라 DB의 CREATE/ALTER/GRANT/REVOKE/INSERT/UPDATE/DELETE는 모두 금지하며 별도 승인 전에는 실행하지 않는다. 결과 저장소 schema와 upsert ownership도 아직 확정되지 않았다.

| 데이터 영역 | AI 권한 | 목적 |
|---|---|---|
| Service source data | SELECT only | 분석 input 조회 |
| AI-owned Nutrition Result | 후속 승인 필요, 현재 쓰기 금지 | 분석 결과, version, provenance 저장 목표 |

Pet, Product, Member, Order 등 서비스 source data는 AI가 수정하지 않는다.

## 필요한 logical field

### Pet source data

- pet identity
- species
- life-stage source
- weight, if available
- allergy profile와 profile status

### Product source data

- product identity
- target species
- target life-stage evidence
- guaranteed analysis 또는 nutrition evidence
- ingredient evidence
- source/provenance metadata

### Analysis metadata와 결과

- Rule Version
- Reference Version
- analyzed_at
- input/readiness, nutrition coverage, comparison, safety, final analysis status
- reason code와 provenance trace

누락된 logical field는 임의 기본값으로 대체하지 않는다. Rule Engine이 분석할 수 없는 값은 `UNKNOWN` 또는 `INSUFFICIENT_DATA` 계열로 남긴다.

## Repository 경계

실제 schema 확정 전에는 Repository interface와 SQL 구현을 추가하지 않는다. 후속 구현의 책임은 다음과 같이 분리한다.

- PetRepository: `pet_id`로 Pet source data 조회
- ProductRepository: `product_id`로 Product, nutrition, ingredient source data 조회
- NutritionResultRepository: AI-owned Nutrition Result 저장

각 Repository는 Service DB Adapter가 사용할 source record만 반환하고, Rule Engine은 Canonical Input만 받는다.

## 최초 baseline의 미확인 목록 (현재 미확인 목록이 아님)

- 실제 AWS database engine, table명, column명, primary/foreign key
- 서비스 인증과 DB credential 전달 방식
- AI-owned result table의 retention, update key, transaction 정책
- FE request의 `pet_id`와 `product_id` identifier format

현재 최소 내부 입력 DTO, 확인된 변환 규칙 및 남은 운영 연결 blocker는 [서비스 입력 계약](service_source_input_contract.md)을 따른다. dump의 schema 존재가 실제 운영 접속·권한·계약 검증을 뜻하지 않는다. 필요한 DDL은 향후 `nutrition/docs/` 내 `PROPOSED` / `NOT_APPLIED` 문서로만 제안한다. 이번 재검토에서는 새로운 table이나 DDL을 생성하지 않았다.
