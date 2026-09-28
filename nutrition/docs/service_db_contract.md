# Nutrition Service DB Contract

## 목적과 현재 상태

이 문서는 AWS Service DB와 Nutrition AI 사이의 logical contract를 정의한다. 현재 AWS schema는 이 repository에서 확인되지 않았으므로 실제 table명, column명, SQL은 정의하거나 구현하지 않는다.

## 데이터 책임

| 데이터 영역 | AI 권한 | 목적 |
|---|---|---|
| Service source data | SELECT only | 분석 input 조회 |
| AI-owned Nutrition Result | INSERT / UPDATE | 분석 결과, version, provenance 저장 |

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

## 미확인 사항

- 실제 AWS database engine, table명, column명, primary/foreign key
- 서비스 인증과 DB credential 전달 방식
- AI-owned result table의 retention, update key, transaction 정책
- FE request의 `pet_id`와 `product_id` identifier format
