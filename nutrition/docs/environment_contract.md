# Nutrition Environment Contract

## 원칙

Nutrition AI는 필요한 DB 연결과 rule/reference version만 설정한다. Backend `.env` 전체를 복사하지 않으며 OAuth, Toss, Backend 전용 credential은 Nutrition AI contract에 포함하지 않는다.

현재 baseline runtime은 local selected artifact를 사용하며 AWS DB 연결을 구현하지 않았다. 아래 key는 future AWS Service DB integration에서 필요한 logical configuration 예시일 뿐, 현재 실행에 필수값이 아니다.

```dotenv
NUTRITION_DB_HOST=
NUTRITION_DB_PORT=
NUTRITION_DB_NAME=
NUTRITION_DB_USER=
NUTRITION_DB_PASSWORD=

NUTRITION_RULE_VERSION=
NUTRITION_REFERENCE_VERSION=
```

실제 인프라 naming convention이 제공되면 그 convention을 우선한다. 이 문서와 repository에는 실제 secret, AWS credential, connection string을 저장하지 않는다.

## 운영 구분

| 환경 | 현재 baseline | AWS integration 이후 |
|---|---|---|
| 데이터 source | 선택된 local artifact | Service DB read-only source data |
| 결과 저장 | 없음 | AI-owned result store만 write |
| CI | 외부 DB 연결 없음 | unit/regression은 외부 DB 없이 유지 |
| integration test | 없음 | 별도 credential과 별도 단계로 운영 |
