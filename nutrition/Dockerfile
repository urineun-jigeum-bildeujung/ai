# requirements.lock 주석 기준 재현 환경(Python 3.11.x)에 맞춤.
FROM python:3.11.15-slim-bookworm

# 파이썬 캐시 파일을 만들지 않고 로그를 즉시 표준 출력으로 전달합니다.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# runtime allergen evidence 조회는 SQLite mode=ro를 사용한다.
# 파일이 없거나 손상된 경우 빈 DB를 생성하지 않고 fail-close한다.
# data 디렉터리는 기존 artifact 접근 호환성을 위해 non-root 사용자 소유로 유지한다.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin nutrition

COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir --requirement requirements.lock

# scripts/nutrition 안의 실제 로직 모듈, data(시드/기준 데이터, 992KB)만 복사한다.
# tests/API/docs는 런타임에 쓰이지 않아 이미지에 넣지 않는다.
# scripts는 실행 중 쓸 일이 없어 root 소유로 남겨(프로세스가 침해돼도 코드 변조를
# 못 하게) 두고, data만 nutrition 소유로 줘서 read-only artifact 접근 권한을 유지한다.
COPY --chown=root:root scripts ./scripts
COPY --chown=nutrition:nutrition data ./data

USER nutrition

EXPOSE 8002

# --host 0.0.0.0을 빠뜨리면 컨테이너 밖(Service/Ingress)에서 접속이 안 된다
# (루프백에만 붙는 기본값 때문에 Alloy Faro 수집기에서 겪었던 것과 동일한 유형의 실수).
ENTRYPOINT ["python", "-m", "uvicorn", "scripts.api_nutrition:app", "--host", "0.0.0.0", "--port", "8002"]
