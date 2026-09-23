# requirements.lock 주석 기준 재현 환경(Python 3.11.x)에 맞춤.
FROM python:3.11.15-slim-bookworm

# 파이썬 캐시 파일을 만들지 않고 로그를 즉시 표준 출력으로 전달합니다.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# allergen_repository.py가 data/processed/allergen_evidence_catalog_p2.db를
# sqlite3.connect()로 여는데, 이 파일은 레포에 없어서 최초 연결 시 sqlite3가
# 빈 파일로 새로 만든다 — 그 디렉토리에 쓰기 권한이 있어야 하므로, 데이터를
# COPY하는 시점부터 non-root 사용자 소유로 만든다(나중에 chown하는 것보다
# 레이어 수가 줄어듦). 그래서 requirements.lock/pip install보다 먼저 사용자를
# 만든다.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin nutrition

COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir --requirement requirements.lock

# scripts/nutrition 안의 실제 로직 모듈, data(시드/기준 데이터, 992KB)만 복사한다.
# tests/API/docs는 런타임에 쓰이지 않아 이미지에 넣지 않는다.
COPY --chown=nutrition:nutrition scripts ./scripts
COPY --chown=nutrition:nutrition data ./data

USER nutrition

EXPOSE 8002

# --host 0.0.0.0을 빠뜨리면 컨테이너 밖(Service/Ingress)에서 접속이 안 된다
# (루프백에만 붙는 기본값 때문에 Alloy Faro 수집기에서 겪었던 것과 동일한 유형의 실수).
ENTRYPOINT ["python", "-m", "uvicorn", "scripts.api_nutrition:app", "--host", "0.0.0.0", "--port", "8002"]
