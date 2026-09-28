# 재구매 배치 실행 기준과 같은 Python 3.11 패치 버전을 사용합니다.
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

# 파이썬 캐시 파일을 만들지 않고 로그를 즉시 표준 출력으로 전달합니다.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# LightGBM과 XGBoost가 사용하는 OpenMP Runtime만 설치하고 apt 캐시는 제거합니다.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./requirements-runtime.txt
RUN python -m pip install --no-cache-dir --requirement requirements-runtime.txt

# 모델·데이터·Secret은 복사하지 않고 실행 시 읽기 전용 경로로 주입합니다.
COPY scripts ./scripts

# 운영 배치를 관리자 권한으로 실행하지 않습니다.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin repurchase
USER repurchase

ENTRYPOINT ["python", "-m", "scripts.run_repurchase_batch"]
