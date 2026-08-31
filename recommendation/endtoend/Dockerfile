# 상품 추천 시스템 — end-to-end 개발 환경 (KcELECTRA + DeepFM 실제 학습 포함)
#
# 참고: Docker Desktop for Mac은 Apple Silicon MPS(GPU)에 접근 불가 -> CPU로 동작.
# KcELECTRA 파인튜닝은 CPU에서 시간이 오래 걸릴 수 있음 -> GPU 할당 이후
# 같은 이미지를 GPU 서버(리눅스)에서 재사용하는 것을 전제로 한다.
# GPU 서버 배포 시에는 베이스 이미지를 nvidia/cuda 계열로 교체하고
# requirements.txt의 torch를 해당 CUDA 버전에 맞게 재설치한다.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "check_env.py"]