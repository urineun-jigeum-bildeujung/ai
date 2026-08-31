# -*- coding: utf-8 -*-
"""
환경 확인 스크립트.
도커 컨테이너 안에서 실행: docker run --rm endtoend
또는 로컬에서: python3 check_env.py
"""

import sys

print(f"Python 버전: {sys.version}")
print("-" * 60)

# --- PyTorch ---
try:
    import torch
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    print(f"MPS(Apple GPU) 사용 가능 여부: {mps_available} (도커 컨테이너 안에서는 항상 False가 정상)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> 사용할 device: {device}")

    x = torch.rand(3, 3, device=device)
    y = torch.rand(3, 3, device=device)
    z = x @ y
    print(f"텐서 연산 테스트 성공: {z.shape}")
except ImportError:
    print("PyTorch 미설치. requirements.txt 설치를 확인하세요.")

print("-" * 60)

# --- HuggingFace 스택 (KcELECTRA 파인튜닝용) ---
try:
    import transformers
    import datasets
    import accelerate
    import tokenizers
    print(f"transformers: {transformers.__version__}")
    print(f"datasets: {datasets.__version__}")
    print(f"accelerate: {accelerate.__version__}")
    print(f"tokenizers: {tokenizers.__version__}")
except ImportError as e:
    print(f"HuggingFace 스택 누락: {e}")

print("-" * 60)

# --- 데이터 처리 ---
try:
    import pandas as pd
    import numpy as np
    import sklearn
    print(f"pandas: {pd.__version__}, numpy: {np.__version__}, scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"필수 패키지 누락: {e}")

print("-" * 60)

# --- MLflow ---
try:
    import mlflow
    print(f"mlflow: {mlflow.__version__}")
except ImportError:
    print("mlflow 미설치.")

print("-" * 60)
print("환경 확인 완료.")