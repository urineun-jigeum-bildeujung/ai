# -*- coding: utf-8 -*-
"""
로컬 models/deepfm/ 의 모델 아티팩트를 S3에 업로드하는 스크립트.

download_model_from_s3.py와 대칭 구조. 재학습 후 새 버전을 배포하려면
이 스크립트로 S3에 올리고, 인프라팀/배포 환경변수의 MODEL_S3_PREFIX를
새 버전 문자열로 바꾸면 된다.

사용법:
    python scripts/upload_model_to_s3.py \
        --bucket petflow-dev-ml-artifacts \
        --prefix recommendation/deepfm/v1.0.0 \
        --src models/deepfm

    또는 환경변수로:
        MODEL_S3_BUCKET=petflow-dev-ml-artifacts
        MODEL_S3_PREFIX=recommendation/deepfm/v1.0.0
        MODEL_LOCAL_DIR=models/deepfm
    python scripts/upload_model_to_s3.py

업로드 전 필수 파일 체크:
    - feature_encoder.json
    - model_config.json
    - deepfm_model.pt
(load_deepfm()이 읽는 파일들. 셋 중 하나라도 없으면 업로드하지 않고 바로 에러.)

주의:
    같은 prefix(버전)에 다시 업로드하면 기존 파일을 덮어쓴다.
    재배포 중 트래픽이 섞이는 걸 피하려면 버전 문자열을 새로 올려서
    (예: v1.0.0 -> v1.0.1) 별도 경로로 업로드하는 걸 권장.
"""

import argparse
import os
import sys

REQUIRED_FILES = ["feature_encoder.json", "model_config.json", "deepfm_model.pt"]


def _get_s3_client():
    try:
        import boto3
    except ImportError:
        print(
            "[upload_model_to_s3] boto3가 설치되어 있지 않습니다. "
            "requirements(-api).txt에 boto3를 추가하고 pip install을 먼저 실행하세요.",
            file=sys.stderr,
        )
        raise
    return boto3.client("s3")


def upload_model(bucket: str, prefix: str, src_dir: str) -> list:
    """src_dir 안의 파일을 전부 S3 <bucket>/<prefix>/ 아래로 업로드한다.

    Returns:
        업로드한 S3 key 목록.
    """
    if not prefix.endswith("/"):
        prefix = prefix + "/"

    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(src_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"{src_dir} 안에 필수 파일이 없습니다: {missing}. "
            "재학습/저장이 끝난 모델 디렉토리가 맞는지 확인하세요."
        )

    s3 = _get_s3_client()
    uploaded = []

    for root, _dirs, files in os.walk(src_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, src_dir).replace(os.sep, "/")
            key = prefix + rel_path

            print(f"[upload_model_to_s3] uploading {local_path} -> s3://{bucket}/{key}")
            s3.upload_file(local_path, bucket, key)
            uploaded.append(key)

    return uploaded


def _parse_args():
    parser = argparse.ArgumentParser(description="DeepFM 모델 아티팩트를 S3로 업로드")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("MODEL_S3_BUCKET"),
        help="S3 버킷명 (기본값: 환경변수 MODEL_S3_BUCKET)",
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("MODEL_S3_PREFIX"),
        help="S3 prefix, 버전 디렉토리까지 포함 (기본값: 환경변수 MODEL_S3_PREFIX, 예: recommendation/deepfm/v1.0.0)",
    )
    parser.add_argument(
        "--src",
        default=os.environ.get("MODEL_LOCAL_DIR", "models/deepfm"),
        help="업로드할 로컬 디렉토리 (기본값: 환경변수 MODEL_LOCAL_DIR 또는 models/deepfm)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if not args.bucket:
        print("[upload_model_to_s3] --bucket 또는 MODEL_S3_BUCKET 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)
    if not args.prefix:
        print("[upload_model_to_s3] --prefix 또는 MODEL_S3_PREFIX 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)

    uploaded = upload_model(args.bucket, args.prefix, args.src)
    print(f"[upload_model_to_s3] 완료: {len(uploaded)}개 파일 업로드 (s3://{args.bucket}/{args.prefix})")


if __name__ == "__main__":
    main()