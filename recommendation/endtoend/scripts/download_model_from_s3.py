# -*- coding: utf-8 -*-
"""
DeepFM 모델 아티팩트를 S3에서 로컬로 내려받는 스크립트.

배경:
- models/deepfm/ 은 .gitignore 대상이라 Git에는 올라가지 않음.
- 지금까지는 "Docker 이미지 빌드 시 COPY로 포함" 방식을 가정했지만,
  모델 용량이 커서 이미지 빌드/배포가 무거워지는 문제가 있어
  S3 업로드 + 컨테이너 기동 시 다운로드 방식으로 전환.

사용법:
    python scripts/download_model_from_s3.py
    (환경변수로 설정)
        MODEL_S3_BUCKET=<버킷명>
        MODEL_S3_PREFIX=recommendation/deepfm/v3.0.0   (버전 디렉토리)
        MODEL_LOCAL_DIR=models/deepfm                   (생략 시 기본값)

    또는 인자로 직접 지정:
    python scripts/download_model_from_s3.py \
        --bucket <버킷명> \
        --prefix recommendation/deepfm/v3.0.0 \
        --dest models/deepfm

동작:
    1. S3 <bucket>/<prefix>/ 아래의 객체 목록을 전부 조회.
    2. 각 객체를 <dest>/<객체의 prefix 이후 상대경로>로 다운로드.
    3. 이미 같은 파일이 로컬에 있고 크기가 같으면 재다운로드 생략 (재시작 시 매번 받지 않도록).
    4. 다운로드 받은 파일 목록을 받은 순서로 출력 (서버 기동 로그에서 확인용).

최소 기대 산출물 (deepfm_model.py의 load_deepfm()이 읽는 파일들):
    - feature_encoder.json
    - model_config.json
    - deepfm_model.pt
"""

import argparse
import os
import sys


def _get_s3_client():
    try:
        import boto3
    except ImportError:
        print(
            "[download_model_from_s3] boto3가 설치되어 있지 않습니다. "
            "requirements.txt에 boto3를 추가하고 pip install을 먼저 실행하세요.",
            file=sys.stderr,
        )
        raise
    return boto3.client("s3")


def download_model(bucket: str, prefix: str, dest_dir: str) -> list:
    """S3 <bucket>/<prefix>/ 아래 객체를 전부 dest_dir로 내려받는다.

    Returns:
        실제로 (재)다운로드한 파일의 로컬 경로 목록.
    """
    s3 = _get_s3_client()

    if not prefix.endswith("/"):
        prefix = prefix + "/"

    os.makedirs(dest_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    downloaded = []
    found_any = False

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue  # "디렉토리" placeholder 객체는 건너뜀
            found_any = True

            rel_path = key[len(prefix):]
            local_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

            remote_size = obj["Size"]
            if os.path.exists(local_path) and os.path.getsize(local_path) == remote_size:
                print(f"[download_model_from_s3] skip (이미 존재, 크기 동일): {rel_path}")
                continue

            print(f"[download_model_from_s3] downloading s3://{bucket}/{key} -> {local_path}")
            s3.download_file(bucket, key, local_path)
            downloaded.append(local_path)

    if not found_any:
        raise FileNotFoundError(
            f"s3://{bucket}/{prefix} 아래에 객체가 없습니다. "
            "버킷/프리픽스(버전) 값을 확인하세요."
        )

    return downloaded


def _parse_args():
    parser = argparse.ArgumentParser(description="DeepFM 모델 아티팩트를 S3에서 로컬로 다운로드")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("MODEL_S3_BUCKET"),
        help="S3 버킷명 (기본값: 환경변수 MODEL_S3_BUCKET)",
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("MODEL_S3_PREFIX"),
        help="S3 prefix, 버전 디렉토리까지 포함 (기본값: 환경변수 MODEL_S3_PREFIX, 예: recommendation/deepfm/v3.0.0)",
    )
    parser.add_argument(
        "--dest",
        default=os.environ.get("MODEL_LOCAL_DIR", "models/deepfm"),
        help="로컬 저장 디렉토리 (기본값: 환경변수 MODEL_LOCAL_DIR 또는 models/deepfm)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if not args.bucket:
        print("[download_model_from_s3] --bucket 또는 MODEL_S3_BUCKET 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)
    if not args.prefix:
        print("[download_model_from_s3] --prefix 또는 MODEL_S3_PREFIX 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)

    downloaded = download_model(args.bucket, args.prefix, args.dest)
    print(f"[download_model_from_s3] 완료: {len(downloaded)}개 파일 다운로드 (대상: {args.dest})")


if __name__ == "__main__":
    main()