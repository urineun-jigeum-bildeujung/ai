"""재구매 분석에 사용할 공개 거래 데이터셋을 검증하며 내려받습니다.

공식 배포 주소에서 UCI Online Retail II와 Complete Journey 원본 파일을
내려받아 ``data/raw``에 저장합니다. 예상 SHA-256과 실제 해시가 일치하는지
검증하고 출처·버전·라이선스를 명세에 기록해 데이터 계보를 추적합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset_registry import DATASET_FILES, DatasetFile
from .paths import RAW_DIR, REPORT_DIR

MANIFEST_PATH = REPORT_DIR / "data_source_manifest.json"
MANIFEST_SCHEMA_VERSION = "2.0"


class DatasetIntegrityError(RuntimeError):
    """내려받은 파일이 등록된 무결성 조건을 충족하지 않을 때 발생합니다."""


class ManifestError(RuntimeError):
    """다운로드 명세를 안전하게 읽거나 갱신할 수 없을 때 발생합니다."""


@dataclass(frozen=True)
class VerificationEvidence:
    """파일 검증 과정에서 실제로 확인한 해시와 검사 항목을 보관합니다."""

    actual_sha256: str
    checks: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedDatasetFile:
    """검증된 데이터 파일의 출처 명세와 로컬 검증 결과를 함께 보관합니다."""

    source: DatasetFile
    bytes: int
    evidence: VerificationEvidence
    verified_at: str

    def to_manifest_entry(self) -> dict[str, Any]:
        """검증 결과를 JSON 명세에 기록할 수 있는 딕셔너리로 변환합니다."""
        return {
            "source": asdict(self.source),
            "local_file": {
                "bytes": self.bytes,
                "actual_sha256": self.evidence.actual_sha256,
                "verification": list(self.evidence.checks),
                "verified_at": self.verified_at,
            },
        }


def utc_now_iso() -> str:
    """모든 다운로드·검증 시각을 비교 가능한 UTC ISO 8601 문자열로 반환합니다."""
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    """파일 내용을 읽어 재현성 확인에 사용할 SHA-256 해시를 반환합니다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> str:
    """실제 SHA-256을 계산하고 예상값과 다르면 명시적인 오류를 발생시킵니다."""
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise DatasetIntegrityError(
            f"{path.name}의 SHA-256이 예상값과 다릅니다. "
            f"예상={expected_sha256}, 실제={actual_sha256}"
        )
    return actual_sha256


def verify_zip_archive(path: Path, expected_member: str) -> None:
    """ZIP 구조와 내부 파일명을 검사해 손상되거나 다른 압축 파일을 차단합니다."""
    with zipfile.ZipFile(path) as zip_file:
        corrupted_member = zip_file.testzip()
        if corrupted_member is not None:
            raise DatasetIntegrityError(
                f"{path.name}에서 손상된 항목을 발견했습니다: {corrupted_member}"
            )
        if zip_file.namelist() != [expected_member]:
            raise DatasetIntegrityError(
                f"{path.name}의 내부 파일 구성이 예상과 다릅니다: {zip_file.namelist()}"
            )


def verify_dataset_file(path: Path, spec: DatasetFile) -> VerificationEvidence:
    """파일 검사를 수행하고 실제로 통과한 검증 근거를 반환합니다."""
    actual_sha256 = verify_checksum(path, spec.expected_sha256)
    checks = ["sha256"]
    if spec.archive_member is not None:
        verify_zip_archive(path, spec.archive_member)
        checks.append("zip_structure")
    return VerificationEvidence(actual_sha256, tuple(checks))


def download_file(spec: DatasetFile, force: bool) -> VerifiedDatasetFile:
    """데이터셋 파일을 안전하게 내려받고 검증 결과를 반환합니다."""
    destination = RAW_DIR / spec.filename
    if destination.exists() and not force:
        evidence = verify_dataset_file(destination, spec)
        print(f"[검증 완료] {destination.name} 파일이 이미 존재합니다.")
    else:
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(
            spec.url,
            headers={"User-Agent": "gollajugae-ai-data-analysis/1.0"},
        )
        print(f"[다운로드] {spec.dataset}: {spec.filename}")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            evidence = verify_dataset_file(temporary, spec)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    return VerifiedDatasetFile(
        source=spec,
        bytes=destination.stat().st_size,
        evidence=evidence,
        verified_at=utc_now_iso(),
    )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """기존 명세를 읽고 현재 스키마와 파일 항목 구조를 검증합니다."""
    if not path.exists():
        return []

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"다운로드 명세를 읽을 수 없습니다: {path}") from error

    if not isinstance(manifest, dict):
        raise ManifestError("다운로드 명세의 최상위 값은 객체여야 합니다.")
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            "다운로드 명세 스키마 버전이 현재 코드와 다릅니다. "
            "전체 데이터셋을 다시 검증해 명세를 갱신하세요."
        )

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ManifestError("다운로드 명세의 files 값은 목록이어야 합니다.")
    for entry in files:
        if not isinstance(entry, dict):
            raise ManifestError("다운로드 명세의 각 파일 항목은 객체여야 합니다.")
        source = entry.get("source")
        local_file = entry.get("local_file")
        if not isinstance(source, dict) or not isinstance(local_file, dict):
            raise ManifestError(
                "다운로드 명세 파일 항목에 source와 local_file 객체가 필요합니다."
            )
        if not isinstance(source.get("filename"), str):
            raise ManifestError("다운로드 명세 파일 항목에 filename이 필요합니다.")
    return files


def merge_manifest_entries(
    existing: list[dict[str, Any]],
    verified: list[VerifiedDatasetFile],
    raw_dir: Path,
) -> list[dict[str, Any]]:
    """현재 존재하는 기존 기록에 새 검증 결과를 파일명 기준으로 병합합니다."""
    entries_by_filename = {
        entry["source"]["filename"]: entry
        for entry in existing
        if (raw_dir / entry["source"]["filename"]).is_file()
    }
    for result in verified:
        entries_by_filename[result.source.filename] = result.to_manifest_entry()
    return [entries_by_filename[name] for name in sorted(entries_by_filename)]


def write_manifest(path: Path, files: list[dict[str, Any]]) -> None:
    """명세를 임시 파일에 완전히 쓴 뒤 원자적으로 기존 파일과 교체합니다."""
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "updated_at": utc_now_iso(),
        "files": files,
    }
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """명령줄에서 데이터셋 선택값과 강제 재다운로드 여부를 읽습니다."""
    dataset_names = tuple(sorted({spec.dataset for spec in DATASET_FILES}))
    parser = argparse.ArgumentParser(
        description="공개 거래 데이터셋을 내려받고 무결성과 출처를 기록합니다."
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *dataset_names),
        default="all",
        help="검증할 데이터셋 이름입니다. 기본값은 전체 데이터셋입니다.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 원본 파일이 있어도 다시 내려받습니다.",
    )
    return parser.parse_args()


def main() -> None:
    """선택된 파일을 내려받고 원본 추적용 다운로드 명세를 생성합니다."""
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    selected = [
        spec
        for spec in DATASET_FILES
        if args.dataset == "all" or spec.dataset == args.dataset
    ]
    verified = [download_file(spec, force=args.force) for spec in selected]
    existing = [] if args.dataset == "all" else load_manifest(MANIFEST_PATH)
    files = merge_manifest_entries(
        existing=existing,
        verified=verified,
        raw_dir=RAW_DIR,
    )
    write_manifest(MANIFEST_PATH, files)
    print(f"[완료] 다운로드 명세: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
