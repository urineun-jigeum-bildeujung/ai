"""Generate two isolated, synthetic service-source snapshots for pipeline testing.

These fabricated outcomes are not evidence of real-user calibration or model quality.
No row or target is copied from the old Test snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.export_local_service_snapshot import verify_snapshot

VERSION = "independent-synthetic-v2"
HEADERS = {
    "orders": (
        "order_id",
        "user_id",
        "ordered_at",
        "paid_at",
        "order_status",
        "purchase_type",
    ),
    "order_items": (
        "order_item_id",
        "order_id",
        "product_id",
        "product_group_id_snapshot",
        "category_code_snapshot",
        "is_replenishable_snapshot",
        "pet_id",
        "quantity",
        "item_status",
        "cancelled_quantity",
        "returned_quantity",
    ),
    "histories": ("history_id", "order_id", "from_status", "to_status", "changed_at"),
    "claims": (
        "claim_id",
        "order_id",
        "claim_type",
        "claim_status",
        "requested_at",
        "completed_at",
    ),
    "claim_items": ("claim_item_id", "claim_id", "order_item_id", "quantity"),
    "pets": ("pet_id", "user_id", "birth_date"),
}
ROLES = ("calibration_development", "final_evaluation")


def _utc(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _create_one(
    directory: Path,
    *,
    role: str,
    start: datetime,
    seed: int,
    id_offset: int,
    user_count: int,
    duration_days: int,
) -> dict[str, object]:
    if directory.exists():
        raise ValueError(f"출력 디렉터리가 이미 존재합니다: {directory}")
    rng = random.Random(seed)
    frames: dict[str, list[tuple[object, ...]]] = {name: [] for name in HEADERS}
    order_seq = item_seq = history_seq = 0
    end = start + timedelta(days=duration_days)
    for index in range(user_count):
        user_id = id_offset + index + 1
        pet_id = id_offset + 1_000_000 + index + 1
        frames["pets"].append(
            (
                pet_id,
                user_id,
                (start - timedelta(days=500 + index % 100)).date().isoformat(),
            )
        )
        for group_index in range(1 + rng.randrange(2)):
            group_id = 10_000 + group_index * 100 + index % 12
            # Latent cadence is sampled independently for each synthetic user/group.
            cadence = rng.uniform(18, 62)
            elapsed = rng.uniform(0, 18)
            while elapsed < duration_days - 2:
                paid = start + timedelta(days=elapsed)
                ordered = paid - timedelta(minutes=5)
                confirmed = paid + timedelta(days=1)
                order_seq += 1
                item_seq += 1
                order_id = id_offset + 2_000_000 + order_seq
                item_id = id_offset + 3_000_000 + item_seq
                frames["orders"].append(
                    (
                        order_id,
                        user_id,
                        _utc(ordered),
                        _utc(paid),
                        "CONFIRMED",
                        "ONE_TIME",
                    )
                )
                frames["order_items"].append(
                    (
                        item_id,
                        order_id,
                        id_offset + 4_000_000 + group_id,
                        group_id,
                        "FOOD",
                        True,
                        pet_id,
                        1,
                        "PAID",
                        0,
                        0,
                    )
                )
                for former, latter, changed in (
                    ("", "PENDING", ordered),
                    ("PENDING", "PAID", paid),
                    ("PAID", "CONFIRMED", confirmed),
                ):
                    history_seq += 1
                    frames["histories"].append(
                        (
                            id_offset + 5_000_000 + history_seq,
                            order_id,
                            former,
                            latter,
                            _utc(changed),
                        )
                    )
                elapsed += max(5.0, cadence * math.exp(rng.gauss(0, 0.18)))
    directory.mkdir(parents=True, mode=0o700)
    files = {}
    for name, header in HEADERS.items():
        path = directory / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(frames[name])
        files[name] = {
            "filename": path.name,
            "rows": len(frames[name]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    config = {
        "version": VERSION,
        "role": role,
        "seed": seed,
        "id_offset": id_offset,
        "user_count": user_count,
        "duration_days": duration_days,
        "cadence_days": [18, 62],
        "jitter_log_sigma": 0.18,
        "claims": "none",
    }
    plan = {
        "schema_version": 1,
        "dataset_role": role,
        "dataset_run_id": f"{role}-{seed}-{start:%Y%m%d}",
        "generator_version": VERSION,
        "source_dataset_version": "fully_synthetic_no_baseline_rows",
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "random_seed": seed,
        "evaluation_start_at": _utc(start),
        "observation_end_at": _utc(end),
        "generation_params": config,
        "limitations": "Synthetic future purchase cadence and no claim cases; pipeline test only, not real-user calibration evidence.",
    }
    (directory / "manifest.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "evaluation-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    verify_snapshot(directory)
    return {
        "directory": str(directory),
        "role": role,
        "counts": {name: len(rows) for name, rows in frames.items()},
    }


def generate_pair(
    output_root: Path,
    *,
    start: datetime,
    user_count: int = 120,
    duration_days: int = 540,
    first_seed: int = 4201,
    first_id_offset: int = 9_000_000_000_000,
) -> list[dict[str, object]]:
    if (
        start.tzinfo is None
        or user_count < 1
        or duration_days < 450
        or first_seed < 0
        or first_id_offset < 0
    ):
        raise ValueError(
            "시작 시각에는 시간대가 필요하며, 사용자 수는 양수, 기간은 450일 이상, 시드와 ID 오프셋은 음수가 아니어야 합니다."
        )
    start = start.astimezone(UTC)
    paths = [output_root / role for role in ROLES]
    if any(path.exists() for path in paths):
        raise ValueError(
            "출력 디렉터리가 이미 존재합니다. 기존 결과는 덮어쓰지 않습니다."
        )
    return [
        _create_one(
            path,
            role=role,
            start=start + timedelta(days=position * (duration_days + 1)),
            seed=first_seed + position,
            id_offset=first_id_offset + position * 1_000_000_000,
            user_count=user_count,
            duration_days=duration_days,
        )
        for position, (path, role) in enumerate(zip(paths, ROLES, strict=True))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-at", required=True, help="시간대가 있는 ISO-8601 시각")
    parser.add_argument("--user-count", type=int, default=120)
    parser.add_argument("--duration-days", type=int, default=540)
    parser.add_argument("--first-seed", type=int, default=4201)
    parser.add_argument("--first-id-offset", type=int, default=9_000_000_000_000)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_pair(
                args.output_root,
                start=datetime.fromisoformat(args.start_at),
                user_count=args.user_count,
                duration_days=args.duration_days,
                first_seed=args.first_seed,
                first_id_offset=args.first_id_offset,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
