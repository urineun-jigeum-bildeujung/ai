"""UCI의 동일 Train·Validation 표본에서 LightGBM 피처 추가 효과를 비교합니다.

기존 전처리와 시간 분할을 재사용하며 Test 예측·평가는 실행하지 않습니다.
후보별 입력 열, 표본 수, 설정, 확률 오차와 보정 결과를 보고서에 보존합니다.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from importlib.metadata import version

import pandas as pd

from .loaders import load_uci_online_retail_ii
from .modeling.lightgbm_baseline import create_lightgbm_classifier
from .modeling.model_selection import evaluate_lightgbm_feature_sets
from .modeling.samples import (
    assign_temporal_splits,
    build_historical_interval_features,
    make_temporal_split,
)
from .paths import REPORT_DIR
from .preprocessing.events import build_uci_purchase_events
from .preprocessing.labels import build_same_product_repurchase_labels
from .preprocessing.uci import classify_uci_rows
from .reporting import dataframe_to_nullable_records, write_text_atomically


def run_feature_comparison(labels: pd.DataFrame) -> dict[str, object]:
    """기존 시간 분할로 세 피처 후보를 학습·평가하고 재현 설정을 기록합니다."""
    samples = build_historical_interval_features(labels)
    split = make_temporal_split(samples)
    samples = assign_temporal_splits(samples, split)
    training = samples.loc[samples["split"].eq("train")].copy()
    validation = samples.loc[samples["split"].eq("validation")].copy()
    evaluation = evaluate_lightgbm_feature_sets(training, validation)
    split_metadata = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in asdict(split).items()
    }
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "lightgbm_feature_addition_v1",
        "evaluation_split": "validation",
        "horizon_days": 30,
        "calibration_bin_count": 10,
        "split": split_metadata,
        "model_parameters": create_lightgbm_classifier().get_params(),
        "runtime": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("lightgbm", "scikit-learn", "pandas", "numpy")
            },
        },
        "comparison": dataframe_to_nullable_records(evaluation.comparison),
        "calibration": dataframe_to_nullable_records(evaluation.calibration),
        "interpretation": (
            "표본과 설정을 고정한 단계적 피처 추가 비교입니다. 각 차이는 앞선 피처가 "
            "주어진 조건에서의 효과이며, 독립적인 인과 효과나 개선의 통계적 확정을 "
            "의미하지 않습니다. 결측값도 입력 정보로 보존합니다."
        ),
    }


def render_feature_comparison(report: dict[str, object]) -> str:
    """피처 구성과 표본 보존 여부를 함께 확인할 수 있는 비교표를 만듭니다."""
    lines = [
        "# UCI LightGBM 피처 단계별 비교",
        "",
        "- 평가: Validation, 30일 내 동일 상품 재구매 확률",
        "- 동일 학습·평가 표본, IPCW 가중치, 모델 설정을 사용합니다.",
        "- Brier·ECE·MCE는 낮을수록 좋고, 이전 후보 대비 개선량은 양수일수록 좋습니다.",
        "",
        "| 후보 | 피처 | 학습 표본 | 평가 전체 / 정답 확인 | Brier | ECE | MCE | 이전 대비 Brier 개선 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["comparison"]:
        improvement = row["brier_improvement_vs_previous"]
        improvement_text = "—" if improvement is None else f"{improvement:+.6f}"
        lines.append(
            f"| {row['feature_set']} | {', '.join(row['feature_columns'])} | "
            f"{row['training_sample_count']:,} | {row['evaluation_sample_count']:,} / "
            f"{row['outcome_known_count']:,} | {row['ipcw_brier_score']:.6f} | "
            f"{row['expected_calibration_error']:.6f} | "
            f"{row['maximum_calibration_error']:.6f} | {improvement_text} |"
        )
    lines.extend(["", report["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    """실제 UCI 데이터의 피처 비교 결과를 JSON과 Markdown으로 저장합니다."""
    classified = classify_uci_rows(load_uci_online_retail_ii())
    events = build_uci_purchase_events(classified.rows)
    labels = build_same_product_repurchase_labels(
        events, observation_end_at=pd.Timestamp(events["ordered_at"].max())
    )
    report = run_feature_comparison(labels)
    markdown = render_feature_comparison(report)
    write_text_atomically(
        REPORT_DIR / "uci_lightgbm_feature_comparison.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_text_atomically(REPORT_DIR / "uci_lightgbm_feature_comparison.md", markdown)
    print(markdown)


if __name__ == "__main__":
    main()
