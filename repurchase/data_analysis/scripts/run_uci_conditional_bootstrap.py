"""AFT 조건부 확률의 시점별 Brier 개선을 사용자 Bootstrap으로 점검합니다.

고정 모델 예측과 IPCW 가중치를 유지한 채 Validation 사용자만 복원추출합니다.
모델을 다시 학습하지 않으므로 이 구간은 모델 학습의 불확실성이 아닙니다.
"""

from __future__ import annotations

import gzip
import json
from typing import Final

import pandas as pd

from .loaders import load_uci_online_retail_ii
from .paths import REPORT_DIR
from .preprocessing.events import build_uci_purchase_events
from .preprocessing.labels import build_same_product_repurchase_labels
from .preprocessing.uci import classify_uci_rows
from .reporting import write_bytes_atomically, write_text_atomically
from .run_uci_conditional_validation import (
    LANDMARK_DAYS,
    evaluate_conditional_landmarks,
)
from .run_uci_xgboost_aft import (
    XGBoostAFTPreparedExperiment,
    prepare_xgboost_aft_experiment,
)

BOOTSTRAP_REPLICATES: Final[int] = 1_000
BOOTSTRAP_RANDOM_SEED: Final[int] = 42
JSON_REPORT_PATH = REPORT_DIR / "uci_aft_conditional_bootstrap.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "uci_aft_conditional_bootstrap.md"
TRIALS_REPORT_PATH = REPORT_DIR / "uci_aft_conditional_bootstrap_trials.json.gz"


def build_conditional_bootstrap_report(
    prepared: XGBoostAFTPreparedExperiment,
    *,
    landmark_days: tuple[int, ...] = LANDMARK_DAYS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_random_seed: int = BOOTSTRAP_RANDOM_SEED,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """시점별 요약과 재현용 반복 자료를 분리해 반환합니다."""
    report = evaluate_conditional_landmarks(
        prepared,
        landmark_days=landmark_days,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
    )
    trials: list[dict[str, object]] = []
    for landmark in report["landmarks"]:
        for trial in landmark.pop("bootstrap_trials"):
            trials.append({"elapsed_days": landmark["elapsed_days"], **trial})
    report["bootstrap_method"] = {
        "unit": "user_id",
        "sampling": "with_replacement",
        "replicates": bootstrap_replicates,
        "random_seed": bootstrap_random_seed,
        "fixed_model_predictions": True,
        "fixed_ipcw_weights": True,
    }
    report["bootstrap_interpretation"] = (
        "95% 구간은 고정 모델·고정 IPCW 가중치에서 Validation 사용자를 다시 뽑았을 때 "
        "Brier 개선 차이가 얼마나 변하는지를 나타냅니다. 정답 확인 행이 있는 사용자만 "
        "재표집하며 같은 사용자의 구매 행을 함께 뽑습니다. "
        "모델 재학습, 데이터 원천 편향, 미래 운영 성능의 불확실성은 포함하지 않습니다. "
        "각 시점의 위험집단이 다르므로 서로 다른 시점의 구간을 직접 비교하지 않습니다."
    )
    return report, trials


def render_conditional_bootstrap(report: dict[str, object]) -> str:
    """사용자 재표집 결과를 점추정과 95% 구간으로 간결하게 보여줍니다."""
    method = report["bootstrap_method"]
    lines = [
        "# UCI AFT 조건부 확률 사용자 Bootstrap",
        "",
        f"- 사용자 단위 복원추출 {method['replicates']:,}회, 시드 {method['random_seed']}",
        "- 개선량 = Train 전체확률 기준선의 IPCW Brier − AFT의 IPCW Brier; 양수일수록 AFT 우세",
        "- 고정 AFT·IPCW 가중치를 재사용하며 Test는 평가하지 않았습니다.",
        "",
        "| 구매 후 | 위험집단 사용자 | 정답 확인 사용자 | 정답 확인 행 | 개선 점추정 | Bootstrap 95% 구간 | 개선 비율 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for landmark in report["landmarks"]:
        summary = landmark["user_bootstrap"]
        lines.append(
            f"| {landmark['elapsed_days']}일 | {landmark['at_risk_user_count']:,} | "
            f"{summary['user_count']:,} | "
            f"{summary['outcome_known_count']:,} | "
            f"{summary['point_brier_improvement']:+.6f} | "
            f"[{summary['bootstrap_lower_95_brier_improvement']:+.6f}, "
            f"{summary['bootstrap_upper_95_brier_improvement']:+.6f}] | "
            f"{summary['bootstrap_positive_improvement_rate']:.1%} |"
        )
    lines.extend(["", report["bootstrap_interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    """실제 UCI 데이터로 검증한 요약·반복 자료를 따로 저장합니다."""
    classified = classify_uci_rows(load_uci_online_retail_ii())
    events = build_uci_purchase_events(classified.rows)
    labels = build_same_product_repurchase_labels(
        events, observation_end_at=pd.Timestamp(events["ordered_at"].max())
    )
    prepared = prepare_xgboost_aft_experiment(labels)
    report, trials = build_conditional_bootstrap_report(prepared)
    markdown = render_conditional_bootstrap(report)
    write_text_atomically(
        JSON_REPORT_PATH,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_bytes_atomically(
        TRIALS_REPORT_PATH,
        gzip.compress(
            (json.dumps(trials, ensure_ascii=False, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
            mtime=0,
        ),
    )
    write_text_atomically(MARKDOWN_REPORT_PATH, markdown)
    print(markdown)


if __name__ == "__main__":
    main()
