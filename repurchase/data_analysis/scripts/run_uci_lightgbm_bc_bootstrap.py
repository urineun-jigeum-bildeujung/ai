"""UCI Validation에서 LightGBM B/C 피처 후보의 개선 안정성을 비교합니다.

B와 C를 각각 한 번 학습·예측한 뒤 같은 사용자를 복원추출합니다. Test 표본은
사용하지 않으며, 요약 결과와 반복별 결과를 분리해 저장합니다.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from .loaders import load_uci_online_retail_ii
from .modeling.evaluation import bootstrap_ipcw_brier_pair_difference_by_user
from .modeling.model_selection import (
    LIGHTGBM_FEATURE_SETS,
    build_lightgbm_feature_pair_predictions,
)
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

BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_RANDOM_SEED = 42
HORIZON_DAYS = 30


def run_bc_bootstrap(
    labels: pd.DataFrame,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    random_seed: int = BOOTSTRAP_RANDOM_SEED,
) -> dict[str, object]:
    """같은 Validation 사용자를 재표집해 B 대비 C의 Brier 개선량을 계산합니다."""
    samples = build_historical_interval_features(labels)
    split = make_temporal_split(samples)
    samples = assign_temporal_splits(samples, split)
    training = samples.loc[samples["split"].eq("train")].copy()
    validation = samples.loc[samples["split"].eq("validation")].copy()
    paired_rows = build_lightgbm_feature_pair_predictions(
        training,
        validation,
        reference_feature_columns=LIGHTGBM_FEATURE_SETS[1][1],
        candidate_feature_columns=LIGHTGBM_FEATURE_SETS[2][1],
        horizon_days=HORIZON_DAYS,
    )
    bootstrap = bootstrap_ipcw_brier_pair_difference_by_user(
        paired_rows,
        bootstrap_replicates=bootstrap_replicates,
        random_seed=random_seed,
    )
    split_metadata = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in asdict(split).items()
    }
    lower = float(bootstrap.summary["bootstrap_lower_95_brier_improvement"])
    upper = float(bootstrap.summary["bootstrap_upper_95_brier_improvement"])
    if lower > 0:
        decision = "C의 Brier 개선이 사용자 구성 변화에서도 일관되게 양수였습니다."
    elif upper < 0:
        decision = "C의 Brier Score가 사용자 구성 변화에서 일관되게 악화됐습니다."
    else:
        decision = (
            "95% Bootstrap 구간이 0을 포함해 C의 개선을 안정적인 우위로 "
            "확정하지 않습니다."
        )
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "lightgbm_bc_user_bootstrap_v1",
        "evaluation_split": "validation",
        "horizon_days": HORIZON_DAYS,
        "reference_feature_set": LIGHTGBM_FEATURE_SETS[1][0],
        "candidate_feature_set": LIGHTGBM_FEATURE_SETS[2][0],
        "split": split_metadata,
        "summary": bootstrap.summary,
        "trials": dataframe_to_nullable_records(bootstrap.trials),
        "decision": decision,
        "scope": (
            "고정된 B/C 모델 예측과 IPCW 가중치에서 사용자를 재표집한 결과입니다. "
            "모델 재학습·검열모형 재추정·새로운 피처 선택의 불확실성은 포함하지 않습니다."
        ),
    }


def render_bc_bootstrap(report: dict[str, object]) -> str:
    """점추정과 Bootstrap 구간의 의미를 짧은 Markdown으로 표시합니다."""
    summary = report["summary"]
    return "\n".join(
        [
            "# UCI LightGBM B/C 사용자 Bootstrap",
            "",
            "- 평가: Validation, 30일 내 동일 상품 재구매 확률",
            "- 개선량: Brier(B) − Brier(C), 양수이면 C가 더 좋음",
            f"- 사용자 수: {summary['user_count']:,}",
            f"- 정답 확인 표본: {summary['outcome_known_count']:,}",
            f"- 반복 수 / seed: {summary['bootstrap_replicates']:,} / {summary['random_seed']}",
            "",
            "| B Brier | C Brier | 점추정 개선량 | Bootstrap 평균 | 95% 구간 | 양수 비율 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {summary['point_reference_brier_score']:.6f} | "
                f"{summary['point_candidate_brier_score']:.6f} | "
                f"{summary['point_brier_improvement']:+.6f} | "
                f"{summary['bootstrap_mean_brier_improvement']:+.6f} | "
                f"[{summary['bootstrap_lower_95_brier_improvement']:+.6f}, "
                f"{summary['bootstrap_upper_95_brier_improvement']:+.6f}] | "
                f"{summary['bootstrap_positive_improvement_rate']:.2%} |"
            ),
            "",
            str(report["decision"]),
            "",
            str(report["scope"]),
            "",
        ]
    )


def main() -> None:
    """실제 UCI 데이터의 B/C Bootstrap 요약과 반복별 결과를 저장합니다."""
    classified = classify_uci_rows(load_uci_online_retail_ii())
    events = build_uci_purchase_events(classified.rows)
    labels = build_same_product_repurchase_labels(
        events, observation_end_at=pd.Timestamp(events["ordered_at"].max())
    )
    report = run_bc_bootstrap(labels)
    trials = report.pop("trials")
    write_text_atomically(
        REPORT_DIR / "uci_lightgbm_bc_bootstrap.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_text_atomically(
        REPORT_DIR / "uci_lightgbm_bc_bootstrap_trials.json",
        json.dumps(trials, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    markdown = render_bc_bootstrap(report)
    write_text_atomically(REPORT_DIR / "uci_lightgbm_bc_bootstrap.md", markdown)
    print(markdown)


if __name__ == "__main__":
    main()
