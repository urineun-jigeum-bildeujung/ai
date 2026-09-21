"""Rolling Validation의 구간별 모델 우위가 특정 사용자·상품에 몰렸는지 진단합니다."""

from __future__ import annotations

from math import isclose, isfinite

import pandas as pd

from .evaluation import COUNT_SEGMENT_BINS, COUNT_SEGMENT_LABELS
from .rolling_validation import RollingValidationError


def summarize_entity_contribution_concentration(
    paired_rows: pd.DataFrame,
    *,
    count_column: str,
    count_bucket: str,
    entity_column: str,
) -> dict[str, int | float | str | None]:
    """고정 구간에서 사용자·상품별 순기여를 먼저 합산한 뒤 집중도를 계산합니다.

    양·음의 행별 기여를 곧바로 분리하지 않습니다. 동일한 사용자의 이득과
    손실을 먼저 상쇄해야 그 사용자가 fold 우위에 실제로 기여한 몫이 됩니다.
    """
    required = {
        count_column,
        entity_column,
        "ipcw_outcome_known",
        "ipcw_event_within_horizon",
        "ipcw_weight",
        "reference_predicted_event_probability",
        "candidate_predicted_event_probability",
    }
    missing = required - set(paired_rows.columns)
    if missing:
        raise RollingValidationError(
            f"집중도 분석에 필요한 열이 없습니다: {sorted(missing)}"
        )
    if count_bucket not in COUNT_SEGMENT_LABELS:
        raise RollingValidationError(f"알 수 없는 관측 수 구간입니다: {count_bucket}")
    if paired_rows.empty:
        raise RollingValidationError("집중도를 계산할 평가 표본이 없습니다.")

    buckets = pd.cut(
        paired_rows[count_column],
        bins=COUNT_SEGMENT_BINS,
        labels=COUNT_SEGMENT_LABELS,
        include_lowest=True,
    )
    if buckets.isna().any():
        raise RollingValidationError("집중도 분석에서 구간을 알 수 없는 행이 있습니다.")
    segment = paired_rows.loc[buckets.eq(count_bucket)].copy()
    if segment.empty:
        raise RollingValidationError(
            f"{count_column} {count_bucket} 구간이 비었습니다."
        )
    if (
        segment[entity_column].isna().any()
        or segment[entity_column].astype(str).eq("").any()
    ):
        raise RollingValidationError(f"{entity_column}에 누락된 식별자가 있습니다.")

    all_known = paired_rows.loc[paired_rows["ipcw_outcome_known"]]
    fold_weight = float(all_known["ipcw_weight"].sum())
    if not isfinite(fold_weight) or fold_weight <= 0:
        raise RollingValidationError("fold 전체 IPCW 가중치 합이 유효하지 않습니다.")
    known = segment.loc[segment["ipcw_outcome_known"]].copy()
    if known.empty:
        raise RollingValidationError("집중도 분석 구간에 정답 확인 행이 없습니다.")
    actual = known["ipcw_event_within_horizon"].astype("float64")
    aft_error = known["reference_predicted_event_probability"].sub(actual).pow(2)
    lightgbm_error = known["candidate_predicted_event_probability"].sub(actual).pow(2)
    known["fold_brier_contribution"] = (
        aft_error.sub(lightgbm_error).mul(known["ipcw_weight"]).div(fold_weight)
    )
    if not known["fold_brier_contribution"].map(isfinite).all():
        raise RollingValidationError("사용자·상품별 Brier 기여량이 유한하지 않습니다.")

    # 전체 구간 행 수는 검열 표본도 포함하며, 기여량은 정답 확인 행만 사용합니다.
    entity_rows = segment.groupby(entity_column, observed=True, sort=True).size()
    entity_contributions = known.groupby(entity_column, observed=True, sort=True)[
        "fold_brier_contribution"
    ].sum()
    positive = entity_contributions.clip(lower=0)
    negative = entity_contributions.clip(upper=0)
    positive_total = float(positive.sum())
    negative_total = float(negative.sum())
    net_total = float(entity_contributions.sum())
    if not isclose(positive_total + negative_total, net_total, abs_tol=1e-12):
        raise RollingValidationError("양·음의 사용자·상품 기여량 합계가 다릅니다.")

    positive_shares = positive.loc[positive.gt(0)].div(positive_total)
    row_shares = entity_rows.div(len(segment))
    top_positive_entity = positive_shares.idxmax() if positive_total > 0 else None
    return {
        "count_column": count_column,
        "count_bucket": count_bucket,
        "entity_column": entity_column,
        "sample_count": int(len(segment)),
        "known_sample_count": int(len(known)),
        "entity_count": int(len(entity_rows)),
        "positive_entity_count": int(len(positive_shares)),
        "row_top1_share": float(row_shares.max()),
        "row_top5_share": float(row_shares.nlargest(5).sum()),
        "positive_top1_share": (
            float(positive_shares.max()) if positive_total > 0 else None
        ),
        "positive_top5_share": (
            float(positive_shares.nlargest(5).sum()) if positive_total > 0 else None
        ),
        "positive_top1_entity_row_share": (
            float(row_shares.loc[top_positive_entity])
            if top_positive_entity is not None
            else None
        ),
        "positive_hhi": (
            float(positive_shares.pow(2).sum()) if positive_total > 0 else None
        ),
        "positive_effective_entity_count": (
            float(1.0 / positive_shares.pow(2).sum()) if positive_total > 0 else None
        ),
        "positive_contribution_total": positive_total,
        "negative_contribution_total": negative_total,
        "net_contribution_total": net_total,
    }
