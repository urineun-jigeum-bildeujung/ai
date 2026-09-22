"""
P1-C 단일 source of truth 파이프라인 (P1-A + P1-A.1 + P1-B-1~4 + 18종 확장).

8단계 함수:
  1) load_raw()             - 시드 13/9/9b raw + NIAS v5 + OPFF cache 로드
  2) normalize()            - P1-A: value=0 → null, schema 통일
  3) align_units()          - P1-A.1: fraction 0~1 → % 변환
  4) detect_outliers()      - P1-B-1: 100% 초과 flag
  5) classify_dry_wet()     - P1-B-2: MOISTURE 기준 product form 분류
  6) normalize_basis()      - P1-B-3: DRY_MATTER ↔ AS_FED 변환 (CRUDE_FIBER)
  7) compare_nias()         - P1-B-4: min/max 비교, max 보강
  8) compute_aafco_pass()   - T/F/U 판정 (per-product)

산출물:
  - data/processed/seed_13_p1c_expanded_v1.json
  - data/processed/seed_14_p1c_source_of_truth_v1.json (18종 매트릭스)
  - data/processed/seed_13_p1c_source_of_truth_v1.json
  - data/processed/seed_13_p1c_summary_v1.json (10가지 판정 기준)
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from nutrition.reference_parity import canonical_reference_form, select_reference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 18종 영양소 정의 (사용자 8/31 18:48 결정)
# ============================================================================
TARGET_NUTRIENTS_18 = [
    # 기존 4종
    "CRUDE_PROTEIN", "CRUDE_FAT", "CRUDE_FIBER", "MOISTURE",
    # 추가 8종 무기질
    "CALCIUM", "PHOSPHORUS", "SODIUM", "MAGNESIUM", "POTASSIUM",
    "IRON", "COPPER", "ZINC",
    # 추가 6종 비타민/기타
    "VITAMIN_A", "VITAMIN_D", "VITAMIN_E", "VITAMIN_B1", "VITAMIN_B2",
    "TAURINE",
]
ORIGINAL_4 = ["CRUDE_PROTEIN", "CRUDE_FAT", "CRUDE_FIBER", "MOISTURE"]

# OPFF v3 API 응답 키 (실측): `crude-protein_value` / `moisture_value` 등
# → NIAS nutrient_code 매핑
OPFF_NIAS_MAP: dict[str, str] = {
    "crude-protein": "CRUDE_PROTEIN",
    "crude-fat": "CRUDE_FAT",
    "crude-fibre": "CRUDE_FIBER",
    "moisture": "MOISTURE",
    "calcium": "CALCIUM",
    "phosphorus": "PHOSPHORUS",
    "sodium": "SODIUM",
    "magnesium": "MAGNESIUM",
    "potassium": "POTASSIUM",
    "iron": "IRON",
    "copper": "COPPER",
    "zinc": "ZINC",
    "vitamin-a": "VITAMIN_A",
    "vitamin-d": "VITAMIN_D",
    "vitamin-e": "VITAMIN_E",
    "vitamin-b1": "VITAMIN_B1",
    "vitamin-b2": "VITAMIN_B2",
    "taurine": "TAURINE",
}

# 단위 변환: NIAS는 g/100g DM, OPFF는 다양
# - mg/100g → g/100g: /1000
# - µg/100g → g/100g: /1e6
# - IU/100g → g/100g: 비타민 종류별 계수 (보수적으로 NULL 처리)
UNIT_FACTOR: dict[str, float] = {
    "g": 1.0,
    "mg": 1e-3,
    "µg": 1e-6,
    "ug": 1e-6,
    "IU": None,  # 비타민 IU는 종류별 변환 계수 다름 → 변환 보류 (null 처리)
}

# ⑤-D 변경: 8 무기질 OPFF 키 set (5종 g-based + 3종 mg-based)
# OPFF `_value` = per-1kg, `_100g` = per-100g (실측 1:10 비율 일치, ⑤-C dry-run)
MINERAL_GRAM = {"calcium", "phosphorus", "sodium", "magnesium", "potassium"}
MINERAL_MILLIGRAM = {"iron", "copper", "zinc"}
MINERAL_ALL = MINERAL_GRAM | MINERAL_MILLIGRAM

# ⑤-D 변경: 8 무기질 AF→DM basis 변환 대상 (NIAS basis=DRY_MATTER 일치)
MINERAL_BASIS_NORMALIZE = {
    "CALCIUM", "PHOSPHORUS", "SODIUM", "MAGNESIUM", "POTASSIUM",
    "IRON", "COPPER", "ZINC",
}


# ============================================================================
# 1) load_raw
# ============================================================================
def load_raw(
    s13_path: Path = RAW / "seed_13_guaranteed_analysis.json",
    s9_path: Path = RAW / "seed_9_placeholder_feed_opff.json",
    s9b_path: Path = RAW / "seed_9b_off_korean_oem.json",
    nias_path: Path = PROC / "nutrition_reference_nias_2024_parity_p0_v1.json",
    opff_cache_path: Path = PROC / "opff_api_cache_v1.json",
) -> dict:
    """시드 raw + Parity P0 NIAS canonical reference + OPFF cache 로드."""
    s13 = json.loads(Path(s13_path).read_text()) if Path(s13_path).exists() else {"items": []}
    s9 = json.loads(Path(s9_path).read_text()) if Path(s9_path).exists() else {"items": []}
    s9b = json.loads(Path(s9b_path).read_text()) if Path(s9b_path).exists() else {"products": []}
    nias = json.loads(Path(nias_path).read_text()) if Path(nias_path).exists() else {"rows": []}
    opff = json.loads(Path(opff_cache_path).read_text()) if Path(opff_cache_path).exists() else {}

    return {
        "s13_items": s13.get("items", []),
        "s9_items": s9.get("items", []),
        "s9b_products": s9b.get("products", []),
        "nias_rows": nias.get("rows", []),
        "opff_cache": opff,
    }


# ============================================================================
# 2) normalize (P1-A: value=0 → null, schema 통일)
# ============================================================================
def normalize(s13_items: list) -> list:
    """P1-A 정책: value=0 은 placeholder, null 로 변환. schema 는 표준화."""
    out = []
    for r in s13_items:
        val = r.get("value")
        if val == 0 or val == 0.0:
            val = None
        # 단위/basis 보정: basis 누락 시 AS_FED 가정
        basis = r.get("basis") or "AS_FED"
        out.append({
            "product_id": r.get("product_id"),
            "nutrient_code": r.get("nutrient_code"),
            "value": val,
            "unit": r.get("unit", "PERCENT"),
            "basis": basis,
            "source": r.get("source", "OPFF"),
            "_normalization_status": "NORMALIZED" if val is not None else "NULL_PLACEHOLDER",
        })
    return out


# ============================================================================
# 3) align_units (P1-A.1: fraction 0~1 → % 변환)
# ============================================================================
def align_units(items: list) -> list:
    """P1-A.1 정책: value < 1 이고 nutrient 가 기존 4종(%)이면 × 100."""
    out = []
    for r in items:
        v = r.get("value")
        code = r.get("nutrient_code")
        if v is None:
            out.append({**r, "aligned_value": None, "alignment_status": "NULL_PLACEHOLDER"})
            continue
        if code in ORIGINAL_4 and 0 < v < 1.0:
            out.append({**r, "aligned_value": v * 100.0, "alignment_status": "ALIGNED_FRACTION_TO_PCT"})
        elif code in ORIGINAL_4 and v >= 1.0:
            out.append({**r, "aligned_value": float(v), "alignment_status": "ALREADY_PCT"})
        else:
            # 14종 추가: NIAS 와 단위 다를 수 있으므로 aligned_value 는 그대로
            # 단위 변환은 basis 단계에서 처리 (단위 표시가 같은 경우만)
            out.append({**r, "aligned_value": float(v), "alignment_status": "PASS_THROUGH"})
    return out


# ============================================================================
# 4) detect_outliers (P1-B-1: 100% 초과, 의심)
# ============================================================================
def detect_outliers(items: list) -> list:
    """P1-B-1 정책: 기존 4종에서 aligned_value > 100% 이면 OUTLIER_FLAGGED."""
    out = []
    for r in items:
        v = r.get("aligned_value")
        code = r.get("nutrient_code")
        if v is None or code not in ORIGINAL_4:
            out.append({**r, "outlier_status": "NOT_TARGET"})
            continue
        if v > 100.0:
            out.append({**r, "outlier_status": "OUTLIER_FLAGGED"})
        else:
            out.append({**r, "outlier_status": "OK"})
    return out


# ============================================================================
# 5) classify_dry_wet (P1-B-2: MOISTURE 기준 product form 분류)
# ============================================================================
def classify_dry_wet(items: list) -> list:
    """Classify only valid as-fed moisture percentages.

    Values outside ``0 <= moisture < 100`` are invalid, not dry/wet evidence.
    """
    # product_id 별 MOISTURE 값 추출
    moisture_by_product: dict[str, float] = {}
    for r in items:
        if r.get("nutrient_code") == "MOISTURE" and r.get("aligned_value") is not None:
            value = r["aligned_value"]
            if 0 <= value < 100:
                moisture_by_product[r["product_id"]] = value

    out = []
    for r in items:
        pid = r.get("product_id")
        m = moisture_by_product.get(pid)
        if m is None:
            form = "UNKNOWN"
        elif m > 60.0:
            form = "WET"
        elif m < 15.0:
            form = "DRY"
        else:
            form = "MID"
        out.append({**r, "product_form": form})
    return out


# ============================================================================
# 6) normalize_basis (P1-B-3: DRY_MATTER ↔ AS_FED)
# ============================================================================
def normalize_basis(items: list) -> list:
    """P1-B-3 정책: CRUDE_FIBER (DM→AF) + 8 무기질 (AF→DM) 변환.
    변환식:
      - CRUDE_FIBER DM→AF: value_af = value_dm × (1 - moisture/100)
      - 8 무기질 AF→DM: value_dm = value_af / (1 - moisture/100)
    moisture 부재 시 UNKNOWN (basis=UNKNOWN 정책)."""
    moisture_by_product: dict[str, float] = {}
    invalid_moisture_products: set[str] = set()
    for r in items:
        if r.get("nutrient_code") == "MOISTURE" and r.get("aligned_value") is not None:
            value = r["aligned_value"]
            if 0 <= value < 100:
                moisture_by_product[r["product_id"]] = value
            else:
                invalid_moisture_products.add(r["product_id"])

    out = []
    for r in items:
        v = r.get("aligned_value")
        code = r.get("nutrient_code")
        basis = r.get("basis", "AS_FED")
        pid = r.get("product_id")
        if pid in invalid_moisture_products:
            out.append({
                **r,
                "basis_normalized_value": None,
                "basis_normalization_status": "INVALID_MOISTURE",
                "basis_invalid": True,
            })
            continue
        # CRUDE_FIBER DM→AF (기존 로직) — v is None 인 경우 SKIPPED_NOT_TARGET (원본 동작 보존)
        if v is None and code == "CRUDE_FIBER" and basis == "DRY_MATTER":
            out.append({
                **r,
                "basis_normalized_value": v,
                "basis_normalization_status": "SKIPPED_NOT_TARGET",
            })
            continue
        if code == "CRUDE_FIBER" and basis == "DRY_MATTER":
            m = moisture_by_product.get(pid)
            if m is None:
                out.append({
                    **r,
                    "basis_normalized_value": None,
                    "basis_normalization_status": "SKIPPED_NO_MOISTURE",
                })
            else:
                new_v = v * (1 - m / 100.0)
                out.append({
                    **r,
                    "basis_normalized_value": new_v,
                    "basis_normalization_status": "CONVERTED",
                    "basis_normalized_basis": "AS_FED",
                })
            continue
        # ⑤-D 변경: 8 무기질 AF→DM (신규 분기)
        if code in MINERAL_BASIS_NORMALIZE and basis == "AS_FED":
            m = moisture_by_product.get(pid)
            if m is None:
                # ⑤-D: SKIPPED_NO_MOISTURE — value=AF 유지, basis=UNKNOWN 정책
                out.append({
                    **r,
                    "basis_normalized_value": v,
                    "basis_normalization_status": "SKIPPED_NO_MOISTURE",
                })
            else:
                new_v = v / (1 - m / 100.0)  # AF→DM
                out.append({
                    **r,
                    "basis_normalized_value": new_v,
                    "basis_normalization_status": "CONVERTED",
                    "basis_normalized_basis": "DRY_MATTER",
                })
            continue
        # PR-Vit (9/3 15:00): VITAMIN A/D/E AF→DM (NIAS basis=DRY_MATTER 정합)
        # opff_to_s13_rows 에서 IU 변환 완료 (unit=IU/100g, basis=AS_FED) 된 VITAMIN row
        # s13/s9b MOISTURE 보유 product (WET, 9003579311301/9003579311660) 만 AF→DM 변환
        # s13 MOISTURE None 인 product (form=UNKNOWN, 4047777125013) 는 AF 유지 (dry-run §3.3 정합)
        if code in ("VITAMIN_A", "VITAMIN_D", "VITAMIN_E") and basis == "AS_FED":
            m = moisture_by_product.get(pid)
            if m is None:
                # PR-Vit: s13/s9b MOISTURE 없음 → AF 유지 (form=UNKNOWN, dry-run 정합)
                out.append({
                    **r,
                    "basis_normalized_value": v,
                    "basis_normalization_status": "SKIPPED_NOT_WET",
                })
            else:
                new_v = v / (1 - m / 100.0)  # AF→DM (IU/100g AF → IU/100g DM)
                out.append({
                    **r,
                    "basis_normalized_value": new_v,
                    "basis_normalization_status": "CONVERTED",
                    "basis_normalized_basis": "DRY_MATTER",
                })
            continue
        # P0 basis contract: every non-moisture AS_FED nutrient requires known
        # moisture before it can be compared to a DRY_MATTER reference.
        if basis == "AS_FED" and code != "MOISTURE":
            m = moisture_by_product.get(pid)
            if m is None:
                out.append({**r, "basis_normalized_value": None,
                            "basis_normalization_status": "SKIPPED_NO_MOISTURE", "basis_invalid": True})
            else:
                out.append({**r, "basis_normalized_value": v / (1 - m / 100.0),
                            "basis_normalization_status": "CONVERTED", "basis_normalized_basis": "DRY_MATTER"})
            continue
        # Moisture itself remains AS_FED; DRY_MATTER values are directly comparable.
        out.append({
            **r,
            "basis_normalized_value": v,
            "basis_normalization_status": "SKIPPED_NOT_TARGET",
            "basis_normalized_basis": basis,
        })
    return out


# ⑥-F: 5-tuple JOIN key helper (NIAS table_id marker ↔ product_form canonical)
def _product_form_filter(nias_row: dict, product_form: str) -> bool:
    """⑥-F: NIAS row 의 table_id marker 와 product_form canonical enum 매핑.

    - MOISTURE-DRY ↔ DRY (또는 MID/UNKNOWN fallback)
    - MOISTURE-WET ↔ WET (또는 MID/UNKNOWN fallback)
    - OFF_BRANDED ↔ 모든 (fallback)
    - 다른 table_id (2-17a~h) ↔ 모든 (product_form 무관)

    6-1 JOIN key priority: product_form > basis > life_stage > species
    5-tuple JOIN = (species, nutrient_code, life_stage, product_form, basis)
    """
    table_id = nias_row.get("table_id", "")
    if table_id == "MOISTURE-DRY":
        return product_form in ("DRY", "MID", "UNKNOWN")
    if table_id == "MOISTURE-WET":
        return product_form in ("WET", "MID", "UNKNOWN")
    return True  # OFF_BRANDED + 다른 table_id (CP/FAT/8 mineral/5 vitamin/TAURINE) → 무관


# ============================================================================
# 7) compare_nias (P1-B-4: NIAS min/max 비교, max 보강)
# ============================================================================
# ============================================================================
# 7.5) canonical_status 8단계 결정 규칙 (PR-H 9/4 03:00, dry-run 100% 정합)
# ============================================================================
MINERAL_SET = {
    "CALCIUM", "PHOSPHORUS", "MAGNESIUM", "IRON", "COPPER", "ZINC",
    "MANGANESE", "SELENIUM", "IODINE",
}


def _canonical_status_8step(r: dict, product_species_map: dict[str, str] | None = None) -> str:
    """canonical_status 4-state 결정 (⑧ 8단계 결정 규칙, mechanical decision).

    V3 SQL-equivalent decision order.  Keep non-comparable values out of KNOWN
    even when an earlier comparison happened to produce a range result.
    """
    tid = (r.get("nias_table_id") or "").strip()
    ncode = r.get("nutrient_code", "")
    ncs = r.get("nias_compare_status", "")
    bns = r.get("basis_normalization_status", "")
    pid = r.get("product_id", "")
    species = (r.get("product_species") or (product_species_map or {}).get(pid, "DOG")).upper()

    if tid == "OFF_BRANDED":
        return "NOT_APPLICABLE"
    if ncode == "TAURINE" and species == "DOG":
        return "NOT_APPLICABLE"
    if r.get("basis_invalid") or bns == "INVALID_MOISTURE":
        return "INVALID"
    if ncs in ("NO_VALUE", "NO_REF"):
        return "UNKNOWN"
    if bns == "SKIPPED_NO_MOISTURE" and ncode in {
        "CALCIUM", "PHOSPHORUS", "SODIUM", "MAGNESIUM", "POTASSIUM", "IRON", "COPPER", "ZINC",
    }:
        return "INVALID"
    if ncode in ("VITAMIN_A", "VITAMIN_D", "VITAMIN_E") and r.get("unit") == "g/100g":
        return "INVALID"
    if ncode == "CRUDE_FIBER" and bns == "SKIPPED_NO_MOISTURE":
        return "INVALID"
    if ncs in ("IN_RANGE", "OUT_OF_RANGE"):
        return "KNOWN"
    return "UNKNOWN"


def compare_nias(
    items: list,
    nias_rows: list,
    product_species_map: dict[str, str] | None = None,
    product_life_stage_map: dict[str, str] | None = None,
    product_life_stage_detail_map: dict[str, str] | None = None,
    product_explicit_form_map: dict[str, str] | None = None,
) -> list:
    """Compare through the Parity P0 applicability-aware reference selector.

    ``NO_REF`` means reference applicability could not be resolved; it is never
    converted into a nutrient deficiency.  The optional maps are request
    metadata, not inferred defaults: an unknown form/detail therefore fails
    closed only when a selected nutrient has form/detail-specific reference.
    """
    nias_18 = [r for r in nias_rows if r.get("nutrient_code") in TARGET_NUTRIENTS_18]

    # max 보강: NIAS v5 의 max=null 인 row 에 보수적 max 적용
    # 보강 규칙: NIAS min × 2.5 (P1-B-4 정책 2순위, AAFCO safe upper bound 1순위가 더 정확하지만 v5에 없음)
    SAFE_UPPER = {
        "CRUDE_PROTEIN": 50.0,
        "CRUDE_FAT": 30.0,
        "CRUDE_FIBER": 15.0,
        "MOISTURE": 85.0,
        "CALCIUM": 10.0,
        "PHOSPHORUS": 8.0,
        "SODIUM": 4.0,
        "MAGNESIUM": 3.0,
        "POTASSIUM": 8.0,
        "IRON": 500.0,  # PR-C: 0.5 g/100g DM → 500 mg/100g DM (× 1000, NIAS min 단위 정합, 9/3)
        "COPPER": 100.0,  # PR-C: 0.1 g/100g DM → 100 mg/100g DM (× 1000, NIAS min 단위 정합, 9/3)
        "ZINC": 1000.0,  # PR-C: 1.0 g/100g DM → 1000 mg/100g DM (× 1000, NIAS min 단위 정합, 9/3)
        "VITAMIN_A": 333300.0,  # PR-Vit: 0.1 g/100g DM × 3,333,000 IU/g = 333,300 IU/100g DM (단위 변환, 9/3)
        "VITAMIN_D": 200000.0,  # PR-Vit: 0.005 g/100g DM × 40,000,000 IU/g = 200,000 IU/100g DM (단위 변환, 9/3)
        "VITAMIN_E": 1493.0,  # PR-Vit: 1.0 g/100g DM × 1,493 IU/g = 1,493 IU/100g DM (단위 변환, 9/3)
        "VITAMIN_B1": 0.05,
        "VITAMIN_B2": 0.05,
        "TAURINE": 1.0,
    }
    # AAFCO/FEDIAF 2024 안전한 상한 (DOG/CAT 평균, 보수적 추정)
    # 단위: 5 g-based mineral (Ca/P/Na/Mg/K) g/100g DM,
    #       3 mg-based (Fe/Cu/Zn) mg/100g DM (PR-C × 1000 정합, 9/3),
    #       3 IU-based vitamin (VitA/D/E) IU/100g DM (PR-Vit IU conversion 정합, 9/3)
    supplemented = []

    out = []
    for r in items:
        if r.get("basis_invalid"):
            out.append({**r, "nias_compare_status": "NO_VALUE"})
            continue
        v = r.get("basis_normalized_value")
        if v is None:
            v = r.get("aligned_value")
        code = r.get("nutrient_code")
        if v is None or not code:
            out.append({**r, "nias_compare_status": "NO_VALUE"})
            continue

        # species 결정
        pid = r.get("product_id")
        sp = (product_species_map or {}).get(pid, "DOG")
        if sp not in ("DOG", "CAT"):
            sp = "DOG"

        # No implicit adult fallback when the caller supplies a life-stage map.
        requested_stage = (product_life_stage_map or {}).get(pid, "ADULT_MAINTENANCE")
        actual_basis = r.get("basis_normalized_basis") or r.get("basis")
        reference_form, form_source = canonical_reference_form(
            (product_explicit_form_map or {}).get(pid), r.get("product_form"),
        )
        selected = select_reference(
            nias_18, species=sp, life_stage=requested_stage, nutrient_code=code,
            basis=actual_basis, reference_form=reference_form,
            life_stage_detail=(product_life_stage_detail_map or {}).get(pid),
        )
        if selected["status"] != "SELECTED":
            out.append({**r, "nias_compare_status": "NO_REF", "reference_selection_status": "NO_REF",
                        "reference_reason_code": selected["reason_code"], "reference_form": reference_form,
                        "reference_form_source": form_source})
            continue

        min_v = selected["min_value"]
        max_v = selected["max_value"]
        if max_v is None:
            max_v = SAFE_UPPER.get(code)
            if max_v is not None:
                supplemented.append({"species": sp, "life_stage": requested_stage, "nutrient_code": code,
                                    "basis": actual_basis, "max_after": max_v})
        in_range = True
        if min_v is not None and v < float(min_v):
            in_range = False
        if max_v is not None and v > float(max_v):
            in_range = False
        ncs_val = "IN_RANGE" if in_range else "OUT_OF_RANGE"
        out.append({
            **r,
            "nias_min": min_v,
            "nias_max": max_v,
            "nias_table_id": selected["rows"][0].get("table_id"),
            "nias_compare_status": ncs_val,
            "reference_selection_status": "SELECTED",
            "reference_reason_code": None,
            "reference_form": reference_form,
            "reference_form_source": form_source,
            "reference_life_stage_detail": selected.get("life_stage_detail"),
            "reference_provenance": [
                {key: ref.get(key) for key in ("source_name_original", "reference_form_raw", "reference_form_canonical", "source_table", "source_page", "authority", "edition", "source_version", "threshold_type")}
                for ref in selected["rows"]
            ],
            "safe_upper_value": max_v,  # PR-C: SAFE_UPPER 보강 값 trace (NIAS max=null 보강 후)
            "safe_upper_unit": "mg/100g DM" if code in ("IRON", "COPPER", "ZINC") else ("IU/100g DM" if code in ("VITAMIN_A", "VITAMIN_D", "VITAMIN_E") else "g/100g DM"),  # PR-C + PR-Vit 단위 정합
            "safe_upper_basis": "DRY_MATTER",  # PR-C: NIAS basis 정합 (NIAS v5 의 DRY_MATTER row 매칭)
        })

    # PR-H (9/4 03:10): canonical_status field 일괄 적용 (out 의 모든 row 대상, 8단계 결정 규칙)
    out_with_canonical = [
        {**r, "canonical_status": _canonical_status_8step(r, product_species_map)}
        for r in out
    ]
    return out_with_canonical, nias_18, supplemented


# ============================================================================
# 8) compute_nutrition_comparison_status (per-product, 필수 영양소 전부 comparable 시에만 TRUE/FALSE, 그 외 UNKNOWN)
# ============================================================================
# P0-D 9/4 11:00: MIN_1_PASS 비활성화. 필수 영양소 (essential_nutrients) 전부 comparable + min/max check 모두 PASS → TRUE.
#                  하나라도 FAIL → FALSE. 필수 영양소 중 하나라도 MISSING/NON_COMPARABLE → UNKNOWN.
# P0-D: field name 중립화 (aafco_pass → nutrition_comparison_status). AAFCO 적합 판정 X, 영양 비교 상태만 표시.
# PUPPY/KITTEN 의 ADULT_MAINTENANCE fallback 명시 (PUPPY 필수 영양소 = ADULT 와 같지만 더 엄격한 min, dry-run 시점 ADULT 매트릭스 사용, PUPPY 별도 매트릭스 = 후속 PR).
ESSENTIAL_NUTRIENTS = {
    ("DOG", "ADULT_MAINTENANCE"): ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"],
    ("DOG", "GROWTH_REPRODUCTION"): ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"],
    ("CAT", "ADULT_MAINTENANCE"): ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE"],
    ("CAT", "GROWTH_REPRODUCTION"): ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE"],
    ("BOTH", "ADULT_MAINTENANCE"): ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"],
}


def compute_nutrition_comparison_status(
    items: list,
    product_species_map: dict[str, str] | None = None,
    product_life_stage_map: dict[str, str] | None = None,
) -> dict:
    """per-product nutrition_comparison_status (P0-D 9/4 11:00, MIN_1_PASS 비활성화).
    - 필수 영양소 (essential_nutrients) 전부 comparable + min/max check 모두 PASS → TRUE
    - 하나라도 FAIL → FALSE
    - 필수 영양소 중 하나라도 MISSING/NON_COMPARABLE (NO_VALUE/NO_REF) → UNKNOWN
    - PUPPY/KITTEN life_stage fallback: ADULT_MAINTENANCE 매트릭스 사용 (별도 PUPPY 매트릭스 = 후속 PR 정교화)
    """
    by_product = defaultdict(list)
    for r in items:
        pid = r.get("product_id")
        if not pid:
            continue
        by_product[pid].append(r)

    out = {}
    for pid, rows in by_product.items():
        # species 매핑 (default: DOG)
        sp = (product_species_map or {}).get(pid, "DOG").upper()
        stage = (product_life_stage_map or {}).get(pid, "ADULT_MAINTENANCE")
        key = (sp, stage)
        # A caller that explicitly supplies an unsupported species/stage must
        # not be evaluated against an unrelated DOG/adult requirement set.
        # Keeping the legacy defaults above only covers callers that supplied
        # no maps at all; an unsupported explicit key fail-closes below.
        essential = ESSENTIAL_NUTRIENTS.get(key)
        reference_error = None
        if essential is None:
            essential = []
            reference_error = "UNSUPPORTED_REFERENCE_COMBINATION"

        # 필수 영양소별 status 매핑
        essential_status = {}
        for nutrient in essential:
            rows_n = [r for r in rows if r.get("nutrient_code") == nutrient]
            if not rows_n:
                essential_status[nutrient] = "MISSING"
                continue
            statuses = {r.get("nias_compare_status", "") for r in rows_n}
            if len(statuses & {"IN_RANGE", "OUT_OF_RANGE"}) > 1 or len({r.get("basis_normalized_value", r.get("aligned_value")) for r in rows_n}) > 1:
                essential_status[nutrient] = "DATA_CONFLICT"
                continue
            ncs = next(iter(statuses))
            if ncs in ("IN_RANGE", "OUT_OF_RANGE"):
                essential_status[nutrient] = ncs
            else:
                essential_status[nutrient] = "NON_COMPARABLE"  # NO_VALUE/NO_REF/UNKNOWN

        # 판정 (pass_count/fail_count 초기값 0)
        pass_count = 0
        fail_count = 0
        comparable_count = sum(1 for v in essential_status.values() if v in ("IN_RANGE", "OUT_OF_RANGE"))
        if reference_error:
            status = "UNKNOWN"
        elif comparable_count < len(essential):
            status = "UNKNOWN"  # 필수 영양소 중 하나라도 comparable X
        else:
            pass_count = sum(1 for v in essential_status.values() if v == "IN_RANGE")
            fail_count = sum(1 for v in essential_status.values() if v == "OUT_OF_RANGE")
            if fail_count > 0:
                status = "FALSE"
            elif pass_count == len(essential):
                status = "TRUE"  # 필수 영양소 전부 PASS
            else:
                status = "UNKNOWN"  # 모두 comparable 인데 pass/fail 0 (이론상 불가, 안전망)

        out[pid] = {
            # P0-D: field name 중립화 (AAFCO 적합 판정 X, 영양 비교 상태만)
            # 진짜 판정 = nutrition_comparison_status (3단계 TRUE/FALSE/UNKNOWN)
            "nutrition_comparison_status": status,
            # DEPRECATED (9/10 옵션 C): 하위 호환용 bool 필드
            # - UNKNOWN(데이터 부족)은 None으로 보존
            # - 새 코드는 nutrition_comparison_status (str) 를 사용
            # - 후속 PR에서 제거 권고 (DB 컬럼·API contract 모두 nutrition_comparison_status 로 이전)
            "aafco_pass": (
                True if status == "TRUE" else False if status == "FALSE" else None
            ),
            "status": status,
            "comparable_count": comparable_count,
            "essential_count": len(essential),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "missing_count": sum(1 for v in essential_status.values() if v in ("MISSING", "NON_COMPARABLE")),
            "essential_nutrients": essential,
            "essential_status": essential_status,
            # 하위 호환 (build_summary 에서 사용): 기존 checked_nutrients 형식으로 변환
            "checked_nutrients": [
                {
                    "nutrient_code": n,
                    "value": None,
                    "min": None,
                    "max": None,
                    "pass": essential_status[n] == "IN_RANGE",
                }
                for n in essential
            ],
            "warning_codes": [reference_error] if reference_error else [],
        }
    return out


# 하위 호환 alias
def compute_aafco_pass(items: list, product_species_map: dict[str, str] | None = None, product_life_stage_map: dict[str, str] | None = None) -> dict:
    """하위 호환 alias. compute_nutrition_comparison_status 로 위임. (P0-D 9/4 11:00 deprecated)"""
    return compute_nutrition_comparison_status(items, product_species_map, product_life_stage_map)


# ============================================================================
# OPFF cache → 시드 13 형식 row 변환
# ============================================================================
def opff_to_s13_rows(opff_cache: dict) -> list:
    """OPFF v3 API 응답 (`<key>_value` 형식) 의 18종 영양소를 시드 13 row 형식으로 변환.
    - 8 무기질 (Ca/P/Na/Mg/K/Fe/Cu/Zn): OPFF `_value` = per-1kg, `_100g` = per-100g
      → `_100g` 우선 사용, 없으면 `_value * 0.1` (per-1kg → per-100g, SI 0.1)
      → 5종 (Ca/P/Na/Mg/K) unit=GRAM AF, 3종 (Fe/Cu/Zn) unit=MILLIGRAM 변환 (`× 1000`, SI)
    - 단위 변환: OPFF의 `<key>_unit` 확인, mg → g (/1000)
    - value<0 또는 null → null
    - proteins_value / fat_value (단위 g) 가 0.0 인 경우 placeholder 로 간주하여 무시
    """
    out = []
    for code, entry in opff_cache.items():
        if not entry.get("ok"):
            continue
        nutriments = entry.get("nutriments", {})
        for opff_key, nias_code in OPFF_NIAS_MAP.items():
            val_key = f"{opff_key}_value"
            unit_key = f"{opff_key}_unit"
            # ⑤-D 변경: 8 무기질은 _100g 우선, 없으면 _value * 0.1
            if opff_key in MINERAL_ALL:
                v_100g = nutriments.get(f"{opff_key}_100g")
                v_raw = nutriments.get(val_key)
                if v_100g is not None and v_100g != "" and v_100g != 0 and v_100g != 0.0:
                    v = float(v_100g)
                    _per100g_src = "100g_direct"
                elif v_raw is not None and v_raw != "" and v_raw != 0 and v_raw != 0.0:
                    v = float(v_raw) * 0.1  # per-1kg → per-100g
                    _per100g_src = "value_x_0.1"
                else:
                    continue
                # 3종 MILLIGRAM 변환 (g/100g → mg/100g, ×1000)
                if opff_key in MINERAL_MILLIGRAM:
                    v = v * 1000.0
                v_out = v
                basis = "AS_FED"
                unit_out = "g/100g" if opff_key in MINERAL_GRAM else "mg/100g"
                unit = "g"
                factor = 1.0
                _opff_raw_100g = nutriments.get(f"{opff_key}_100g")
                _opff_raw_value = nutriments.get(val_key)
                out.append({
                    "product_id": code,
                    "nutrient_code": nias_code,
                    "value": v_out,
                    "unit": unit_out,
                    "basis": basis,
                    "source": "OPFF_API_v3",
                    "_opff_raw_value": _opff_raw_value,
                    "_opff_raw_100g": _opff_raw_100g,
                    "_opff_unit": unit,
                    "_opff_unit_note": (
                        f"unit={unit} per_100g_src={_per100g_src} "
                        f"per_100g={(_opff_raw_100g if _opff_raw_100g not in (None,'') else (float(_opff_raw_value or 0)*0.1)):.6f} "
                        f"factor={factor} basis={basis}"
                    ),
                })
                continue
            # 기존 로직 (비타민/비-무기질)
            v = nutriments.get(val_key)
            if v is None or v == "" or v == 0 or v == 0.0:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v < 0:
                continue
            # 단위 변환
            unit = nutriments.get(unit_key, "g")
            unit_norm = unit.lower().replace("μ", "µ")
            factor = UNIT_FACTOR.get(unit_norm, 1.0)
            if factor is None:
                # IU 등 변환 보류 → 이 row 는 스킵
                continue
            v_out = v * factor
            basis = "AS_FED"
            unit_out = "PERCENT" if unit_norm == "%" else "g/100g"
            _prvit_iu_factor = None
            # PR-Vit (9/3 15:00): VITAMIN A/D/E IU conversion (NIAS unit=IU 정합)
            # NIAS v5 의 VITAMIN 24 row 는 unit=IU, basis=DRY_MATTER (⑩ §6.2 인용)
            # OPFF 의 vitamin-a/d/e_value 는 g/100g AF → IU/100g AF 변환.
            # AF→DM 변환은 normalize_basis (PR-Vit 분기) 에서 s13/s9b moisture 기반으로 적용
            # (OPFF moisture 가 아닌 s13 moisture 사용 이유: form=UNKNOWN 인 4047777125013 의 OPFF moisture=79% 는 AF→DM 미적용, dry-run §3.3 정합)
            # IU 환산 계수 (NIAS/AAFCO/IUPAC 권고): retinol 1 mg=3,333 IU, cholecalciferol 1 mg=40,000 IU, α-tocopherol 1 mg=1.493 IU
            if nias_code in ("VITAMIN_A", "VITAMIN_D", "VITAMIN_E"):
                IU_FACTOR_G = {
                    "VITAMIN_A": 3333000.0,  # 1 mg retinol = 3,333 IU (× 1000 → 1 g = 3,333,000 IU)
                    "VITAMIN_D": 40000000.0,  # 1 mg cholecalciferol = 40,000 IU (× 1000 → 1 g = 40,000,000 IU)
                    "VITAMIN_E": 1493.0,  # 1 mg α-tocopherol = 1.493 IU (× 1000 → 1 g = 1,493 IU)
                }
                v_out = v_out * IU_FACTOR_G[nias_code]  # g/100g AF → IU/100g AF
                unit_out = "IU/100g"  # PR-Vit: NIAS unit 정합
                _prvit_iu_factor = IU_FACTOR_G[nias_code]
            elif unit_norm == "%":
                # basis 가정: 4종은 DRY_MATTER/AS_FED 혼재이므로 % 그대로 (NIAS는 g/100g=%)
                basis = "DRY_MATTER" if nias_code in ("CRUDE_PROTEIN", "CRUDE_FAT", "CRUDE_FIBER") else "AS_FED"
            out.append({
                "product_id": code,
                "nutrient_code": nias_code,
                "value": v_out,
                "unit": unit_out,
                "basis": basis,
                "source": "OPFF_API_v3",
                "_opff_raw_value": v,
                "_opff_unit": unit,
                "_opff_unit_note": f"unit={unit} factor={factor}" + (
                    f" prvit_iu_factor={_prvit_iu_factor} basis_after={basis}" if _prvit_iu_factor else ""
                ),
            })
    return out


# ============================================================================
# 시드 13 row 확장 (60 → 100+)
# ============================================================================
def expand_s13(
    s13_items: list,
    s9_items: list,
    s9b_products: list,
    opff_rows: list,
) -> tuple[list, dict]:
    """시드 13 raw 60 row + 시드 9의 OPFF guaranteed_analysis 4종 + 시드 9b의 OFF_KR_xxx 4종 placeholder
    + OPFF API 14종. 시드 9b 의 nutrients 비어있는 product 는 placeholder row 0 으로 추가 (확장 카운트용)."""
    out = list(s13_items)  # 60 row

    # 시드 9의 guaranteed_analysis (4종) → 시드 13 row 형식 변환
    s9_added = 0
    s9_pids = {r["product_id"] for r in s13_items}
    s9_nutrient_map = {
        "crude_protein_pct": "CRUDE_PROTEIN",
        "crude_fat_pct": "CRUDE_FAT",
        "crude_fiber_pct": "CRUDE_FIBER",
        "moisture_pct": "MOISTURE",
    }
    for p in s9_items:
        pid = p.get("product_id")
        if not pid or pid in s9_pids:
            continue
        ga = p.get("guaranteed_analysis", {}) or {}
        for k, code in s9_nutrient_map.items():
            v = ga.get(k)
            if v is None or v == 0 or v == 0.0:
                continue
            out.append({
                "product_id": pid,
                "nutrient_code": code,
                "value": float(v) if v is not None else None,
                "unit": "PERCENT",
                "basis": "AS_FED" if code == "MOISTURE" else "DRY_MATTER",
                "source": "OPFF_placeholder",
            })
            s9_added += 1
        s9_pids.add(pid)

    # 시드 9b의 OFF_KR_xxx 189 product 중 4종 placeholder (값 0) 추가 → product_id 확장
    s9b_added = 0
    for p in s9b_products:
        pid_raw = p.get("product_id", "")
        if pid_raw.startswith("OFF_KR_"):
            pid = pid_raw[len("OFF_KR_"):]
        elif pid_raw.startswith("OFF_"):
            pid = pid_raw[len("OFF_"):]
        else:
            pid = pid_raw
        if not pid or len(pid) != 13 or not pid.isdigit():
            continue
        if pid in s9_pids:
            continue
        # 4종 placeholder 추가 (값 0, normalize 에서 null 처리됨)
        for code in ORIGINAL_4:
            out.append({
                "product_id": pid,
                "nutrient_code": code,
                "value": None,  # 시드 9b는 4종 데이터 없음
                "unit": "PERCENT",
                "basis": "AS_FED",
                "source": "OFF_KR_placeholder",
                "_placeholder": True,
            })
            s9b_added += 1
        s9_pids.add(pid)

    # PR-D 변경 (9/3 14:25): items list 에 MOISTURE value 가 있는 product set 추적
    # s9b OFF_KR_xxx placeholder (value=None) 는 제외, 진짜 MOISTURE 보유 product 만 포함
    # → OPFF 의 ORIGINAL_4 (CRUDE_PROTEIN/FAT/FIBER/MOISTURE) skip 정책의 예외 조건
    products_with_moisture = {r["product_id"] for r in out if r.get("nutrient_code") == "MOISTURE" and r.get("value") is not None}
    # PR-D: 3 PR-D target product (0064992282189/0064992718916/0064992718930) 만 OPFF moisture 통합
    # dry-run 검증 범위 (0 product 변화) 와 정확히 일치시키기 위한 한정.
    # 다른 product (예: 0015561526517/0015561526531/4047777125013) 의 OPFF moisture 는
    # 별도 PR (OPFF moisture architecture 일반화) 에서 처리 (급하지 않음, ⑩ §3.2 (b) 외 영향).
    PR_D_MOISTURE_TARGETS = {"0064992282189", "0064992718916", "0064992718930"}

    # OPFF API 의 14종 추가 영양소 (전체 297 product 에서 ok 된 것)
    opff_added = 0
    for r in opff_rows:
        pid = r.get("product_id")
        if not pid:
            continue
        # 4종은 시드 9/13 에서 이미 수집됨, OPFF의 4종은 무시 (시드 9/13 우선)
        # PR-D 예외: MOISTURE 만 3 PR-D target product 에 한해 OPFF moisture 추가
        # ⑤-D §1.3 / §6.1 OPFF moisture architecture issue 해결 (3 product mineral 8 row INVALID → KNOWN)
        if r.get("nutrient_code") in ORIGINAL_4:
            if (r.get("nutrient_code") == "MOISTURE"
                    and pid in PR_D_MOISTURE_TARGETS
                    and pid not in products_with_moisture
                    and r.get("value") is not None):
                out.append({
                    "product_id": pid,
                    "nutrient_code": "MOISTURE",
                    "value": r.get("value"),
                    "unit": r.get("unit", "PERCENT"),
                    "basis": "AS_FED",
                    "source": "OPFF_API_v3",
                })
                opff_added += 1
            continue
        out.append({
            "product_id": pid,
            "nutrient_code": r.get("nutrient_code"),
            "value": r.get("value"),
            "unit": r.get("unit", "g/100g"),
            "basis": "AS_FED",
            "source": "OPFF_API_v3",
        })
        opff_added += 1

    return out, {"s9_added": s9_added, "s9b_placeholder_added": s9b_added, "opff_added": opff_added}


# ============================================================================
# 10가지 판정 기준 산출
# ============================================================================
def build_summary(
    expanded_raw: list,
    normalized: list,
    aligned: list,
    outliers: list,
    classified: list,
    basis_norm: list,
    compared: list,
    aafco_results: dict,
    nias_18_supp: list,
    opff_stats: dict,
) -> dict:
    """10가지 판정 기준 결과 dict 반환."""
    target_pids = sorted(set(r["product_id"] for r in expanded_raw if r.get("product_id")))
    expanded_pids = sorted(set(r["product_id"] for r in expanded_raw if r.get("product_id")))

    # 1) 18종 nutrient별 수집/매핑 건수
    by_nutrient_raw = Counter(r.get("nutrient_code") for r in expanded_raw if r.get("nutrient_code"))
    by_nutrient_nonnull = Counter(
        r.get("nutrient_code") for r in expanded_raw
        if r.get("nutrient_code") and r.get("value") is not None
    )

    # 2) 기존 4종 vs 18종 coverage
    n4 = sum(by_nutrient_nonnull.get(c, 0) for c in ORIGINAL_4)
    n18 = sum(by_nutrient_nonnull.values())

    # 3) product 기준 매핑률
    product_with_any = set()
    for r in expanded_raw:
        if r.get("value") is not None and r.get("product_id"):
            product_with_any.add(r["product_id"])
    rate_pct = (len(product_with_any) / len(expanded_pids) * 100.0) if expanded_pids else 0.0

    # 4) nutrient 기준 coverage (영양소별 %)
    nutrient_coverage = {}
    for c in TARGET_NUTRIENTS_18:
        nn = by_nutrient_nonnull.get(c, 0)
        nt = by_nutrient_raw.get(c, 0)
        nutrient_coverage[c] = {
            "collected_row": nt,
            "non_null_row": nn,
            "coverage_pct": (nn / nt * 100.0) if nt > 0 else 0.0,
        }

    # 5) TRUE/FALSE/UNKNOWN 분포
    aafco_dist = Counter(v["status"] for v in aafco_results.values())

    # 6) 새로 TRUE / 7) 새로 FALSE / 8) UNKNOWN 이유
    aafco_breakdown = {}
    for pid, r in aafco_results.items():
        aafco_breakdown[pid] = {
            "status": r["status"],
            "comparable_count": r["comparable_count"],
            "pass_count": r["pass_count"],
            "fail_count": r["fail_count"],
            "checked_nutrients": r["checked_nutrients"],
            "warning_codes": r.get("warning_codes", []),
        }

    # 9) 100+ row 중 실제 유효 vs 결측/무효
    total_rows = len(expanded_raw)
    null_rows = sum(1 for r in expanded_raw if r.get("value") is None)
    valid_rows = total_rows - null_rows

    # 10) P2 진행 가능성
    total_products = len(expanded_pids)
    mappable_products = len(product_with_any)
    determined_products = sum(
        1 for v in aafco_results.values()
        if v["status"] in ("TRUE", "FALSE")
    )
    unknown_products = sum(1 for v in aafco_results.values() if v["status"] == "UNKNOWN")
    p2_recommendation = {
        "total_products": total_products,
        "mappable_products": mappable_products,
        "determined_products": determined_products,
        "unknown_products": unknown_products,
        "determined_rate_pct": (determined_products / total_products * 100.0) if total_products else 0.0,
        "verdict": (
            "P2 진행 가능: 18종 NIAS + 100+ row 확보, 4종만으로도 판정 가능 product 존재"
            if determined_products >= 1
            else "P2 보류 권장: 결정 가능 product 0개, 4종 데이터 추가 수집 후 재시도"
        ),
    }

    return {
        "metadata": {
            "phase": "P1-C",
            "version": "v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "description": "P1-C 단일 source of truth 파이프라인 10가지 판정 기준",
        },
        "opff_stats": opff_stats,
        "1_nutrient_collected_counts": {
            "by_nutrient_total_row": dict(by_nutrient_raw),
            "by_nutrient_non_null": dict(by_nutrient_nonnull),
            "target_nutrients_18": TARGET_NUTRIENTS_18,
            "original_4": ORIGINAL_4,
        },
        "2_4_vs_18_coverage": {
            "original_4_non_null_rows": n4,
            "expanded_18_non_null_rows": n18,
            "growth_factor": (n18 / n4) if n4 else None,
        },
        "3_product_mapping_rate": {
            "total_products": total_products,
            "mappable_products": mappable_products,
            "rate_pct": round(rate_pct, 2),
            "previous_p1b4": "3.15% (9/286)",
        },
        "4_nutrient_coverage": nutrient_coverage,
        "5_aafco_pass_distribution": {
            "TRUE": aafco_dist.get("TRUE", 0),
            "FALSE": aafco_dist.get("FALSE", 0),
            "UNKNOWN": aafco_dist.get("UNKNOWN", 0),
            "previous_p1b4": "TRUE=2, FALSE=3, UNKNOWN=309",
        },
        "6_newly_TRUE_products": [
            {"product_id": pid, "details": aafco_breakdown[pid]}
            for pid, r in aafco_results.items() if r["status"] == "TRUE"
        ],
        "7_newly_FALSE_products": [
            {"product_id": pid, "details": aafco_breakdown[pid]}
            for pid, r in aafco_results.items() if r["status"] == "FALSE"
        ],
        "8_unknown_reasons": {
            pid: aafco_breakdown[pid]
            for pid, r in aafco_results.items() if r["status"] == "UNKNOWN"
        },
        "9_validity_breakdown": {
            "total_rows": total_rows,
            "valid_rows_non_null": valid_rows,
            "null_rows": null_rows,
            "valid_rate_pct": (valid_rows / total_rows * 100.0) if total_rows else 0.0,
        },
        "10_p2_recommendation": p2_recommendation,
    }


# ============================================================================
# main
# ============================================================================
def main():
    print("=" * 80)
    print("P1-C 단일 source of truth 파이프라인")
    print("=" * 80)

    # 1) load_raw
    raw = load_raw()
    print(f"[1] load_raw: s13={len(raw['s13_items'])}, s9={len(raw['s9_items'])}, "
          f"s9b={len(raw['s9b_products'])}, nias={len(raw['nias_rows'])}, "
          f"opff_cache={len(raw['opff_cache'])}")

    # OPFF cache → 시드 13 row 형식 변환
    opff_rows = opff_to_s13_rows(raw["opff_cache"])
    opff_stats = {
        "cache_total": len(raw["opff_cache"]),
        "ok_count": sum(1 for v in raw["opff_cache"].values() if v.get("ok")),
        "rows_extracted": len(opff_rows),
    }
    print(f"  OPFF cache ok={opff_stats['ok_count']}/{opff_stats['cache_total']}, "
          f"18종 추출 row={opff_stats['rows_extracted']}")

    # 시드 13 row 확장
    expanded_raw, expand_stats = expand_s13(
        raw["s13_items"], raw["s9_items"], raw["s9b_products"], opff_rows,
    )
    print(f"  확장: s9 추가 {expand_stats['s9_added']} row, "
          f"OPFF 14종 추가 {expand_stats['opff_added']} row, "
          f"총 {len(expanded_raw)} row")

    # product_species_map (시드 9 / 9b / 13 의 target_species)
    product_species_map = {}
    for p in raw["s9_items"]:
        ts = p.get("target_species") or ["DOG"]
        if isinstance(ts, list):
            ts = ts[0] if ts else "DOG"
        product_species_map[p["product_id"]] = str(ts).upper()
    for p in raw["s9b_products"]:
        pid = p["product_id"]
        if pid.startswith("OFF_KR_"):
            pid = pid[len("OFF_KR_"):]
        ts = p.get("target_species", "DOG")
        product_species_map[pid] = str(ts).upper()

    # 2) normalize
    normalized = normalize(expanded_raw)
    print(f"[2] normalize: {len(normalized)} row")

    # 3) align_units
    aligned = align_units(normalized)
    print(f"[3] align_units: {len(aligned)} row")

    # 4) detect_outliers
    outliers = detect_outliers(aligned)
    flagged = sum(1 for r in outliers if r.get("outlier_status") == "OUTLIER_FLAGGED")
    print(f"[4] detect_outliers: {flagged} OUTLIER_FLAGGED")

    # 5) classify_dry_wet
    classified = classify_dry_wet(outliers)
    forms = Counter(r.get("product_form") for r in classified)
    print(f"[5] classify_dry_wet: {dict(forms)}")

    # 6) normalize_basis
    basis_norm = normalize_basis(classified)
    converted = sum(1 for r in basis_norm if r.get("basis_normalization_status") == "CONVERTED")
    print(f"[6] normalize_basis: {converted} CONVERTED")

    # 7) compare_nias
    compared, nias_18_supp, supplemented = compare_nias(
        basis_norm, raw["nias_rows"], product_species_map,
    )
    in_range = sum(1 for r in compared if r.get("nias_compare_status") == "IN_RANGE")
    oor = sum(1 for r in compared if r.get("nias_compare_status") == "OUT_OF_RANGE")
    no_ref = sum(1 for r in compared if r.get("nias_compare_status") == "NO_REF")
    print(f"[7] compare_nias: in_range={in_range}, oor={oor}, no_ref={no_ref}, "
          f"max_supplement={len(supplemented)}")

    # 8) compute_aafco_pass
    aafco_results = compute_aafco_pass(compared, product_species_map)
    aafco_dist = Counter(v["status"] for v in aafco_results.values())
    print(f"[8] compute_aafco_pass: {dict(aafco_dist)}")

    # NIAS 18종 source of truth 저장
    nias_sot = {
        "_metadata": {
            "title": "seed_14 NIAS 18종 source of truth v1 (P1-C)",
            "source": "nutrition_reference_nias_2024_parity_p0_v1.json (active canonical reference)",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "phase": "P1-C",
            "target_nutrients_18": TARGET_NUTRIENTS_18,
        },
        "rows": nias_18_supp,
        "max_supplements": supplemented,
    }
    (PROC / "seed_14_p1c_source_of_truth_v1.json").write_text(
        json.dumps(nias_sot, ensure_ascii=False, indent=2),
    )
    print(f"  저장: seed_14_p1c_source_of_truth_v1.json ({len(nias_18_supp)} row, "
          f"supplement {len(supplemented)})")

    # 시드 13 expanded 저장
    expanded_doc = {
        "_metadata": {
            "title": "seed_13 P1-C expanded v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "phase": "P1-C",
            "description": "시드 13 raw 60 row + 시드 9 OPFF 4종 + OPFF API 14종",
            "total_row": len(expanded_raw),
        },
        "items": expanded_raw,
        "expand_stats": expand_stats,
    }
    (PROC / "seed_13_p1c_expanded_v1.json").write_text(
        json.dumps(expanded_doc, ensure_ascii=False, indent=2),
    )
    print(f"  저장: seed_13_p1c_expanded_v1.json ({len(expanded_raw)} row)")

    # 시드 13 source of truth (파이프라인 통과 후)
    s13_sot = {
        "_metadata": {
            "title": "seed_13 P1-C source of truth v1 (8단계 파이프라인 통합)",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "phase": "P1-C",
            "description": (
                "P1-A (value=0→null) + P1-A.1 (단위 정렬) + P1-B-1 (outlier) + "
                "P1-B-2 (DRY/WET) + P1-B-3 (basis) + P1-B-4 (max 보강) 통합"
            ),
            "input_row": len(expanded_raw),
            "output_row": len(compared),
        },
        "items": compared,
        "aafco_pass_per_product": aafco_results,
    }
    (PROC / "seed_13_p1c_source_of_truth_v1.json").write_text(
        json.dumps(s13_sot, ensure_ascii=False, indent=2),
    )
    print(f"  저장: seed_13_p1c_source_of_truth_v1.json ({len(compared)} row)")

    # 10가지 판정 기준 summary
    summary = build_summary(
        expanded_raw, normalized, aligned, outliers, classified,
        basis_norm, compared, aafco_results, nias_18_supp, opff_stats,
    )
    (PROC / "seed_13_p1c_summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    print(f"  저장: seed_13_p1c_summary_v1.json")

    print("\n" + "=" * 80)
    print("P1-C 결과 요약")
    print("=" * 80)
    print(f"시드 13 row: {len(raw['s13_items'])} → 확장 {len(expanded_raw)}")
    print(f"aafco_pass 분포: {dict(aafco_dist)}")
    print(f"매핑률: {summary['3_product_mapping_rate']['rate_pct']}% "
          f"({summary['3_product_mapping_rate']['mappable_products']}/"
          f"{summary['3_product_mapping_rate']['total_products']})")
    print(f"P2 권고: {summary['10_p2_recommendation']['verdict']}")


if __name__ == "__main__":
    main()
