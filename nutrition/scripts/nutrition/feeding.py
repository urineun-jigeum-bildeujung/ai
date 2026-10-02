"""Independent, deterministic RER/MER calculation; no Safety mutations.

Formula contract: Notion AI 통합 데이터 스키마 — 백엔드 전달용 §3-7.
Coefficient selection and energy evidence are supplied by versioned policies;
this calculator never supplies a fallback coefficient or calorie value.
"""
from decimal import Decimal, ROUND_HALF_UP
import math

CALCULATION_VERSION = "feeding_rule_v1"


def _positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _round(value):
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def calculate_feeding(*, weight_kg, energy, coefficient, species, supported_state=True,
                      allow_energy_requirement=False):
    """Return null amounts on missing/invalid input, including numeric overflow.

Rounding is presentation-only (one decimal, half up); MER and grams use the
unrounded intermediate values. Evidence metadata is independent of readiness.
"""
    energy = energy or {}
    coefficient = coefficient or {}
    kcal = energy.get("energy_density_kcal_per_kg")
    factor = coefficient.get("coefficient")
    reasons = []
    for name, value in (("WEIGHT", weight_kg), ("ENERGY_DENSITY", kcal), ("COEFFICIENT", factor)):
        if value is None:
            if name == "COEFFICIENT":
                reasons.extend(coefficient.get("reason_codes") or ["COEFFICIENT_MISSING"])
            else:
                reasons.append(name + "_MISSING")
        elif _positive(value) is None:
            reasons.append(name + "_INVALID")
    if species not in {"dog", "cat"} or not supported_state:
        reasons.append("PET_STATE_UNSUPPORTED")
    if factor is not None and not all(coefficient.get(k) for k in (
        "coefficient_code", "coefficient_source_type", "coefficient_version",
    )):
        reasons.append("COEFFICIENT_PROVENANCE_MISSING")
    if kcal is not None and energy.get("energy_basis") != "AS_FED":
        reasons.append("ENERGY_BASIS_UNSUPPORTED")
    if kcal is not None and not all(energy.get(k) for k in ("energy_source_type", "energy_version")):
        reasons.append("ENERGY_PROVENANCE_MISSING")
    result = {
        "status": "INSUFFICIENT_DATA", "daily_serving_g": None,
        "rer_kcal_per_day": None, "mer_kcal_per_day": None,
        "energy_density_kcal_per_kg": _positive(kcal),
        "coefficient": _positive(factor),
        "coefficient_code": coefficient.get("coefficient_code"),
        "coefficient_source_type": coefficient.get("coefficient_source_type"),
        "coefficient_version": coefficient.get("coefficient_version"),
        "energy_source_type": energy.get("energy_source_type"),
        "energy_version": energy.get("energy_version"),
        "energy_basis": energy.get("energy_basis"),
        "coefficient_policy_status": coefficient.get("policy_status"),
        "coefficient_policy_contract_version": coefficient.get("policy_contract_version"),
        "calculation_version": CALCULATION_VERSION,
        "reason_codes": reasons,
        "data_generation_type": energy.get("data_generation_type"),
        "production_evidence": energy.get("production_evidence"),
    }
    # RER is an independent resting-energy estimate; it is not a feeding dose.
    if _positive(weight_kg) is not None and species in {"dog", "cat"}:
        try:
            rer = 70 * _positive(weight_kg) ** 0.75
            rounded_rer = _round(rer)
            if not math.isfinite(rer) or rounded_rer <= 0:
                raise ValueError("invalid RER")
            result["rer_kcal_per_day"] = rounded_rer
        except (ArithmeticError, ValueError):
            result["reason_codes"].append("CALCULATION_INVALID")
    if (allow_energy_requirement and result["reason_codes"] == ["ENERGY_DENSITY_MISSING"]
            and result["rer_kcal_per_day"] is not None):
        try:
            mer = _round(rer * _positive(factor))
            if not math.isfinite(mer) or mer <= 0:
                raise ValueError("invalid MER")
            result.update(status="ENERGY_REQUIREMENT_READY", mer_kcal_per_day=mer)
        except (ArithmeticError, ValueError):
            result["reason_codes"].append("CALCULATION_INVALID")
        return result
    if result["reason_codes"]:
        return result
    try:
        mer = rer * _positive(factor)
        grams = mer / (_positive(kcal) / 1000)
        if any(not math.isfinite(v) or v <= 0 for v in (rer, mer, grams)):
            raise ValueError("invalid calculation")
        amounts = tuple(_round(v) for v in (rer, mer, grams))
        if any(v <= 0 for v in amounts):
            raise ValueError("rounded amount is not positive")
    except (ArithmeticError, ValueError):
        result["reason_codes"] = ["CALCULATION_INVALID"]
        return result
    result.update(status="READY", rer_kcal_per_day=amounts[0],
                  mer_kcal_per_day=amounts[1], daily_serving_g=amounts[2])
    return result
