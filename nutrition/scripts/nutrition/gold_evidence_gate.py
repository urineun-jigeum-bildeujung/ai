"""Pure fail-close gate for human-intake Gold product evidence."""
from __future__ import annotations

from numbers import Real
from typing import Any

try:  # Support both package imports and direct script/API sibling imports.
    from .gtin_validation import is_valid_gtin
except ImportError:  # pragma: no cover - direct script/API import compatibility
    from gtin_validation import is_valid_gtin


REQUIRED = {
    "DOG": {"CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"},
    "CAT": {"CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE"},
}

VALID_LIFE_STAGES = {"ADULT_MAINTENANCE", "GROWTH_REPRODUCTION", "ALL_LIFE_STAGES"}
VALID_DOG_GROWTH_DETAILS = {"EARLY_GROWTH_AND_REPRODUCTION", "LATE_GROWTH"}
VALID_PRODUCT_FORMS = {"DRY", "CANNED"}
VALID_BASES = {"AS_FED", "DRY_MATTER"}

# These sources can directly bind a local GTIN to a labeled product variant.
# An official nutrient web page alone is deliberately not in this set.
IDENTITY_SOURCE_TYPES = {
    "PACKAGE_LABEL_WITH_GTIN",
    "MANUFACTURER_PACKAGE_IMAGE_WITH_GTIN",
    "MANUFACTURER_PRODUCT_DOCUMENT_WITH_GTIN",
    "IMPORTER_PACKAGE_LABEL_WITH_GTIN",
}
NUTRIENT_SOURCE_TYPES = {
    "MANUFACTURER_GUARANTEED_ANALYSIS",
    "MANUFACTURER_PACKAGE_LABEL",
    "OFFICIAL_MANUFACTURER",
    "IMPORTER_GUARANTEED_ANALYSIS",
}
IDENTITY_BINDING_FIELDS = (
    "candidate_id",
    "product_id",
    "canonical_gtin",
    "manufacturer",
    "manufacturer_product_name",
    "product_variant",
    "package_size",
    "market_region",
    "formula_version",
)


def _present(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _identity_provenance_complete(candidate: dict[str, Any]) -> bool:
    return (
        all(_present(candidate.get(field)) for field in IDENTITY_BINDING_FIELDS)
        and candidate.get("identity_source_type") in IDENTITY_SOURCE_TYPES
        and _present(candidate.get("identity_source_url_or_document"))
        and _present(candidate.get("identity_source_authority"))
    )


def _matching_runtime_evidence(candidate: dict[str, Any], row: dict[str, Any]) -> bool:
    """Accept only evidence explicitly bound to this exact local product identity."""
    value = row.get("value")
    return (
        row.get("verified") is True
        and all(row.get(field) == candidate.get(field) for field in IDENTITY_BINDING_FIELDS)
        and row.get("nutrient_source_type") in NUTRIENT_SOURCE_TYPES
        and row.get("guarantee_type") == "GUARANTEED"
        and isinstance(value, Real)
        and not isinstance(value, bool)
        and value > 0
        and row.get("unit") == "PERCENT"
        and row.get("basis") in VALID_BASES
        and all(_present(row.get(field)) for field in (
            "nutrient_source_url_or_document",
            "nutrient_source_authority",
            "observed_at",
            "retrieved_at",
            "verification_method",
        ))
    )


def evaluate_runtime_eligibility(candidate: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a deterministic, fail-close eligibility decision.

    The function does not compare nutrient values. It only decides whether a
    human-provided evidence chain is complete enough to permit a later runtime
    call. Retailer-only and price-comparison evidence can never make it eligible.
    """
    reasons: list[str] = []
    species = str(candidate.get("species") or "").upper()
    if candidate.get("gtin_valid") is not True or not is_valid_gtin(candidate.get("canonical_gtin")):
        reasons.append("INVALID_GTIN")
    if candidate.get("identity_state") != "VERIFIED": reasons.append("IDENTITY_NOT_VERIFIED")
    if not _identity_provenance_complete(candidate): reasons.append("IDENTITY_PROVENANCE_INSUFFICIENT")
    if candidate.get("identity_conflict") == "HARD_IDENTITY_CONFLICT": reasons.append("HARD_IDENTITY_CONFLICT")
    if species not in REQUIRED: reasons.append("SPECIES_UNRESOLVED")
    if candidate.get("life_stage") not in VALID_LIFE_STAGES: reasons.append("LIFE_STAGE_UNRESOLVED")
    if candidate.get("life_stage_evidence_state") != "VERIFIED_AUTHORITATIVE": reasons.append("LIFE_STAGE_PROVENANCE_UNVERIFIED")
    if candidate.get("product_form") not in VALID_PRODUCT_FORMS: reasons.append("PRODUCT_FORM_UNRESOLVED")
    if candidate.get("product_form_evidence_state") != "VERIFIED_AUTHORITATIVE": reasons.append("PRODUCT_FORM_PROVENANCE_UNVERIFIED")
    if species == "DOG" and str(candidate.get("life_stage")) == "GROWTH_REPRODUCTION":
        if candidate.get("life_stage_detail") not in VALID_DOG_GROWTH_DETAILS:
            reasons.append("DOG_GROWTH_LIFE_STAGE_DETAIL_UNRESOLVED")
        if candidate.get("life_stage_detail_evidence_state") != "VERIFIED_AUTHORITATIVE":
            reasons.append("DOG_GROWTH_LIFE_STAGE_DETAIL_PROVENANCE_UNVERIFIED")

    verified = [row for row in evidence if row.get("verified") is True]
    usable = [row for row in verified if _matching_runtime_evidence(candidate, row)]
    if len(usable) != len(verified): reasons.append("EVIDENCE_PROVENANCE_INSUFFICIENT")
    evidence_codes = {
        str(row.get("nutrient_code")) for row in usable
    }
    missing = sorted(REQUIRED.get(species, set()) - evidence_codes)
    if missing: reasons.append("REQUIRED_NUTRIENT_EVIDENCE_MISSING")
    return {"runtime_eligible": not reasons, "reason_codes": reasons, "missing_nutrients": missing}
