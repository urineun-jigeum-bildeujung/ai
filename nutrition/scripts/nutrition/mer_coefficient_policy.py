"""No approved single-value MER table exists in the verified project contract.

Backend sever/dev 230e598 and Notion schema section 3-7 expose inputs and
planning ranges. They do not authorize choosing a range endpoint or midpoint.
This resolver intentionally cannot turn request-supplied numbers into policy.
"""
POLICY_CONTRACT_VERSION = "mer_coefficient_unresolved_v1"


def resolve_mer_coefficient(pet):
    reasons = ["MER_COEFFICIENT_UNRESOLVED"]
    if pet.get("life_stage") in {None, "", "UNKNOWN"}:
        reasons.append("PET_LIFE_STAGE_UNRESOLVED")
    bcs = pet.get("bcs")
    if isinstance(bcs, bool) or not isinstance(bcs, int) or not 1 <= bcs <= 5:
        reasons.append("BCS_MISSING_OR_INVALID")
    if not isinstance(pet.get("is_neutered"), bool):
        reasons.append("NEUTER_STATUS_MISSING_OR_INVALID")
    return {"coefficient": None, "coefficient_code": None,
            "coefficient_source_type": None, "coefficient_version": None,
            "policy_status": "UNRESOLVED", "policy_contract_version": POLICY_CONTRACT_VERSION,
            "reason_codes": reasons}
