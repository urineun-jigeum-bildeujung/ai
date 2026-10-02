# Service-ID Feeding contract — MER policy unresolved

Version: feeding_rule_v1. Implementation and local tests do not imply deployment
or authenticated E2E verification.

`POST /api/nutrition/analyze/by-service-id` adds a `feeding` object. Existing
Nutrition comparison, readiness, provenance and Safety retain their own axes.
The legacy analyze/by-product-id routes remain unchanged.

## Formula and units

RER = 70 × weight_kg^0.75 (kcal/day).
MER = RER × approved coefficient (kcal/day).
daily_serving_g = MER / (energy_density_kcal_per_kg / 1000).

RER is a resting-energy estimate, not a feeding recommendation or prescription.
Use unrounded intermediate values. Presentation uses one decimal with
ROUND_HALF_UP; nonfinite, overflow, underflow and zero-rounded doses fail closed.

References:
- Project formula / Mock energy ranges: [AI 통합 데이터 스키마 §3-6/3-7](https://app.notion.com/p/29baa56b8d3483a7be1401b8800df6a5).
- Exponential RER formula: [Merck Veterinary Manual](https://www.merckvetmanual.com/management-and-nutrition/nutrition-small-animals/nutritional-requirements-of-small-animals).

## MER coefficient policy

No approved versioned single-value coefficient table was found in current
Backend `sever/dev` revision `230e598833f684c6c9f2ce605776e230a5b6f236`,
AI docs/data, or the relevant Notion contract. The historical 80-row seed claim
and planning ranges are not an approved runtime selection rule.

`mer_coefficient_unresolved_v1` always returns a null coefficient with
`MER_COEFFICIENT_UNRESOLVED`. It ignores numeric coefficients supplied in source
records. The HTTP request remains service IDs only and rejects extra fields.
MER and daily_serving_g remain null until a reviewed policy is implemented.
Valid weight and DOG/CAT species may still expose RER independently.

## Mock energy

`mock_energy_v1.json` is a Nutrition-owned integration fixture, not manufacturer
measurement. The project Mock DRY_FOOD 3200–4500 and WET_FOOD 500–1000 kcal/kg
ranges give deterministic midpoint fixtures 3850 and 750 on an AS_FED basis.
These fixture values do not select or approve a MER coefficient.

Only matching existing Service Mock identity, resolved READY/PARTIAL nutrition
fixture provenance and explicit DRY_FOOD/WET_FOOD forms receive Mock energy.
Unavailable profiles, other forms/categories, mismatched identity, missing or
corrupt energy artifacts return missing energy. Production SKU evidence gets
no synthetic calorie fallback.

## Output

`feeding` contains status, daily_serving_g, rer_kcal_per_day, mer_kcal_per_day,
energy_density_kcal_per_kg, energy_basis, energy_source_type, energy_version,
coefficient, coefficient_code, coefficient_source_type, coefficient_version,
coefficient_policy_status, coefficient_policy_contract_version,
calculation_version and reason_codes.

INSUFFICIENT_DATA means a dose could not be calculated. BLOCKED means existing
Safety excludes the product; FEEDING_SAFETY_EXCLUDED is added and no dose/MER
is exposed. Neither result changes nutrition_comparison_status or aafco_pass.
READY is available only to the calculator with explicit valid coefficient and
provenance; current Service-ID runtime policy cannot produce READY feeding.
Unit tests use synthetic coefficients solely to verify arithmetic.

## Service persistence and life-stage gaps

Latest Pet domain, JPA entity, registration/update DTO, PetAllergy repository
and migrations have no allergy_profile_status or explicit life-stage field.
Pet allergy 0 rows remains UNKNOWN; explicit KNOWN_NONE cannot be reconstructed.
No AI DDL/write or inferred KNOWN_NONE is introduced.

Repository reads existing bcs, is_neutered and birth_date columns as additional
source input. Missing or invalid BCS/neuter state adds reasons. Birth date is
not converted into a new stage policy. The legacy Nutrition Mock age-rule
compatibility path is preserved, but Feeding sees the original Service stage;
it cannot inherit that compatibility fallback as an approved feeding policy.
