# Nutrition Input Integration, 2026-10-05

## Contract

Backend `sever/dev` adds `pet.allergy_profile_status`, default `UNKNOWN`, constrained to `UNKNOWN`, `KNOWN_NONE`, `KNOWN_LIST`. Register/PATCH/detail use JSON `allergyProfileStatus`. Existing nonempty allergy lists are backfilled as `KNOWN_LIST`; historical empty lists remain `UNKNOWN`. A declaration inconsistent with the list is rejected with HTTP 400. Unrelated Pet updates preserve the stored status.

Frontend `web/dev` sends `KNOWN_NONE` only after an explicit none selection, together with an empty array. It restores that checkbox only from the saved state. Missing status from an older Backend plus an empty list remains `UNKNOWN`.

Nutrition reads the column through `to_jsonb(pet)->>'allergy_profile_status'`, so an older schema returns null rather than a SQL missing-column error. An omitted analysis/compare request status preserves the repository state; explicit null/UNKNOWN or a conflicting declaration remains UNKNOWN. No source writes occur in Nutrition.

Pet life stage is computed by Nutrition from ISO `birth_date`, using completed calendar months in Asia/Seoul. Under 12 months is `GROWTH_REPRODUCTION`; otherwise `ADULT_MAINTENANCE`. Invalid/future dates are rejected. Pregnancy/lactation are never inferred. Product stage and reference coverage remain independent evidence gates.

## Mapping

[Service code mapping](service_allergen_mapping_v2.csv) covers all 38 Backend enum values. 33 food codes are supported; four TOXIC values remain in a separate namespace and OTHER needs actual free-text evidence. The new service-specific identities use Backend revision `c23e7a1ab1b1a31ff32a84e19a50eab7c28bf5c9`, `modules/common-core/src/main/java/com/golajugaenyang/common/core/domain/AllergenCode.java`. This is exact identity mapping, not a clinical cross-reactivity policy or a replacement for the v3 persisted evidence catalog.

Pet TUNA stays tuna, SALMON stays salmon. A product's existing v3 parent fish alias remains evidence for a broad FISH profile. Generic fish evidence cannot clear a specific fish profile. Unregistered names, partial matches, and ambiguous material remain unresolved.

For product 141, lamb/oat/potato allergen references remain resolved. Carrot and beet have exact ingredient identities `CARROT` and `BEET` with `ingredient_resolution_status=RESOLVED`; their `allergen_code=null` and `mapping_method=UNRESOLVED` remain unchanged. The source is the observed local `ai/data/raw/seed_15_ingredient_synonym_v2.json`, SHA256 `1b87866d1b6369bb93bc2c3e2bce70a30507fe8572142955201e2393de993393`. Its null allergen fields are missing evidence, not reviewed NON_ALLERGEN decisions. A KNOWN_LIST profile can therefore still receive `UNMAPPED_INGREDIENT` for 141. KNOWN_NONE skips the allergy comparison while other safety gates remain active.

## Live Baseline

Read-only Gateway calls on 2026-10-05 authenticated two owned Pets and analyzed products 141 and 302, with request status omitted and with KNOWN_LIST. The token was read from non-echoing stdin, not written to files.

- Backend detail has no stored profile field in the running version.
- DUCK remains ALLERGY_PROFILE_UNKNOWN, even with KNOWN_LIST.
- CHICKEN with omitted status remains ALLERGY_PROFILE_UNKNOWN; explicit KNOWN_LIST on 141 reaches UNMAPPED_INGREDIENT for carrot/beet.
- Both Pets resolve ADULT_MAINTENANCE. Product 302 has zero nutrition_items and, for the CHICKEN profile, an exact service allergy conflict plus PRODUCT_LIFE_STAGE_UNKNOWN. It is not a missing Pet stage.
- An unauthenticated Nutrition Gateway request returns 401.

These are baseline server observations. They do not demonstrate deployment of this local patch.

## Release Order

Local verification: Nutrition 616 tests passed; Backend common-core/member-service 22 tests passed; Frontend typecheck, formatting, lint, unit suite (1964 tests), production build, browser E2E (173), and server-fetch E2E (87) passed. Three added detail-state cases also passed in a 42-test related suite. PostgreSQL migration execution against the actual Service DB and post-deployment Gateway verification have not been performed.

1. Review and deploy Backend V3 migration with member-service. Do not backfill empty lists as KNOWN_NONE.
2. Deploy Nutrition after review. The optional-column read also supports an older Backend schema.
3. Frontend remains local-only until its owner approves release.
4. Validate explicit none registration/update/detail, owned Pet analysis, DUCK/TURKEY/TUNA exact matching, 141 identity versus evidence, and 302 product-stage/nutrient gaps through the public Gateway.

Release requires separate PR-head CI, post-merge pipeline, migration, actual image/SHA/Ready, and authenticated Gateway evidence. None of those runtime gates is established by the local tests above. GitOps dev member-service currently selects the dev profile with Flyway baseline-on-migrate, whereas the Backend infra profile disables Flyway; the actual applied V3 migration must be checked rather than inferred from either file. Current Kubernetes access is unconfirmed; historical EKS DNS/tunnel addresses must not be reused. Carrot/beet safety evidence, 302 manufacturer nutrient/stage evidence, and the original Notion document URL remain unresolved external inputs.
