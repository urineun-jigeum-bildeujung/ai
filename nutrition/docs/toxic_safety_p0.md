# TOXIC Safety P0 — 2026-10-02

The reviewed local TOXIC patch is retained in this release. `product_cautions`
is read with SELECT in the existing read-only transaction. Only applicable
Backend TOXIC codes block; CONDITIONAL/NUTRITION policies remain undefined.
Allergen reasons and Nutrition comparison are independent and unchanged.

Source: Backend CautionIngredientCode revision
`eb48f6c20ccd132f436113cc59861f45dcd885b1`, SHA-256
`8baea45d54231c0398d716193d4460fcb1044cb64c5c86a49749000f64bfa411`.
The versioned policy contains the existing exact 23-code snapshot, including
14 TOXIC codes. No new inferred toxic policy is introduced.

Verification in the isolated current-develop worktree:

- Entire Nutrition suite with this patch: 309 passed, 1 existing warning.
- All 14 toxic codes against DOG/CAT, KNOWN_NONE, concurrent allergy conflict,
  and Nutrition TRUE preserved are covered by synthetic tests.
- Current dev SELECT audit: active 322, Mock 286, non-Mock 36;
  fixture READY/PARTIAL/UNAVAILABLE = 107/13/166; 269 caution-free, 17 with
  existing cautions. `transaction_read_only=on`; no DB writes.
- Current dev product SELECT source replayed locally with an explicit
  synthetic DOG pet: legacy domain projection unchanged for 286/286.
- Existing authenticated cohort identifiers replayed locally with synthetic
  inputs; these are not new authenticated Gateway requests or actual Pet-source
  verification. Actual-source 23-case patch verification: NOT VERIFIED.
- Live toxic E2E: NOT VERIFIED. Prior dev audit recorded zero active toxic
  rows; synthetic tests are not live operational verification.

An automatic approval review rejected transferring the archived authenticated
cohort and local patch source into the dev Pod for in-memory verification.
The command did not run. No Pod files, Secret, ConfigMap, or DB were changed.
