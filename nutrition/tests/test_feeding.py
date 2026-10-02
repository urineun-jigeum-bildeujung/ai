"""Calculation tests use explicit synthetic coefficients, not approved policy."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/nutrition"))
from feeding import calculate_feeding, _round
from mock_feeding_fixture import mock_energy


def run(**kwargs):
    inputs = dict(weight_kg=8, species="dog",
                  energy={"energy_density_kcal_per_kg": 3850,
                          "energy_basis": "AS_FED", "energy_source_type": "SYNTHETIC_UNIT_TEST", "energy_version": "test_v1"},
                  coefficient={"coefficient": 1.5, "coefficient_code": "TEST_ONLY",
                               "coefficient_source_type": "SYNTHETIC_UNIT_TEST", "coefficient_version": "test_v1"})
    inputs.update(kwargs)
    return calculate_feeding(**inputs)


@pytest.mark.parametrize("species", ["dog", "cat"])
@pytest.mark.parametrize("kcal,grams", [(3850, 129.7), (750, 666.0)])
def test_species_and_form_formula(species, kcal, grams):
    result = run(species=species, energy={"energy_density_kcal_per_kg": kcal,
                                        "energy_basis": "AS_FED", "energy_source_type": "SYNTHETIC_UNIT_TEST", "energy_version": "test_v1"})
    assert result["status"] == "READY"
    assert result["rer_kcal_per_day"] == 333.0
    assert result["mer_kcal_per_day"] == 499.5
    assert result["daily_serving_g"] == grams


@pytest.mark.parametrize("value", [None, 0, -1, float('nan'), float('inf'), True, "8"])
def test_weight_fail_close(value):
    result = run(weight_kg=value)
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["daily_serving_g"] is None
    assert result["reason_codes"] == ["WEIGHT_MISSING" if value is None else "WEIGHT_INVALID"]


@pytest.mark.parametrize("field,prefix", [("energy", "ENERGY_DENSITY"), ("coefficient", "COEFFICIENT")])
@pytest.mark.parametrize("value", [None, 0, -1, float('nan'), float('inf'), True])
def test_kcal_and_coefficient_fail_close(field, prefix, value):
    inputs = run()
    meta = ({"energy_density_kcal_per_kg": value, "energy_basis": "AS_FED", "energy_source_type": "TEST", "energy_version": "v1"}
            if field == "energy" else {"coefficient": value, "coefficient_code": "TEST",
                                       "coefficient_source_type": "TEST", "coefficient_version": "v1"})
    result = run(**{field: meta})
    assert inputs["status"] == "READY"
    assert result["daily_serving_g"] is None
    assert prefix + ("_MISSING" if value is None else "_INVALID") in result["reason_codes"]


def test_unsupported_state_and_unknown_coefficient():
    result = run(coefficient=None, supported_state=False)
    assert result["reason_codes"] == ["COEFFICIENT_MISSING", "PET_STATE_UNSUPPORTED"]
    assert result["daily_serving_g"] is None
    assert "PET_STATE_UNSUPPORTED" in run(species="both")["reason_codes"]


def test_provenance_is_required_and_calculation_is_deterministic():
    assert run() == run() == run()
    assert run(coefficient={"coefficient": 1.5})["reason_codes"] == ["COEFFICIENT_PROVENANCE_MISSING"]
    assert run(energy={"energy_density_kcal_per_kg": 3850, "energy_basis": "AS_FED"})["reason_codes"] == ["ENERGY_PROVENANCE_MISSING"]


def test_rounding_half_up_and_no_intermediate_rounding():
    assert _round(10.05) == 10.1
    assert _round(10.0499999) == 10.0
    result = run(weight_kg=1)
    assert result["rer_kcal_per_day"] == 70
    assert result["daily_serving_g"] == 27.3


def test_overflow_underflow_are_null_not_zero():
    for result in [run(weight_kg=1e-300), run(energy={"energy_density_kcal_per_kg": 1e-320,
                                                "energy_basis": "AS_FED", "energy_source_type": "TEST", "energy_version": "v1"})]:
        assert result["status"] == "INSUFFICIENT_DATA"
        assert result["daily_serving_g"] is None
        assert result["reason_codes"] == ["CALCULATION_INVALID"]


@pytest.mark.parametrize("form,kcal", [("DRY_FOOD", 3850), ("WET_FOOD", 750), ("FREEZE_DRY", None)])
def test_mock_energy_requires_explicit_supported_form(form, kcal):
    source = {"id": 1, "sku": "MOCK-0001", "category_code": "FOOD", "subcategory_code": form}
    provenance = {"service_product_id": "1", "service_sku": "MOCK-0001", "type": "MOCK_INTEGRATION_FIXTURE", "fixture_status": "FIXTURE_PARTIAL", "fixture_profile": "DOG_ADULT_DRY"}
    result = mock_energy(source, provenance)
    assert (result or {}).get("energy_density_kcal_per_kg") == kcal
    assert mock_energy(source, {**provenance, "fixture_status": "FIXTURE_UNAVAILABLE"}) is None
    assert mock_energy(source, {**provenance, "type": "MANUFACTURER"}) is None
