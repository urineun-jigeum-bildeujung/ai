"""Nutrition-owned energy lookup, separate from existing nutrient provenance."""
import json
from pathlib import Path

from mock_integration_fixture import is_mock_sku

ROOT = Path(__file__).resolve().parents[2] / "data" / "integration"


def mock_energy(source, nutrition_provenance):
    """Only a resolved Service Mock food fixture can receive Mock energy.

Read the small versioned artifact per call: no shared mutable data, and a
missing/corrupt artifact fails closed instead of changing Nutrition/Safety.
"""
    if (not is_mock_sku(source.get("sku"))
            or str(nutrition_provenance.get("service_product_id")) != str(source.get("id"))
            or nutrition_provenance.get("service_sku") != source.get("sku")
            or nutrition_provenance.get("type") != "MOCK_INTEGRATION_FIXTURE"
            or nutrition_provenance.get("fixture_status") not in {"FIXTURE_READY", "FIXTURE_PARTIAL"}
            or not nutrition_provenance.get("fixture_profile")
            or source.get("category_code") != "FOOD"):
        return None
    try:
        artifact = json.loads((ROOT / "mock_energy_v1.json").read_text())
        if (artifact.get("energy_version") != "mock_energy_v1"
                or artifact.get("energy_source_type") != "MOCK_INTEGRATION_FIXTURE"
                or artifact.get("energy_basis") != "AS_FED"):
            return None
        form = artifact["forms"].get(source.get("subcategory_code"))
        if not form:
            return None
        return {"energy_density_kcal_per_kg": form["energy_density_kcal_per_kg"],
                "energy_source_type": artifact["energy_source_type"],
                "energy_version": artifact["energy_version"],
                "energy_basis": artifact["energy_basis"],
                "data_generation_type": "SCHEMA_DRIVEN_SYNTHETIC",
                "production_evidence": False}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
