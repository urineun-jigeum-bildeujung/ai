import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts/nutrition')]
from presentation import build_nutrition_presentation


def item(**changes):
    return {'nutrient_code':'CRUDE_PROTEIN','value':.26,'unit':'PERCENT','basis':'AS_FED',
            'aligned_value':26, 'basis_normalized_value':28.9, 'basis_normalized_basis':'DRY_MATTER',
            'nias_min':18, 'nias_max':50, 'safe_upper_unit':'g/100g DM',
            'nias_compare_status':'IN_RANGE','canonical_status':'KNOWN',
            'source':'MOCK_INTEGRATION_FIXTURE:mock_nutrition_fixture_v1', **changes}


def test_actual_values_bounds_and_provenance():
    row = build_nutrition_presentation([item()], 'TRUE')['rows'][0]
    assert row['value'] == .26 and row['normalized_value'] == 28.9
    assert row['basis'] == 'AS_FED' and row['normalized_basis'] == 'DRY_MATTER'
    assert row['reference_min'] == 18 and row['reference_max'] == 50
    assert row['reference_unit'] == 'g/100g DM' and row['canonical_status'] == 'KNOWN'
    assert row['source_type'] == 'MOCK_INTEGRATION_FIXTURE' and row['production_evidence'] is False
    assert not {'position','properRange','bar_width','color','functions'} & row.keys()


def test_normalized_priority_no_ref_out_of_range_and_counts():
    rows = [item(basis_normalized_value=None), item(nutrient_code='PHOSPHORUS',nias_compare_status='NO_REF'),
            item(nutrient_code='CALCIUM',nias_compare_status='OUT_OF_RANGE'),
            item(nutrient_code='MOISTURE',nias_compare_status='NO_VALUE',basis_invalid=True)]
    result = build_nutrition_presentation(rows, 'UNKNOWN')
    by_code = {r['nutrient_code']:r for r in result['rows']}
    assert by_code['CRUDE_PROTEIN']['normalized_value'] == 26
    assert by_code['PHOSPHORUS']['reference_min'] is None
    assert by_code['PHOSPHORUS']['reference_max'] is None
    assert by_code['CALCIUM']['compare_status'] == 'OUT_OF_RANGE'
    assert by_code['MOISTURE']['normalized_value'] is None
    assert result['summary'] == {'total_count':4,'comparable_count':2,'in_range_count':1,'out_of_range_count':1,'unknown_count':2}
    assert result == build_nutrition_presentation(list(reversed(rows)), 'UNKNOWN')


def test_fallback_value_and_display_name():
    row = build_nutrition_presentation([{'nutrient_code':'NEW_CODE','value':42,'nias_compare_status':'NO_REF'}], 'UNKNOWN')['rows'][0]
    assert row['display_name'] == 'NEW_CODE' and row['normalized_value'] == 42


def test_selected_nias_unit_uses_curated_metadata_not_legacy_upper_unit():
    rows = [item(nutrient_code='VITAMIN_B1',reference_selection_status='SELECTED',safe_upper_unit='g/100g DM'),
            item(nutrient_code='MOISTURE',reference_selection_status='SELECTED',
                 basis='AS_FED',basis_normalized_basis='AS_FED',safe_upper_unit='g/100g DM')]
    by_code = {r['nutrient_code']:r for r in build_nutrition_presentation(rows,'TRUE')['rows']}
    assert by_code['VITAMIN_B1']['reference_unit'] == 'mg/100g DM'
    assert by_code['MOISTURE']['reference_unit'] == 'PERCENT'
