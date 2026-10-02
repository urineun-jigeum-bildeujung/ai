import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts/nutrition')]
from suitability import build_suitability


def comparison(passes=5):
    """Build a complete five-nutrient comparison with a chosen number of passing values."""
    return {'nutrition_comparison_status':'TRUE' if passes == 5 else 'FALSE',
            'essential_count':5,'comparable_count':5,'pass_count':passes,'fail_count':5-passes,
            'essential_status':{str(i):'IN_RANGE' if i < passes else 'OUT_OF_RANGE' for i in range(5)}}


def score(c=None, target='MATCHED', safety='NOT_APPLICABLE', excluded=False):
    """Build suitability for a comparison with configurable target and safety gates."""
    return build_suitability(nutrition_comparison=c or comparison(),
        target_compatibility={'status':target}, safety_status=safety, excluded=excluded)


@pytest.mark.parametrize('passes,expected', [(5,100),(4,80),(3,60),(0,0)])
def test_objective_ratio(passes, expected):
    """Verify deterministic integer pass percentages and reasons for failing nutrients."""
    result = score(comparison(passes))
    assert type(result['match_score']) is int and result['match_score'] == expected
    assert score(comparison(passes)) == result
    assert len(result['reason_codes']) == 5-passes


@pytest.mark.parametrize('status', ['MISSING','NON_COMPARABLE','DATA_CONFLICT'])
def test_incomplete_or_conflicting_required_never_scores(status):
    """Verify missing, noncomparable, or conflicting required nutrients suppress the score."""
    c = comparison(); c['essential_status']['0'] = status; c['comparable_count'] = 4
    assert score(c)['match_score'] is None
    assert f'NUTRIENT_{status}:0' in score(c)['reason_codes']


@pytest.mark.parametrize('target,safety,excluded,status', [
    ('MATCHED','SAFETY_BLOCKED',False,'BLOCKED'),
    ('MATCHED','SAFETY_DATA_INSUFFICIENT',False,'INSUFFICIENT_DATA'),
    ('MATCHED','NOT_APPLICABLE',True,'BLOCKED'),
    ('MISMATCH','NOT_APPLICABLE',False,'TARGET_MISMATCH'),
    ('CONFLICT','NOT_APPLICABLE',False,'TARGET_CONFLICT'),
    ('UNKNOWN','NOT_APPLICABLE',False,'INSUFFICIENT_DATA')])
def test_gates(target, safety, excluded, status):
    """Verify safety and target gates return the expected status with a null score."""
    result = score(target=target,safety=safety,excluded=excluded)
    assert result['match_score'] is None and result['status'] == status


def test_unknown_or_empty_matrix_never_zero():
    """Verify unknown or empty nutrient matrices yield null instead of a zero score."""
    c = comparison(); c['nutrition_comparison_status'] = 'UNKNOWN'
    assert score(c)['match_score'] is None
    assert score({'nutrition_comparison_status':'TRUE','essential_count':0})['match_score'] is None
