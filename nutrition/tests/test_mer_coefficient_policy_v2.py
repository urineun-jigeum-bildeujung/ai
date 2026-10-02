import sys
from datetime import date
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts/nutrition')]
import mer_coefficient_policy_v2 as policy


@pytest.mark.parametrize('species,months,neutered,expected', [
    ('DOG',3,None,3.0),('DOG',4,None,2.0),('DOG',11,None,2.0),
    ('DOG',12,True,1.6),('DOG',12,False,1.8),('DOG',84,True,1.6),
    ('CAT',11,None,2.5),('CAT',12,True,1.2),('CAT',12,False,1.4),('CAT',84,False,1.4)])
def test_month_boundaries_and_baseline_provenance(monkeypatch, species, months, neutered, expected):
    today = date(2026,10,3); monkeypatch.setattr(policy,'_today',lambda:today)
    year, month = divmod(today.year * 12 + today.month - 1 - months, 12)
    result = policy.resolve_mer_coefficient({'species':species,'birth_date':date(year,month+1,3),
                                          'is_neutered':neutered,'age':999})
    assert result['coefficient'] == expected and result['policy_status'] == 'RESOLVED'
    assert result['coefficient_source_type'] == 'MERCK_HEALTHY_BASELINE'
    assert result['coefficient_version'] == 'healthy_merck_v1'
    assert result['applicability'] == 'HEALTHY_BASELINE_STARTING_ESTIMATE'
    assert result['disclaimer_code'] == 'BASELINE_ESTIMATE_NOT_MEDICAL_PRESCRIPTION'


@pytest.mark.parametrize('changes,reason', [
    ({'is_neutered':None},'NEUTER_STATUS_MISSING_OR_INVALID'),
    ({'species':'BIRD'},'PET_SPECIES_UNSUPPORTED'),({'age':None},'PET_AGE_UNRESOLVED'),
    ({'age':True},'PET_AGE_UNRESOLVED'),({'age':float('nan')},'PET_AGE_UNRESOLVED'),
    ({'birth_date':'invalid','age':None},'PET_AGE_UNRESOLVED')])
def test_unresolved(changes, reason):
    result = policy.resolve_mer_coefficient({'species':'DOG','age':2,'is_neutered':True,**changes})
    assert result['coefficient'] is None and result['reason_codes'] == [reason]


def test_bcs_disease_and_request_factor_do_not_change_policy():
    results = [policy.resolve_mer_coefficient({'species':'DOG','age':2,'is_neutered':True,
               'bcs':bcs,'coefficient':10,'disease':'unknown'}) for bcs in (None,1,3,5,100)]
    assert all(r == results[0] for r in results) and results[0]['coefficient'] == 1.6
