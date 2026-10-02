"""Authenticated Service-ID compare: validation, source reads and domain statuses."""
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/nutrition')]
import api_nutrition as api
from service_compare import compare_service_analyses
from test_feeding_service_contract import pet, product

PATH = '/api/nutrition/compare'
BODY = {'pet_id':10,'product_ids':[1,2],'allergy_profile_status':'KNOWN_NONE'}
HEADERS = {'X-Internal-Secret':'compare-test-only','X-Member-Id':'42'}


@pytest.fixture
def configured(monkeypatch):
    """Configure Gateway authentication and return mocked pet and product readers."""
    monkeypatch.setattr(api.service_repository,'configured',lambda:True)
    monkeypatch.setenv('INTERNAL_GATEWAY_SECRET',HEADERS['X-Internal-Secret'])
    get_pet = MagicMock(return_value=pet(target_breed_size='SMALL'))
    get_product = MagicMock(side_effect=lambda pid:product(id=pid,sku=f'MOCK-{pid:04d}',target_breed_size='SMALL',feeding_target='표시 정보',feeding_method='원문 유지'))
    monkeypatch.setattr(api.service_repository,'get_pet',get_pet)
    monkeypatch.setattr(api.service_repository,'get_product',get_product)
    return get_pet, get_product


@pytest.mark.parametrize('changes', [
    {'product_ids':[1]}, {'product_ids':[1,2,3]}, {'product_ids':[1,1]},
    {'product_ids':[0,2]}, {'product_ids':[-1,2]}, {'product_ids':['1',2]},
    {'product_ids':[True,2]}, {'product_ids':[1.0,2]}, {'product_ids':None},
    {'pet_id':0}, {'member_id':42}])
def test_invalid_requests_never_read_db(configured, changes):
    """Verify invalid comparison requests return 422 before any source reads."""
    with TestClient(api.app) as client:
        response = client.post(PATH,json={**BODY,**changes},headers=HEADERS)
    assert response.status_code == 422
    for mock in configured: mock.assert_not_called()
    if changes == {'product_ids':[1,1]}:
        assert 'COMPARE_PRODUCT_IDS_MUST_BE_DISTINCT' in response.text


@pytest.mark.parametrize('headers', [{}, {'X-Internal-Secret':'wrong','X-Member-Id':'42'},
    {'X-Internal-Secret':'compare-test-only'}, {**HEADERS,'X-Member-Id':'0'},
    [*HEADERS.items(), ('X-Member-Id','43')]])
def test_auth_before_source_reads(configured, headers):
    """Verify missing, invalid, or duplicate authentication headers prevent source reads."""
    with TestClient(api.app) as client:
        response = client.post(PATH,json=BODY,headers=headers)
    assert response.status_code == 401
    for mock in configured: mock.assert_not_called()


def test_configuration_semantics(configured, monkeypatch):
    """Verify absent authentication or source configuration returns the specific 503 error."""
    with TestClient(api.app) as client:
        monkeypatch.delenv('INTERNAL_GATEWAY_SECRET')
        response = client.post(PATH,json=BODY)
        assert response.status_code == 503 and response.json()['detail'] == 'SERVICE_AUTH_NOT_CONFIGURED'
        monkeypatch.setattr(api.service_repository,'configured',lambda:False)
        response = client.post(PATH,json=BODY)
        assert response.status_code == 503 and response.json()['detail'] == 'SERVICE_SOURCE_NOT_CONFIGURED'


@pytest.mark.parametrize('where,error,status,detail', [
    ('pet',api.service_repository.ServiceNotFound('PET_NOT_FOUND'),404,'PET_NOT_FOUND'),
    ('product',api.service_repository.ServiceNotFound('PRODUCT_NOT_FOUND'),404,'PRODUCT_NOT_FOUND'),
    ('product',api.service_repository.ServiceUnavailable('private-test-dsn'),503,'SERVICE_DB_UNAVAILABLE')])
def test_source_errors_preserved(configured, where, error, status, detail):
    """Verify source errors retain public status codes while hiding private details."""
    configured[0 if where == 'pet' else 1].side_effect = error
    with TestClient(api.app) as client:
        response = client.post(PATH,json=BODY,headers=HEADERS)
    assert response.status_code == status and response.json() == {'detail':detail}
    assert 'private-test-dsn' not in response.text
    if where == 'pet': configured[1].assert_not_called()


def test_invalid_source_sanitized(configured):
    """Verify malformed pet data produces a sanitized source-validation error."""
    configured[0].return_value = {}
    with TestClient(api.app) as client:
        response = client.post(PATH,json=BODY,headers=HEADERS)
    assert response.status_code == 422 and response.json() == {'detail':'SERVICE_SOURCE_INVALID'}


def test_real_engine_reuse_pet_once_products_separate_and_tie(configured, monkeypatch):
    """Verify one pet read feeds two engine analyses with tied scores and synthetic provenance."""
    real = api.analyze_service_records
    calls = []
    def capture(p, q):
        """Record independent input snapshots before delegating to the real analysis engine."""
        calls.append(deepcopy((p, q)))
        return real(p, q)
    monkeypatch.setattr(api,'analyze_service_records',capture)
    with TestClient(api.app) as client:
        response = client.post(PATH,json=BODY,headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert result['comparison_status'] == 'READY'
    assert result['comparison']['score_comparable'] is True
    assert result['comparison']['higher_match_score_product_id'] is None
    assert result['comparison']['match_score_delta'] == 0
    assert result['comparison']['nutrient_differences']
    configured[0].assert_called_once_with(10,42)
    assert [c.args for c in configured[1].call_args_list] == [(1,), (2,)]
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert calls[0][0]['allergy_profile_status'] == 'KNOWN_NONE'
    for item in result['products']:
        assert item['suitability']['match_score'] == 100
        assert item['target_compatibility']['status'] == 'MATCHED'
        assert item['feeding']['status'] == 'READY'
        assert item['feeding']['daily_serving_g'] == 138.4
        assert item['feeding']['production_evidence'] is False
        assert item['product_label']['feeding_method'] == '원문 유지'
    assert not {'BEST','HEALTHIEST','better','winner'} & result['comparison'].keys()


def analysis(score=100, **changes):
    """Build an analysis with a configurable score and one overridable nutrient row."""
    return {'product_id':'1','suitability':{'match_score':score},
            'presentation':{'rows':[{'nutrient_code':'CRUDE_PROTEIN','display_name':'조단백질',
                'normalized_value':26.0,'normalized_basis':'DRY_MATTER','unit':'PERCENT',
                'reference_unit':'g/100g DM','compare_status':'IN_RANGE',**changes}]}}


@pytest.mark.parametrize('scores,expected,higher,delta', [
    ([100,80],'READY',1,20),([60,80],'READY',2,20),([100,100],'READY',None,0),
    ([None,80],'PARTIAL',None,None),([None,None],'PARTIAL',None,None),
    ([True,80],'PARTIAL',None,None)])
def test_status_and_score_logic(scores, expected, higher, delta):
    """Verify comparison status, higher score, and deltas for valid and unavailable scores."""
    result = compare_service_analyses(10,[1,2],[analysis(scores[0]),analysis(scores[1],normalized_value=30.0)])
    assert result['comparison_status'] == expected
    assert result['comparison']['higher_match_score_product_id'] == higher
    assert result['comparison']['match_score_delta'] == delta
    assert result['comparison']['nutrient_differences'][0]['absolute_delta'] == 4.0


@pytest.mark.parametrize('changes', [
    {'normalized_basis':'AS_FED'}, {'reference_unit':'mg/100g DM'},
    {'unit':'mg/100g'}, {'normalized_value':None}, {'normalized_value':True},
    {'reference_unit':None}])
def test_no_false_comparison_across_basis_or_units(changes):
    """Verify incompatible bases, units, or invalid values suppress nutrient differences."""
    result = compare_service_analyses(10,[1,2],[analysis(None),analysis(None,**changes)])
    assert result['comparison_status'] == 'UNAVAILABLE'
    assert result['comparison']['nutrient_differences'] == []


def test_duplicate_nutrient_evidence_not_selected():
    """Verify duplicate evidence for a nutrient prevents comparison of that nutrient."""
    duplicate = analysis(None)
    duplicate['presentation']['rows'] *= 2
    result = compare_service_analyses(10,[1,2],[duplicate,analysis(None)])
    assert result['comparison_status'] == 'UNAVAILABLE'


def test_compare_partial_with_missing_target_and_blocked_feeding(configured):
    """Verify missing targets allow partial comparison and allergy exclusions block feeding."""
    configured[0].return_value['target_breed_size'] = None
    with TestClient(api.app) as client:
        response = client.post(PATH,json=BODY,headers=HEADERS)
        assert response.json()['comparison_status'] == 'PARTIAL'
        configured[0].return_value['allergies'] = ['CHICKEN']
        configured[1].side_effect = lambda pid:product(id=pid,allergen_flags=['CHICKEN'])
        response = client.post(PATH,json={**BODY,'allergy_profile_status':'KNOWN_LIST'},headers=HEADERS)
    assert response.status_code == 200
    assert all(p['feeding']['status'] == 'BLOCKED' and p['feeding']['mer_kcal_per_day'] is None
               for p in response.json()['products'])
