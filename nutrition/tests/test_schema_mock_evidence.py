"""Mock-only evidence completion preserves conflicts and unknown input gates."""
from datetime import date
import json

import pytest
from fastapi.testclient import TestClient
from test_service_profile_birth_policy import pet, product
import api_nutrition as api
import service_db_adapter as adapter
import mock_integration_fixture as fixture
from mock_integration_fixture import build_mock_fixture, is_mock_source
from mock_feeding_fixture import mock_energy


def fixture_product(product_id):
    if product_id == 141:
        return product(id=141, sku='MOCK-FOOD-WET_FOOD-0021', subcategory_code='WET_FOOD',
                       ingredient_codes=['양고기', '귀리', '고구마', '비트', '당근'],
                       allergen_flags=['LAMB', 'OAT_BARLEY', 'SWEET_POTATO'])
    return product(id=302, sku='ONF-004', ingredient_codes=['닭고기', '현미', '호박', '연어오일'],
                   allergen_flags=['CHICKEN', 'RICE', 'SALMON'])


@pytest.mark.parametrize('product_id,allergies,status', [
    (141, [], 'NOT_APPLICABLE'), (141, ['CHICKEN'], 'NO_CONFLICT_DETECTED'),
    (141, ['LAMB'], 'SAFETY_BLOCKED'), (302, [], 'NOT_APPLICABLE'),
    (302, ['LAMB'], 'NO_CONFLICT_DETECTED'), (302, ['CHICKEN'], 'SAFETY_BLOCKED'),
    (302, ['SALMON'], 'SAFETY_BLOCKED'), (302, ['FISH'], 'SAFETY_BLOCKED'),
    (302, ['OTHER'], 'SAFETY_DATA_INSUFFICIENT'),
])
def test_schema_mock_cases(product_id, allergies, status):
    source = fixture_product(product_id)
    result = api.analyze_service_records(pet(birth_date=date(2023, 1, 1), allergies=allergies,
        allergy_profile_status='KNOWN_LIST' if allergies else 'KNOWN_NONE'), source)
    assert result['safety_status'] == status
    loaded = adapter.adapt_product(source)
    assert len(loaded['product']['nutrition_items']) == 5
    assert loaded['product']['aafco_life_stage'] == 'ADULT'
    assert loaded['provenance']['nutrition_source']['production_evidence'] is False
    assert result['input_provenance']['integration_data']['production_evidence'] is False
    assert mock_energy(source, loaded['provenance']['nutrition_source']) is not None
    refs = loaded['product']['product_allergen_refs']
    synthetic = [r for r in refs if r['raw_text'] in {'당근', '비트', '호박', '연어오일'}]
    assert synthetic and all(r['production_evidence'] is False for r in synthetic)


@pytest.mark.parametrize('changes', [{'id': 303}, {'sku': 'ONF-999'}, {'sku': '036000291452'}])
def test_registration_requires_exact_id_and_sku(changes):
    source = {**fixture_product(302), **changes}
    assert not is_mock_source(source)
    assert build_mock_fixture(source) is None


def test_nonmock_refs_and_unknown_mock_ingredients_remain_unresolved():
    assert all(r['mapping_method'] == 'UNRESOLVED'
               for r in adapter.structured_refs('302', ['당근', '비트', '호박', '연어오일']))
    source = fixture_product(302)
    source['ingredient_codes'].append('unknown-new-ingredient')
    result = api.analyze_service_records(pet(birth_date=date(2023, 1, 1), allergies=['LAMB'],
        allergy_profile_status='KNOWN_LIST'), source)
    assert result['safety_status'] == 'SAFETY_DATA_INSUFFICIENT'
    assert 'UNMAPPED_INGREDIENT' in result['safety_reason_codes']


def test_mock_oil_conflicts_without_service_flags():
    source = fixture_product(302)
    source['allergen_flags'] = []
    for code in ['SALMON', 'FISH']:
        result = api.analyze_service_records(pet(birth_date=date(2023, 1, 1), allergies=[code],
            allergy_profile_status='KNOWN_LIST'), source)
        assert result['safety_status'] == 'SAFETY_BLOCKED'
        assert 'ALLERGY_CONFLICT' in result['safety_reason_codes']


def test_mock_identity_is_deterministic():
    assert build_mock_fixture(fixture_product(302)) == build_mock_fixture(fixture_product(302))


def test_mock_unknown_profile_and_toxic_gate_are_preserved():
    source = fixture_product(302)
    result = api.analyze_service_records(pet(birth_date=date(2023, 1, 1),
        allergy_profile_status='UNKNOWN'), source)
    assert 'ALLERGY_PROFILE_UNKNOWN' in result['safety_reason_codes']
    source['caution_codes'] = ['ONION']
    result = api.analyze_service_records(pet(birth_date=date(2023, 1, 1)), source)
    assert result['safety_status'] == 'SAFETY_BLOCKED'
    assert 'TOXIC_INGREDIENT' in result['safety_reason_codes']


@pytest.fixture
def ingredient_artifact(monkeypatch, tmp_path):
    fixture._ingredient_identity_payload.cache_clear()
    path = tmp_path / 'mock_ingredient_identity_v1.json'
    monkeypatch.setattr(fixture, 'INGREDIENT_IDENTITY_PATH', path)
    yield path
    fixture._ingredient_identity_payload.cache_clear()


def ingredient_payload(**changes):
    return {'fixture_version': 'mock_ingredient_identity_v1',
            'data_generation_type': 'SCHEMA_DRIVEN_SYNTHETIC',
            'production_evidence': False, 'mappings': {'당근': ['carrot']}, **changes}


@pytest.mark.parametrize('content', [
    None, b'{broken', b'\xff', b'null', b'[]',
    json.dumps(ingredient_payload(mappings=[])).encode(),
    json.dumps(ingredient_payload(mappings={'당근': 'carrot'})).encode(),
    json.dumps(ingredient_payload(mappings={'당근': [None]})).encode(),
    json.dumps(ingredient_payload(production_evidence=True)).encode(),
    json.dumps(ingredient_payload(fixture_version='invalid')).encode(),
])
def test_invalid_ingredient_artifact_preserves_unresolved_refs(ingredient_artifact, content, caplog):
    if content is not None:
        ingredient_artifact.write_bytes(content)
    refs = adapter.structured_refs('302', ['당근'])
    assert fixture.mock_ingredient_refs(refs) is refs
    assert refs[0]['mapping_method'] == 'UNRESOLVED'
    assert 'skipping enrichment' in caplog.text
    assert fixture._ingredient_identity_payload.cache_info().currsize == 0


def test_valid_ingredient_artifact_is_cached_without_mutating_refs(ingredient_artifact):
    ingredient_artifact.write_text(json.dumps(ingredient_payload()), encoding='utf-8')
    refs = adapter.structured_refs('302', ['당근'])
    enriched = fixture.mock_ingredient_refs(refs)
    ingredient_artifact.unlink()
    assert fixture.mock_ingredient_refs(refs) == enriched
    assert fixture._ingredient_identity_payload.cache_info().hits == 1
    assert refs[0]['mapping_method'] == 'UNRESOLVED'
    assert enriched[0]['allergen_code'] == 'carrot'
    assert enriched[0]['production_evidence'] is False


def test_failed_ingredient_artifact_load_is_retried(ingredient_artifact):
    refs = adapter.structured_refs('302', ['당근'])
    assert fixture.mock_ingredient_refs(refs) is refs
    ingredient_artifact.write_text(json.dumps(ingredient_payload()), encoding='utf-8')
    assert fixture.mock_ingredient_refs(refs)[0]['allergen_code'] == 'carrot'


def test_missing_ingredient_artifact_http_keeps_safety_fail_close(ingredient_artifact, monkeypatch):
    for key in ('MEMBER_DATABASE_URL', 'PRODUCT_DATABASE_URL', 'INTERNAL_GATEWAY_SECRET'):
        monkeypatch.setenv(key, 'synthetic-test-value')
    monkeypatch.setattr(api.service_repository, 'get_pet', lambda *_: pet(
        birth_date=date(2023, 1, 1), allergies=['LAMB'], allergy_profile_status='KNOWN_LIST'))
    monkeypatch.setattr(api.service_repository, 'get_product', lambda _: fixture_product(302))
    with TestClient(api.app) as client:
        response = client.post('/api/nutrition/analyze/by-service-id',
                               json={'pet_id': 123, 'product_id': 302},
                               headers={'X-Internal-Secret': 'synthetic-test-value', 'X-Member-Id': '42'})
    assert response.status_code == 200
    result = response.json()
    assert result['safety_status'] == 'SAFETY_DATA_INSUFFICIENT'
    assert 'UNMAPPED_INGREDIENT' in result['safety_reason_codes']
    assert result['feeding']['daily_serving_g'] is None
