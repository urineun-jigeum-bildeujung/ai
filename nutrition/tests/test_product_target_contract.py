"""Independent marketing/size contract and strict synthetic identity."""
import json
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/nutrition')]
import service_db_adapter as adapter
from product_target_contract import (RAW_LABELS, FIXTURE_PATH, parse_raw_target_label,
    load_synthetic_target, resolve_product_target, evaluate_target_compatibility)


def product(**changes):
    """Build a canonical MOCK product with overridable species, stage, and size targets."""
    return {'id': '1', 'service_sku': 'MOCK-0001', 'target_species': 'dog',
            'service_target_age_group': 'ADULT', 'service_target_breed_size': 'SMALL', **changes}


def fixture(**changes):
    """Build a synthetic target fixture matching the default MOCK product identity."""
    return {'service_product_id': 1, 'service_sku': 'MOCK-0001', 'target_species': ['DOG'],
            'target_stage': 'ADULT', 'target_sizes': ['SMALL'], **changes}


def pet_source(**changes):
    """Build a Service pet row for an adult small dog with no known allergies."""
    return {'id': 1, 'species': 'DOG', 'age': 2, 'weight': 8, 'allergies': [],
            'allergy_profile_status': 'KNOWN_NONE', 'target_breed_size': 'SMALL', **changes}


@pytest.mark.parametrize('label,expected', RAW_LABELS.items())
def test_all_twelve_exact_labels(label, expected):
    """Verify each supported label maps exactly after trimming outer whitespace."""
    assert parse_raw_target_label(label) == expected
    assert parse_raw_target_label(' \t' + label + '\n') == expected


@pytest.mark.parametrize('value', ['소형견 성체 사료', '소형견성체', '고양이  성체', '소형견 성묘', '', None])
def test_no_substring_fuzzy_or_inner_whitespace(value):
    """Verify approximate labels and altered internal spacing remain unresolved."""
    assert parse_raw_target_label(value)['target_stage'] == 'UNKNOWN'


def test_nfkc_and_no_shared_mutation():
    """Verify Unicode normalization and independent copies of mapped target lists."""
    raw = '\u1109\u1169형견 성체'
    assert parse_raw_target_label(raw)['target_stage'] == 'ADULT'
    result = parse_raw_target_label('소형견 성체')
    result['target_sizes'].append('LARGE')
    assert parse_raw_target_label('소형견 성체')['target_sizes'] == ['SMALL']


@pytest.mark.parametrize('stage', ['GROWTH', 'ADULT', 'SENIOR'])
def test_service_stages_are_not_aafco(stage):
    """Verify Service marketing stages resolve targets without supplying AAFCO evidence."""
    source = {'id': 1, 'sku': None, 'category_code': 'FOOD', 'target_species': ['DOG'],
              'target_age_group': stage, 'target_breed_size': 'SMALL'}
    adapted = adapter.adapt_product(source, index={})['product']
    assert adapted['aafco_life_stage'] is None
    assert resolve_product_target(adapted)['stage']['value'] == stage


@pytest.mark.parametrize('field,missing,fixture_value,expected', [
    ('service_target_age_group', None, 'ADULT', 'RESOLVED'),
    ('service_target_age_group', 'ADULT', 'ADULT', 'RESOLVED'),
    ('service_target_age_group', 'SENIOR', 'ADULT', 'CONFLICT'),
    ('service_target_breed_size', None, ['SMALL'], 'RESOLVED'),
    ('service_target_breed_size', 'SMALL', ['SMALL'], 'RESOLVED'),
    ('service_target_breed_size', 'LARGE', ['SMALL'], 'CONFLICT'),
])
def test_service_precedence_and_conflicts(field, missing, fixture_value, expected):
    """Verify Service target precedence, fixture fallback, and explicit conflict status."""
    ffield = 'target_stage' if field.endswith('age_group') else 'target_sizes'
    axis = 'stage' if field.endswith('age_group') else 'size'
    result = resolve_product_target(product(**{field: missing}), fixture(**{ffield: fixture_value}))
    assert result[axis]['status'] == expected
    assert result[axis]['source'] == ('SCHEMA_DRIVEN_SYNTHETIC_FIXTURE' if missing is None else 'SERVICE_STRUCTURED')


def test_missing_targets_are_unknown_not_all_life_stages():
    """Verify absent stage and size data remain unknown."""
    resolved = resolve_product_target(product(service_target_age_group=None, service_target_breed_size=None))
    assert resolved['stage']['value'] == 'UNKNOWN'
    assert resolved['size']['status'] == 'UNKNOWN'


@pytest.mark.parametrize('sku,id_', [('MOCK-OTHER', 1), ('MOCK-0001', 999), ('1234567890123', 1)])
def test_fixture_identity_cannot_be_bypassed(sku, id_):
    """Verify mismatched or non-MOCK fixture identities cannot supply target stages."""
    result = resolve_product_target(product(service_target_age_group=None), fixture(service_product_id=id_, service_sku=sku))
    assert result['stage']['value'] == 'UNKNOWN'


def test_artifact_identity_and_exact_labels(monkeypatch, tmp_path):
    """Verify fixture identity checks reject mismatches, missing files, malformed data, and duplicates."""
    import product_target_contract as contract
    artifact = json.loads(FIXTURE_PATH.read_text())
    for row in artifact['items']:
        loaded = load_synthetic_target({'id': row['service_product_id'], 'sku': row['service_sku']})
        assert loaded is not None
        assert loaded['production_evidence'] is False
        assert load_synthetic_target({'id': row['service_product_id'] + 10000, 'sku': row['service_sku']}) is None
    path = tmp_path / 'fixture.json'
    monkeypatch.setattr(contract, 'FIXTURE_PATH', path)
    assert load_synthetic_target({'id': 1, 'sku': 'MOCK-0001'}) is None
    path.write_text('[]')
    assert load_synthetic_target({'id': 1, 'sku': 'MOCK-0001'}) is None
    first = artifact['items'][0]
    artifact['items'].append(deepcopy(first))
    path.write_text(json.dumps(artifact))
    assert load_synthetic_target({'id': first['service_product_id'], 'sku': first['service_sku']}) is None


@pytest.mark.parametrize('months,expected', [(0,'GROWTH'),(3,'GROWTH'),(11,'GROWTH'),(12,'ADULT'),(83,'ADULT'),(84,'SENIOR'),(120,'SENIOR')])
def test_exact_birth_month_boundaries(monkeypatch, months, expected):
    """Verify calendar-month boundaries independently select marketing and reference stages."""
    today = date(2026, 10, 3)
    monkeypatch.setattr(adapter, '_today', lambda: today)
    year, month = divmod(today.year * 12 + today.month - 1 - months, 12)
    result = adapter.adapt_pet(pet_source(birth_date=date(year, month + 1, today.day)))
    assert result['product_target_stage'] == expected
    assert result['life_stage'] == ('GROWTH_REPRODUCTION' if months < 12 else 'ADULT_MAINTENANCE')


@pytest.mark.parametrize('age,expected', [(0,'GROWTH'),(.99,'GROWTH'),(1,'ADULT'),(6.99,'ADULT'),(7,'SENIOR')])
def test_legacy_age_only_for_marketing(age, expected):
    """Verify numeric age selects a marketing stage while the reference stage stays unknown."""
    result = adapter.adapt_pet(pet_source(age=age))
    assert result['product_target_stage'] == expected
    assert result['life_stage'] == 'UNKNOWN'


@pytest.mark.parametrize('kind,value,reason', [('pet','TOY','PET_TARGET_BREED_SIZE_INVALID'),
    ('product','TOY','PRODUCT_TARGET_BREED_SIZE_INVALID'),('stage','ALL_LIFE_STAGES','PRODUCT_TARGET_AGE_GROUP_INVALID')])
def test_source_enum_invalid(kind, value, reason):
    """Verify invalid Service stage or size enums raise the corresponding input error."""
    with pytest.raises(adapter.ServiceInputError, match=reason):
        if kind == 'pet': adapter.adapt_pet(pet_source(target_breed_size=value))
        else: adapter.adapt_product({'id':1, 'category_code':'FOOD',
            'target_age_group':value if kind == 'stage' else None,
            'target_breed_size':value if kind == 'product' else None}, index={})


@pytest.mark.parametrize('species,size,stage,expected', [
    ('dog','SMALL','ADULT','MATCHED'),('dog','LARGE','ADULT','MISMATCH'),
    ('dog',None,'ADULT','UNKNOWN'),('dog','SMALL','GROWTH','MISMATCH'),
    ('cat',None,'ADULT','MATCHED')])
def test_compatibility_axes(species, size, stage, expected):
    """Verify species, stage, and size compatibility, including inapplicable cat size."""
    target = resolve_product_target(product(target_species=species))
    result = evaluate_target_compatibility({'species':species, 'service_target_breed_size':size, 'product_target_stage':stage}, target)
    assert result['status'] == expected
    if species == 'cat': assert result['size']['status'] == 'NOT_APPLICABLE'


def test_all_life_stages_and_conflict_priority():
    """Verify all-life-stage targets match seniors and conflicts outrank mismatches."""
    target = resolve_product_target(product(service_target_age_group=None), fixture(target_stage='ALL_LIFE_STAGES'))
    assert evaluate_target_compatibility({'species':'dog','service_target_breed_size':'SMALL','product_target_stage':'SENIOR'}, target)['status'] == 'MATCHED'
    target = resolve_product_target(product(), fixture(target_stage='SENIOR'))
    assert evaluate_target_compatibility({'species':'cat','product_target_stage':'GROWTH'}, target)['status'] == 'CONFLICT'


@pytest.mark.parametrize('value', ['UNSUPPORTED', 42, ['ADULT'], {'stage':'ADULT'}])
def test_unsupported_stage_value_fail_closes(value):
    """Verify invalid stages stay unknown or conflict with otherwise valid fixture evidence."""
    result = resolve_product_target(product(service_target_age_group=value))
    assert result['stage']['status'] == 'UNKNOWN'
    result = resolve_product_target(product(service_target_age_group=value),fixture())
    assert result['stage']['status'] == 'CONFLICT'


def test_species_conflict_preserves_authoritative_service_species():
    """Verify a fixture species conflict retains the Service species targets."""
    result = resolve_product_target(product(),fixture(target_species=['CAT']))
    assert result['species']['targets'] == ['DOG']
    assert result['species']['status'] == 'CONFLICT'


def test_target_fixture_never_removes_historical_adequacy_gap():
    """Verify synthetic target stages cannot fill missing AAFCO evidence or unlock scoring."""
    import api_nutrition as api
    artifact = json.loads(FIXTURE_PATH.read_text())
    row = next(r for r in artifact['items'] if r['target_species'] == ['CAT'])
    source = {'id':row['service_product_id'],'sku':row['service_sku'],'category_code':'FOOD',
              'subcategory_code':'DRY_FOOD','target_species':['CAT'],'target_age_group':None,
              'ingredient_codes':[],'allergen_flags':[],'caution_codes':[]}
    p = {'id':1,'species':'CAT','age':2,'weight':4,'allergies':[],
         'allergy_profile_status':'KNOWN_NONE','is_neutered':True}
    result = api.analyze_service_records(p,source)
    assert result['input_provenance']['product_target']['stage']['source'] == 'SCHEMA_DRIVEN_SYNTHETIC_FIXTURE'
    assert result['input_provenance']['aafco_life_stage_evidence_status'] == 'UNKNOWN'
    assert 'PRODUCT_LIFE_STAGE_UNKNOWN' in result['safety_reason_codes']
    assert result['suitability']['match_score'] is None
