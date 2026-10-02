"""Offline archived parity and real loopback HTTP validation; no DB writes.

The HTTP Service source boundary is an explicit schema-driven synthetic test
fixture with saved Service product IDs. It is never a live Gateway/DB test.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/nutrition')]
import api_nutrition as api


def legacy_projection(result):
    """Copy an analysis and remove additive fields excluded from archived parity checks."""
    result = deepcopy(result)
    for key in ('feeding', 'product_allergen_refs', 'target_compatibility', 'presentation', 'suitability', 'product_label'):
        result.pop(key, None)
    provenance = result.get('input_provenance', {})
    for key in ('product_target', 'integration_data'):
        provenance.pop(key, None)
    source = provenance.get('nutrition_source') or {}
    for key in ('data_generation_type', 'production_evidence', 'schema_contract'):
        source.pop(key, None)
    return result


def replay(products, baseline):
    """Verify archived hashes for 286 MOCK products and exact target fixture identities.

    Return fixture counts and parity metadata; raise AssertionError on drift.
    """
    pet = {'id': 10, 'species': 'DOG', 'age': 2, 'weight': 8, 'allergies': [],
           'allergy_profile_status': 'KNOWN_NONE', 'life_stage': None}
    statuses = Counter()
    passed = 0
    mock_products = [p for p in products if p['sku'].startswith('MOCK-')]
    for product in mock_products:
        result = api.analyze_service_records(pet, product)
        digest = hashlib.sha256(json.dumps(legacy_projection(result), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert digest == baseline['legacy_hashes'][str(product['id'])], 'LEGACY_PROJECTION_CHANGED'
        passed += 1
        statuses[result['input_provenance']['fixture_status']] += 1
    assert passed == len(mock_products) == 286
    assert statuses == {'FIXTURE_READY': 107, 'FIXTURE_PARTIAL': 13, 'FIXTURE_UNAVAILABLE': 166}
    artifact = json.loads((ROOT / 'data/integration/mock_product_target_v1.json').read_text())
    identities = {(p['id'], p['sku']) for p in mock_products}
    assert all((r['service_product_id'], r['service_sku']) in identities for r in artifact['items'])
    return {'scope': 'LOCAL_ARCHIVED_SELECT_REPLAY_NOT_LIVE_E2E', 'legacy_passed': passed, 'legacy_total': len(mock_products),
            'fixture_counts': dict(statuses), 'target_fixture_verified_identities': len(artifact['items']),
            'projection': 'Existing legacy projection excludes feeding/product_allergen_refs; removes only new response fields and new provenance keys.'}


def runtime(products):
    """Run loopback HTTP contract checks against synthetic Service repository reads.

    Return assertion counts and observed outcomes, restoring repository getters
    and environment variables and stopping the server when checks finish.
    """
    import uvicorn
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(api.app, log_level='critical', lifespan='off'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    checks = 0
    requests = 0

    def check(condition):
        """Count one HTTP contract assertion and fail with a fixed message when false."""
        nonlocal checks
        checks += 1
        assert condition, 'HTTP_DOMAIN_ASSERTION_FAILED'

    def request(path, body=None, authenticated=True):
        """Count and send a loopback request, decoding success bodies or HTTP error details."""
        nonlocal requests
        requests += 1
        headers = {'Content-Type': 'application/json'}
        if authenticated:
            headers.update({'X-Internal-Secret': 'local-contract-v3-test-only', 'X-Member-Id': '1'})
        data = None if body is None else json.dumps(body).encode()
        req = Request(f'http://127.0.0.1:{port}' + path, data=data, headers=headers)
        try:
            with urlopen(req, timeout=5) as response:
                content = response.read()
                check(response.status == 200)
                return content.decode() if path == '/metrics' else json.loads(content)
        except HTTPError as error:
            return {'http_status': error.code, 'body': json.loads(error.read())}

    keys = ('MEMBER_DATABASE_URL', 'PRODUCT_DATABASE_URL', 'INTERNAL_GATEWAY_SECRET', 'NUTRITION_RUNTIME_MODE')
    environment = {key: os.environ.get(key) for key in keys}
    original_getters = api.service_repository.get_pet, api.service_repository.get_product
    try:
        for key in keys:
            os.environ.pop(key, None)
        thread.start()
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.02)
        check(server.started)
        health = request('/health', authenticated=False)
        check('POST /api/nutrition/compare' in health['endpoints'])
        check(health['future_endpoints'] == [])
        ready = request('/ready', authenticated=False)
        check(ready['status'] == 'ready' and ready['runtime_mode'] == 'local_artifact')
        check(all(d['status'] == 'UP' for d in ready['dependencies'].values() if d['required']))
        check('nutrition_http_requests_total' in request('/metrics', authenticated=False))
        candidates = [p for p in products if p['sku'].startswith('MOCK-')
                      and p['category_code'] == 'FOOD' and p['target_age_group'] == 'ADULT'
                      and p['target_species'] == ['DOG'] and p['subcategory_code'] in {'DRY_FOOD', 'WET_FOOD'}
                      and p['id'] % 29 and p['id'] % 11 and not p['caution_codes']]
        first, second = candidates[:2]
        # Synthetic source fields are explicitly supplied from supported schema
        # enums. Saved product identity and its existing nutrient fixture stay exact.
        records = {p['id']: {**p, 'target_breed_size': 'SMALL', 'feeding_target': '성체',
                            'feeding_method': '합성 통합 검증 표시 정보'} for p in (first, second)}
        pet = {'id': 1, 'species': 'DOG', 'birth_date': '2024-10-03', 'age': 2, 'weight': 8,
               'target_breed_size': 'SMALL', 'is_neutered': True, 'bcs': 3, 'allergies': []}
        os.environ['MEMBER_DATABASE_URL'] = os.environ['PRODUCT_DATABASE_URL'] = 'schema-driven-test-source'
        os.environ['INTERNAL_GATEWAY_SECRET'] = 'local-contract-v3-test-only'
        api.service_repository.get_pet = lambda pid, mid: deepcopy(pet)
        api.service_repository.get_product = lambda pid: deepcopy(records[pid])
        body = {'pet_id': 1, 'product_id': first['id'], 'allergy_profile_status': 'KNOWN_NONE'}
        analysis = request('/api/nutrition/analyze/by-service-id', body)
        check(analysis['target_compatibility']['status'] == 'MATCHED')
        check(analysis['target_compatibility']['size']['status'] == 'MATCHED')
        check(analysis['suitability']['match_score'] == 100)
        check(bool(analysis['presentation']['rows']))
        check(analysis['feeding']['status'] == 'READY' and analysis['feeding']['daily_serving_g'] > 0)
        check(analysis['feeding']['coefficient'] == 1.6)
        check(analysis['feeding']['production_evidence'] is False)
        check(analysis['input_provenance']['integration_data']['input_evidence_type'] == 'SCHEMA_DRIVEN_SYNTHETIC_DATA')
        check(analysis['product_label']['feeding_method'] == records[first['id']]['feeding_method'])
        compare = request('/api/nutrition/compare', {'pet_id': 1, 'product_ids': list(records), 'allergy_profile_status': 'KNOWN_NONE'})
        check(compare['comparison_status'] == 'READY')
        check(compare['comparison']['score_comparable'] is True)
        check(compare['comparison']['higher_match_score_product_id'] is None)
        check(compare['comparison']['match_score_delta'] == 0)
        check(bool(compare['comparison']['nutrient_differences']))
        check(not {'BEST', 'HEALTHIEST', 'better', 'winner'} & compare['comparison'].keys())
        records[first['id']]['caution_codes'] = ['CHOCOLATE_CACAO']
        blocked = request('/api/nutrition/analyze/by-service-id', body)
        check(blocked['safety_status'] == 'SAFETY_BLOCKED' and blocked['excluded'])
        check(blocked['feeding']['status'] == 'BLOCKED')
        check(blocked['feeding']['mer_kcal_per_day'] is None and blocked['feeding']['daily_serving_g'] is None)
        check(blocked['suitability']['match_score'] is None)
        check('FEEDING_SAFETY_EXCLUDED' in blocked['feeding']['reason_codes'])
        check(request('/api/nutrition/compare', {'pet_id': 1, 'product_ids': list(records)}, authenticated=False)['http_status'] == 401)
        records[first['id']]['caution_codes'] = []
        pet['allergies'] = ['CHICKEN']
        records[first['id']]['allergen_flags'] = ['CHICKEN']
        listed = request('/api/nutrition/analyze/by-service-id', {**body, 'allergy_profile_status': 'KNOWN_LIST'})
        check(listed['safety_status'] == 'SAFETY_BLOCKED')
        check('ALLERGY_CONFLICT' in listed['safety_reason_codes'])
        check(listed['feeding']['daily_serving_g'] is None)
        pet['allergies'] = []
        records[first['id']]['allergen_flags'] = []
        pet['birth_date'] = '2026-07-03'
        growth = request('/api/nutrition/analyze/by-service-id', body)
        check(growth['pet_reference_stage']['stage'] == 'GROWTH_REPRODUCTION')
        check(growth['target_compatibility']['stage']['pet_stage'] == 'GROWTH')
        check(growth['feeding']['status'] == 'BLOCKED')
        pet['birth_date'] = '2018-10-03'
        senior = request('/api/nutrition/analyze/by-service-id', body)
        check(senior['pet_reference_stage']['stage'] == 'ADULT_MAINTENANCE')
        check(senior['target_compatibility']['stage']['pet_stage'] == 'SENIOR')
        check(senior['suitability']['match_score'] is None)
        check(senior['feeding']['coefficient'] == 1.6)
        pet['birth_date'] = '2024-10-03'
        for specific in ('SALMON', 'TUNA'):
            pet['allergies'] = [specific]
            specific_result = request('/api/nutrition/analyze/by-service-id', {**body, 'allergy_profile_status': 'KNOWN_LIST'})
            check(specific_result['safety_status'] == 'SAFETY_DATA_INSUFFICIENT')
            check(specific_result['feeding']['daily_serving_g'] is None)
        pet['allergies'] = ['LAMB']
        snapshot_141 = next(p for p in products if p['id'] == 141)
        records[141] = deepcopy(snapshot_141)
        alias = request('/api/nutrition/analyze/by-service-id', {**body, 'product_id': 141, 'allergy_profile_status': 'KNOWN_LIST'})
        check(alias['safety_status'] == 'SAFETY_BLOCKED')
        from service_db_adapter import adapt_product
        refs = adapt_product(records[141])['product']['product_allergen_refs']
        by_raw = {r['raw_text']: r for r in refs}
        check(by_raw['양고기']['allergen_code'] == 'lamb')
        check(by_raw['귀리']['allergen_code'] == 'oat')
        check(by_raw['당근']['mapping_method'] == 'UNRESOLVED')
        check(by_raw['비트']['mapping_method'] == 'UNRESOLVED')
        return {'scope': 'REAL_LOOPBACK_HTTP_WITH_SCHEMA_DRIVEN_SYNTHETIC_SERVICE_SOURCE_BOUNDARY',
                'requests': requests, 'assertions_passed': checks,
                'smoke': {'health': 200, 'ready': 200, 'metrics': 200, 'readiness_mode': 'local_artifact'},
                'target': 'MATCHED', 'size': 'MATCHED', 'score': 100, 'feeding': 'READY',
                'safety_excluded_feeding': 'BLOCKED', 'compare': 'READY',
                'baseline_cases': ['KNOWN_NONE', 'KNOWN_LIST', 'growth/adult reference', 'senior/reference separation', 'Product 141 exact aliases', 'specific SALMON/TUNA fail-close'],
                'live_service_db': False, 'gateway_e2e': False}
    finally:
        api.service_repository.get_pet, api.service_repository.get_product = original_getters
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def main():
    """Read archived products and baseline hashes, run checks, and write a JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    products = json.loads(args.source.read_text())['products']
    baseline = json.loads(args.baseline.read_text())
    report = {'regression': replay(products, baseline), 'runtime': runtime(products)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
