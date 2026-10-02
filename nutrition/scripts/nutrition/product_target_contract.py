"""Exact schema-driven marketing targets, independent of AAFCO and Safety."""
import json
import unicodedata
from pathlib import Path

from mock_integration_fixture import is_mock_sku

FIXTURE_PATH = Path(__file__).resolve().parents[2] / 'data/integration/mock_product_target_v1.json'
STAGES = {'GROWTH', 'ADULT', 'SENIOR', 'ALL_LIFE_STAGES'}
SIZES = {'SMALL', 'MEDIUM', 'LARGE'}
RAW_LABELS = {
    prefix + ' ' + label: {'target_species': [species], 'target_sizes': sizes, 'target_stage': stage}
    for prefix, species, sizes in [('소형견', 'DOG', ['SMALL']),
                                  ('중·대형견', 'DOG', ['MEDIUM', 'LARGE']),
                                  ('고양이', 'CAT', [])]
    for label, stage in [('성장기', 'GROWTH'), ('성체', 'ADULT'),
                         ('노령', 'SENIOR'), ('전 연령', 'ALL_LIFE_STAGES')]
}


def parse_raw_target_label(value):
    """Map an exact NFKC-normalized label to independent species, size, and stage data.

    Trim outer whitespace and return unknown targets for unrecognized labels.
    Copy mapped lists so callers cannot mutate the shared label definitions.
    """
    label = unicodedata.normalize('NFKC', value).strip() if isinstance(value, str) else None
    mapped = RAW_LABELS.get(label)
    if mapped is None:
        return {'target_species': [], 'target_sizes': [], 'target_stage': 'UNKNOWN'}
    return {k: list(v) if isinstance(v, list) else v for k, v in mapped.items()}


def load_synthetic_target(source):
    """Return a validated synthetic target row for an exact MOCK product identity.

    Require unique ID/SKU matching, synthetic provenance, and agreement with
    the exact raw label. Return None for missing, invalid, or ambiguous data.
    """
    if not is_mock_sku(source.get('sku')):
        return None
    try:
        payload = json.loads(FIXTURE_PATH.read_text(encoding='utf-8'))
        if (payload.get('fixture_version') != 'mock_product_target_v1'
                or payload.get('source_type') != 'MOCK_INTEGRATION_FIXTURE'
                or payload.get('data_generation_type') != 'SCHEMA_DRIVEN_SYNTHETIC'
                or payload.get('production_evidence') is not False):
            return None
        rows = [row for row in payload['items']
                if str(row.get('service_product_id')) == str(source.get('id'))
                and row.get('service_sku') == source.get('sku')]
        if len(rows) != 1:
            return None
        row = rows[0]
        parsed = parse_raw_target_label(row.get('raw_target_label'))
        if any(row.get(key) != parsed[key] for key in parsed) or parsed['target_stage'] == 'UNKNOWN':
            return None
        return {**row, 'fixture_version': payload['fixture_version'],
                'data_generation_type': payload['data_generation_type'], 'production_evidence': False}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _resolve(service, fixture, allowed, *, sequence=False):
    """Return a target value, resolution status, and source for one target axis.

    Prefer present Service data, use fixture data only when Service data is
    absent, and report conflicting or invalid evidence without inventing a value.
    """
    service_present = service is not None and service != []
    fixture_present = fixture is not None and fixture != []
    def valid(value):
        """Check that a scalar or every list element belongs to the allowed target values."""
        if sequence:
            return isinstance(value, list) and all(isinstance(v, str) and v in allowed for v in value)
        return isinstance(value, str) and value in allowed
    if service_present and fixture_present and (not valid(service) or not valid(fixture) or service != fixture):
        return None, 'CONFLICT', 'SERVICE_STRUCTURED'
    if service_present:
        return (service, 'RESOLVED', 'SERVICE_STRUCTURED') if valid(service) else (None, 'UNKNOWN', 'SERVICE_STRUCTURED')
    if fixture_present:
        return (fixture, 'RESOLVED', 'SCHEMA_DRIVEN_SYNTHETIC_FIXTURE') if valid(fixture) else (None, 'UNKNOWN', 'SCHEMA_DRIVEN_SYNTHETIC_FIXTURE')
    return None, 'UNKNOWN', 'UNKNOWN'


def resolve_product_target(product, synthetic_fixture=None):
    """Resolve species, stage, and size with per-axis status and provenance.

    Use synthetic data only for a matching MOCK ID/SKU. Preserve conflicts
    between Service and fixture evidence and report unresolved target axes.
    """
    # Supplying a fixture cannot bypass the Service identity namespace.
    fixture = synthetic_fixture or {}
    if (not is_mock_sku(product.get('service_sku'))
            or str(fixture.get('service_product_id')) != str(product.get('id'))
            or fixture.get('service_sku') != product.get('service_sku')):
        fixture = {}
    species = product.get('target_species')
    targets = {'dog': ['DOG'], 'cat': ['CAT'], 'both': ['CAT', 'DOG']}.get(species, species if isinstance(species, list) else [])
    targets = sorted(set(targets))
    fixture_species = sorted(set(fixture.get('target_species') or []))
    resolved_species, species_status, species_source = _resolve(targets, fixture_species, {'DOG', 'CAT'}, sequence=True)
    stage, stage_status, stage_source = _resolve(product.get('service_target_age_group'), fixture.get('target_stage'), STAGES)
    size = product.get('service_target_breed_size')
    sizes, size_status, size_source = _resolve([size] if size else [], fixture.get('target_sizes'), SIZES, sequence=True)
    reasons = [f'PRODUCT_TARGET_{axis}_{status}' for axis, status in
               [('SPECIES', species_status), ('STAGE', stage_status), ('SIZE', size_status)]
               if status != 'RESOLVED']
    return {'species': {'targets': resolved_species or targets, 'status': species_status, 'source': species_source},
            'stage': {'value': stage or stage_status, 'status': stage_status, 'source': stage_source,
                      'raw_label': fixture.get('raw_target_label')},
            'size': {'values': sizes or [], 'status': size_status, 'source': size_source},
            'mapping_version': 'product_target_contract_v2',
            'conflict': 'CONFLICT' in (species_status, stage_status, size_status), 'reason_codes': reasons,
            'fixture_provenance': {k: fixture[k] for k in ('fixture_version', 'data_generation_type', 'production_evidence') if k in fixture}}


def evaluate_target_compatibility(pet, product_target):
    """Compare pet species, marketing stage, and dog size against resolved targets.

    Cat size is not applicable. Overall status prioritizes conflict, mismatch,
    and unknown over matched, retaining per-axis details and reason codes.
    """
    species = str(pet.get('species') or '').upper()
    target_species = product_target['species']
    species_status = ('CONFLICT' if target_species.get('status') == 'CONFLICT' else
                      'UNKNOWN' if target_species.get('status') != 'RESOLVED' or not target_species['targets'] or species not in {'DOG', 'CAT'} else
                      'MATCHED' if species in target_species['targets'] else 'MISMATCH')
    stage = product_target['stage']
    pet_stage = pet.get('product_target_stage')
    stage_status = ('CONFLICT' if stage['status'] == 'CONFLICT' else
                    'UNKNOWN' if stage['status'] != 'RESOLVED' or pet_stage not in {'GROWTH', 'ADULT', 'SENIOR'} else
                    'MATCHED' if stage['value'] in {pet_stage, 'ALL_LIFE_STAGES'} else 'MISMATCH')
    size = product_target['size']
    pet_size = pet.get('service_target_breed_size', pet.get('target_breed_size'))
    size_status = ('NOT_APPLICABLE' if species == 'CAT' else
                   'CONFLICT' if size['status'] == 'CONFLICT' else
                   'UNKNOWN' if species != 'DOG' or pet_size not in SIZES or not size['values'] else
                   'MATCHED' if pet_size in size['values'] else 'MISMATCH')
    statuses = [species_status, stage_status, size_status]
    overall = next(s for s in ('CONFLICT', 'MISMATCH', 'UNKNOWN', 'MATCHED') if s in statuses)
    return {'status': overall, 'policy_version': 'target_compatibility_v2',
            'species': {**target_species, 'pet_species': species, 'status': species_status},
            'stage': {**stage, 'pet_stage': pet_stage, 'status': stage_status,
                      'pet_policy_version': 'pet_product_target_stage_v1'},
            'size': {**size, 'pet_size': pet_size, 'status': size_status},
            'reason_codes': [f'TARGET_{axis}_{status}' for axis, status in
                             zip(('SPECIES', 'STAGE', 'SIZE'), statuses) if status in {'CONFLICT', 'MISMATCH', 'UNKNOWN'}]}
