"""Expose engine values and reference bounds without UI geometry or claims."""
import json
import math

from reference_parity import ARTIFACT

DISPLAY_NAMES = dict(zip(
    ['CRUDE_PROTEIN', 'CRUDE_FAT', 'MOISTURE', 'CRUDE_FIBER', 'CALCIUM', 'PHOSPHORUS',
     'SODIUM', 'MAGNESIUM', 'POTASSIUM', 'IRON', 'COPPER', 'ZINC', 'TAURINE',
     'VITAMIN_A', 'VITAMIN_D', 'VITAMIN_E', 'VITAMIN_B1', 'VITAMIN_B2'],
    ['조단백질', '조지방', '수분', '조섬유', '칼슘', '인', '나트륨', '마그네슘', '칼륨',
     '철', '구리', '아연', '타우린', '비타민 A', '비타민 D', '비타민 E', '비타민 B1', '비타민 B2']))
ORDER = ['CRUDE_PROTEIN', 'CRUDE_FAT', 'MOISTURE', 'CALCIUM', 'PHOSPHORUS', 'TAURINE']


def nutrient_sort_key(code):
    return (ORDER.index(code) if code in ORDER else len(ORDER), code)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_nutrition_presentation(compared_items, nutrition_status):
    rows = []
    # Reference unit metadata comes from the selected curated artifact. The
    # engine's legacy safe_upper_unit is not the selected NIAS unit for every code.
    reference_units = {}
    if any(item.get('reference_selection_status') == 'SELECTED' for item in compared_items):
        try:
            for reference in json.loads(ARTIFACT.read_text(encoding='utf-8'))['rows']:
                key = (reference['nutrient_code'], reference['basis'])
                reference_units.setdefault(key, set()).add(reference['unit'])
        except (OSError, ValueError, KeyError, TypeError):
            reference_units = {}
    for item in compared_items:
        code = item['nutrient_code']
        status = item.get('nias_compare_status', 'UNKNOWN')
        basis = item.get('basis_normalized_basis') or item.get('basis')
        normalized = next((item[k] for k in ('basis_normalized_value', 'aligned_value', 'value')
                           if item.get(k) is not None), None)
        # Invalid normalization must never present a fallback as normalized evidence.
        if item.get('basis_invalid'):
            normalized = None
        reference_unit = item.get('reference_unit')
        if reference_unit is None and item.get('reference_selection_status') == 'SELECTED':
            units = reference_units.get((code, basis), set())
            if len(units) == 1:
                unit = next(iter(units))
                if unit == 'PERCENT':
                    reference_unit = 'PERCENT'
                else:
                    prefix = {'GRAM': 'g/100g', 'MILLIGRAM': 'mg/100g', 'IU': 'IU/100g'}.get(unit)
                    suffix = {'DRY_MATTER': 'DM', 'AS_FED': 'AF'}.get(basis)
                    reference_unit = prefix + ' ' + suffix if prefix and suffix else None
        elif reference_unit is None:
            reference_unit = item.get('safe_upper_unit')
            if reference_unit and basis == 'AS_FED':
                reference_unit = reference_unit.replace(' DM', ' AF')
        source = item.get('source_type') or item.get('source')
        synthetic = isinstance(source, str) and source.startswith('MOCK_INTEGRATION_FIXTURE')
        rows.append({'nutrient_code': code, 'display_name': DISPLAY_NAMES.get(code, code),
                     'value': item.get('value'), 'unit': item.get('unit'), 'basis': item.get('basis'),
                     'normalized_value': normalized, 'normalized_basis': basis,
                     'reference_min': None if status == 'NO_REF' else item.get('nias_min'),
                     'reference_max': None if status == 'NO_REF' else item.get('nias_max'),
                     'reference_unit': None if status == 'NO_REF' else reference_unit,
                     'compare_status': status, 'canonical_status': item.get('canonical_status'),
                     'reference_reason_code': item.get('reference_reason_code'),
                     'reference_provenance': item.get('reference_provenance', []),
                     'source_type': 'MOCK_INTEGRATION_FIXTURE' if synthetic else source,
                     'source': item.get('source'),
                     'production_evidence': False if synthetic else item.get('production_evidence')})
    rows.sort(key=lambda r: (nutrient_sort_key(r['nutrient_code']), str(r['basis']), str(r['unit']), str(r['value'])))
    counts = {s: sum(r['compare_status'] == s for r in rows) for s in ('IN_RANGE', 'OUT_OF_RANGE')}
    comparable = sum(counts.values())
    return {'nutrition_comparison_status': nutrition_status, 'rows': rows,
            'summary': {'total_count': len(rows), 'comparable_count': comparable,
                        'in_range_count': counts['IN_RANGE'], 'out_of_range_count': counts['OUT_OF_RANGE'],
                        'unknown_count': len(rows) - comparable}}
