"""Compare two analyzed Service products without selecting nutrient winners."""
from presentation import numeric, nutrient_sort_key


def _unit_family(unit):
    """Map supported nutrient unit aliases to a common family, or return None."""
    if not isinstance(unit, str):
        return None
    value = unit.upper().replace(' ', '')
    if value in {'PERCENT', '%', 'GRAM', 'G/100G', 'G/100GDM', 'G/100GAF'}:
        return 'PERCENT'
    if value in {'MILLIGRAM', 'MG/100G', 'MG/100GDM', 'MG/100GAF'}:
        return 'MG/100G'
    if value in {'IU', 'IU/100G', 'IU/100GDM', 'IU/100GAF'}:
        return 'IU/100G'
    return None


def _comparable_rows(result):
    """Index presentation rows by nutrient code, excluding codes with duplicate evidence."""
    grouped = {}
    for row in result.get('presentation', {}).get('rows', []):
        code = row['nutrient_code']
        grouped.setdefault(code, []).append(row)
    # Multiple evidence rows for one nutrient are ambiguous, not an invitation
    # to pick the most convenient row.
    return {code: rows[0] for code, rows in grouped.items() if len(rows) == 1}


def compare_service_analyses(pet_id, product_ids, analyses):
    """Compare two analyses in product_ids order for the same pet.

    Compare finite suitability scores and unique nutrient rows with compatible
    bases and units. Return READY for comparable scores, PARTIAL for nutrient
    comparisons alone, or UNAVAILABLE when neither comparison is possible.
    """
    scores = [r.get('suitability', {}).get('match_score') for r in analyses]
    score_comparable = all(numeric(s) for s in scores)
    higher = (product_ids[0] if scores[0] > scores[1] else product_ids[1] if scores[1] > scores[0] else None) if score_comparable else None
    rows1, rows2 = (_comparable_rows(r) for r in analyses)
    differences = []
    for code in sorted(rows1.keys() & rows2.keys(), key=nutrient_sort_key):
        a, b = rows1[code], rows2[code]
        family = _unit_family(a.get('reference_unit'))
        if (not all(numeric(r.get('normalized_value')) for r in (a, b))
                or a.get('normalized_basis') not in {'AS_FED', 'DRY_MATTER'}
                or a['normalized_basis'] != b.get('normalized_basis')
                or family is None or family != _unit_family(b.get('reference_unit'))
                or _unit_family(a.get('unit')) != family or _unit_family(b.get('unit')) != family):
            continue
        differences.append({'nutrient_code': code, 'display_name': a['display_name'],
                            'product_1': {'value': a['normalized_value'], 'compare_status': a['compare_status']},
                            'product_2': {'value': b['normalized_value'], 'compare_status': b['compare_status']},
                            'absolute_delta': abs(a['normalized_value'] - b['normalized_value']),
                            'basis': a['normalized_basis'], 'unit': family})
    return {'comparison_status': 'READY' if score_comparable else 'PARTIAL' if differences else 'UNAVAILABLE',
            'pet_id': pet_id, 'product_ids': product_ids,
            'products': [{key: result.get(key) for key in ('product_id', 'suitability', 'target_compatibility',
                                                          'feeding', 'presentation', 'product_label')}
                         for result in analyses],
            'comparison': {'score_comparable': score_comparable, 'higher_match_score_product_id': higher,
                           'match_score_delta': abs(scores[0] - scores[1]) if score_comparable else None,
                           'nutrient_differences': differences}}
