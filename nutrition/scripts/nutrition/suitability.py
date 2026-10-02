"""Objective required-nutrient ratio, using the existing P0-D matrix result."""

def build_suitability(*, nutrition_comparison, target_compatibility, safety_status, excluded):
    comparison = nutrition_comparison or {}
    essential = comparison.get('essential_status') or {}
    required = comparison.get('essential_count', len(essential))
    passes = comparison.get('pass_count', 0)
    fails = comparison.get('fail_count', 0)
    comparable = comparison.get('comparable_count', 0)
    target = target_compatibility.get('status')
    reasons = [f'NUTRIENT_{suffix}:{code}' for code, status in sorted(essential.items())
               for suffix in [{'OUT_OF_RANGE': 'OUT_OF_RANGE', 'NON_COMPARABLE': 'NON_COMPARABLE',
                               'MISSING': 'MISSING', 'DATA_CONFLICT': 'DATA_CONFLICT'}.get(status)] if suffix]
    if excluded or safety_status == 'SAFETY_BLOCKED':
        status = 'BLOCKED'
        reasons.append('SAFETY_EXCLUDED')
    elif safety_status == 'SAFETY_DATA_INSUFFICIENT':
        status = 'INSUFFICIENT_DATA'
        reasons.append('SAFETY_DATA_INSUFFICIENT')
    elif target == 'MISMATCH':
        status = 'TARGET_MISMATCH'
        reasons.extend(target_compatibility.get('reason_codes', []))
    elif target == 'CONFLICT':
        status = 'TARGET_CONFLICT'
        reasons.extend(target_compatibility.get('reason_codes', []))
    elif (target != 'MATCHED' or comparison.get('nutrition_comparison_status') not in {'TRUE', 'FALSE'}
          or required <= 0 or comparable != required or passes + fails != required
          or len(essential) != required or any(v not in {'IN_RANGE', 'OUT_OF_RANGE'} for v in essential.values())):
        status = 'INSUFFICIENT_DATA'
        reasons.extend(target_compatibility.get('reason_codes', []))
        reasons.append('REQUIRED_COMPARABLE_DATA_INCOMPLETE')
    else:
        status = 'MATCHED'
    return {'status': status, 'match_score': round(100 * passes / (passes + fails)) if status == 'MATCHED' else None,
            'score_type': 'REQUIRED_NUTRIENT_IN_RANGE_RATIO', 'score_version': 'nutrition_match_score_v1',
            'pass_count': passes, 'fail_count': fails, 'required_count': required,
            'reason_codes': list(dict.fromkeys(reasons))}
