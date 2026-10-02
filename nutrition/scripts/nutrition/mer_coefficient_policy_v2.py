"""healthy_merck_v1: starting estimates, never a health diagnosis."""
import math
from decimal import Decimal

from service_db_adapter import _birth_age, _today, ServiceInputError

POLICY_CONTRACT_VERSION = 'healthy_merck_v1'


def resolve_mer_coefficient(pet):
    """Return a healthy baseline MER factor with policy provenance and reasons.

    Prefer birth-date age, falling back to numeric age when unresolved. Growth
    factors depend on species and age; adult factors also require a boolean
    neuter flag. Unsupported or incomplete inputs return an unresolved policy.
    """
    species = str(pet.get('species') or '').upper()
    months = None
    if pet.get('birth_date') is not None:
        try:
            years, _ = _birth_age(pet['birth_date'], _today())
            months = round(years * 12)
        except ServiceInputError:
            pass
    if months is None:
        years = pet.get('age_years', pet.get('age'))
        if isinstance(years, (int, float, Decimal)) and not isinstance(years, bool):
            try:
                years = float(years)
                if math.isfinite(years) and years >= 0:
                    months = years * 12
            except (OverflowError, ValueError):
                pass
    reason = None
    factor = code = None
    if species not in {'DOG', 'CAT'}:
        reason = 'PET_SPECIES_UNSUPPORTED'
    elif months is None:
        reason = 'PET_AGE_UNRESOLVED'
    elif months < 12:
        if species == 'CAT':
            factor, code = 2.5, 'CAT_GROWTH'
        elif months < 4:
            factor, code = 3.0, 'DOG_GROWTH_UNDER_4_MONTHS'
        else:
            factor, code = 2.0, 'DOG_GROWTH_4_TO_11_MONTHS'
    elif type(pet.get('is_neutered')) is not bool:
        reason = 'NEUTER_STATUS_MISSING_OR_INVALID'
    else:
        neutered = pet['is_neutered']
        factor = (1.6 if neutered else 1.8) if species == 'DOG' else (1.2 if neutered else 1.4)
        code = species + '_ADULT_' + ('NEUTERED' if neutered else 'INTACT')
    return {'coefficient': factor, 'coefficient_code': code,
            'coefficient_source_type': 'MERCK_HEALTHY_BASELINE' if factor is not None else None,
            'coefficient_version': POLICY_CONTRACT_VERSION if factor is not None else None,
            'policy_status': 'RESOLVED' if factor is not None else 'UNRESOLVED',
            'policy_contract_version': POLICY_CONTRACT_VERSION,
            'applicability': 'HEALTHY_BASELINE_STARTING_ESTIMATE',
            'disclaimer_code': 'BASELINE_ESTIMATE_NOT_MEDICAL_PRESCRIPTION',
            'reason_codes': [reason] if reason else []}
