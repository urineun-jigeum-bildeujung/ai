"""BE JPA/로컬 schema에서 확인한 source 전용 SELECT 경계. 저장/DDL 없음."""
from contextlib import contextmanager
import os


class ServiceUnavailable(RuntimeError):
    pass


class ServiceNotFound(LookupError):
    pass


def configured():
    return all(os.getenv(key) for key in ("MEMBER_DATABASE_URL", "PRODUCT_DATABASE_URL"))


def service_mode():
    # 일부만 구성된 환경도 local readiness로 조용히 강등하지 않는다.
    return os.getenv("NUTRITION_RUNTIME_MODE", "local") != "local" or any(
        os.getenv(key) for key in ("MEMBER_DATABASE_URL", "PRODUCT_DATABASE_URL")
    )


def _connect(dsn):
    import psycopg2
    return psycopg2.connect(
        dsn, connect_timeout=3,
        options="-c statement_timeout=3000 -c lock_timeout=1000 -c default_transaction_read_only=on",
    )


@contextmanager
def _cursor(env_key):
    connection = None
    try:
        dsn = os.getenv(env_key)
        if not dsn:
            raise ServiceUnavailable("SERVICE_SOURCE_NOT_CONFIGURED")
        connection = _connect(dsn)
        connection.set_session(readonly=True, isolation_level="REPEATABLE READ", autocommit=False)
        with connection.cursor() as cursor:
            yield cursor
    except ServiceNotFound:
        raise
    except Exception:
        # psycopg2 오류의 DSN/SQL/row/credential 내용은 경계 밖에 전달하지 않는다.
        raise ServiceUnavailable("SERVICE_DB_UNAVAILABLE") from None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                raise ServiceUnavailable("SERVICE_DB_UNAVAILABLE") from None


def get_pet(pet_id, member_id):
    """Read one undeleted pet owned by member_id and its ordered allergy codes.

    Raise ServiceNotFound when the ownership-filtered query finds no pet;
    connection failures are translated to ServiceUnavailable by the cursor.
    """
    with _cursor("MEMBER_DATABASE_URL") as cursor:
        cursor.execute(
            "SELECT id, species, age, weight, bcs, is_neutered, birth_date, target_breed_size FROM public.pet "
            "WHERE id = %s AND member_id = %s AND deleted_at IS NULL", (pet_id, member_id),
        )
        row = cursor.fetchone()
        if row is None:
            # 타인 소유/삭제/미존재를 구별하여 유출하지 않는다.
            raise ServiceNotFound("PET_NOT_FOUND")
        cursor.execute("SELECT allergy_code FROM public.pet_allergy WHERE pet_id = %s ORDER BY allergy_code", (pet_id,))
        allergies = [item[0] for item in cursor.fetchall()]
        return dict(zip(("id", "species", "age", "weight", "bcs", "is_neutered", "birth_date", "target_breed_size"), row),
                    allergies=allergies, allergy_profile_status="KNOWN_LIST" if allergies else "UNKNOWN",
                    life_stage=None)


def get_product(product_id):
    """Read one active product with its target, allergen, ingredient, and caution codes.

    Raise ServiceNotFound when no active product exists; connection failures
    are translated to ServiceUnavailable by the cursor.
    """
    with _cursor("PRODUCT_DATABASE_URL") as cursor:
        cursor.execute(
            "SELECT id, sku, product_name, category_code, subcategory_code, target_age_group, "
            "target_breed_size, feeding_target, feeding_method "
            "FROM public.products WHERE id = %s AND is_active = TRUE", (product_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ServiceNotFound("PRODUCT_NOT_FOUND")
        result = dict(zip(("id", "sku", "product_name", "category_code", "subcategory_code", "target_age_group",
                 "target_breed_size", "feeding_target", "feeding_method"), row))
        for field, query in (
            ("target_species", "SELECT species FROM public.product_target_species WHERE product_id = %s ORDER BY species"),
            ("allergen_flags", "SELECT allergen_code FROM public.product_allergens WHERE product_id = %s ORDER BY allergen_code"),
            ("ingredient_codes", "SELECT ingredient_code FROM public.product_ingredients WHERE product_id = %s ORDER BY ingredient_code"),
            ("caution_codes", "SELECT caution_code FROM public.product_cautions WHERE product_id = %s ORDER BY caution_code"),
        ):
            cursor.execute(query, (product_id,))
            result[field] = [item[0] for item in cursor.fetchall()]
        return result


def probe(env_key):
    try:
        with _cursor(env_key) as cursor:
            cursor.execute("SELECT 1")
            if cursor.fetchone() != (1,):
                return {"required": True, "status": "DOWN"}
        return {"required": True, "status": "UP"}
    except ServiceUnavailable:
        return {"required": True, "status": "DOWN"}

def list_active_products():
    """Read the active Service integration product master without mutating DB."""
    with _cursor("PRODUCT_DATABASE_URL") as cursor:
        cursor.execute(
            "SELECT id, sku, product_name, category_code, subcategory_code, target_age_group, "
            "target_breed_size, feeding_target, feeding_method "
            "FROM public.products WHERE is_active = TRUE ORDER BY id"
        )
        base_rows = cursor.fetchall()
        if not base_rows:
            return []
        products = {
            row[0]: dict(zip(
                ("id", "sku", "product_name", "category_code", "subcategory_code", "target_age_group",
                 "target_breed_size", "feeding_target", "feeding_method"),
                row,
            ), target_species=[], allergen_flags=[], ingredient_codes=[], caution_codes=[])
            for row in base_rows
        }
        ids = list(products)
        for field, query in (
            ("target_species", "SELECT product_id, species FROM public.product_target_species WHERE product_id = ANY(%s) ORDER BY product_id, species"),
            ("allergen_flags", "SELECT product_id, allergen_code FROM public.product_allergens WHERE product_id = ANY(%s) ORDER BY product_id, allergen_code"),
            ("ingredient_codes", "SELECT product_id, ingredient_code FROM public.product_ingredients WHERE product_id = ANY(%s) ORDER BY product_id, sort_order"),
            ("caution_codes", "SELECT product_id, caution_code FROM public.product_cautions WHERE product_id = ANY(%s) ORDER BY product_id, caution_code"),
        ):
            cursor.execute(query, (ids,))
            for product_id, value in cursor.fetchall():
                if product_id in products:
                    products[product_id][field].append(value)
        return [products[product_id] for product_id in ids]
