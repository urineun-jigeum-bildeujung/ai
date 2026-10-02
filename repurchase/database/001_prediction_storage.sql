-- AI 전용 재구매 결과 저장 구조입니다. 합의된 대상 DB에서만 적용합니다.
-- PostgreSQL 15 이상이 필요하며, 같은 버전을 다시 적용하면 오류로 중단합니다.
BEGIN;

CREATE SCHEMA repurchase;

CREATE TABLE repurchase.prediction_batches (
    publication_id text PRIMARY KEY CHECK (btrim(publication_id) <> ''),
    idempotency_key text NOT NULL UNIQUE CHECK (btrim(idempotency_key) <> ''),
    as_of_timestamp timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    publication_status text NOT NULL
        CHECK (publication_status IN ('STAGING', 'PUBLISHED', 'FAILED'))
        DEFAULT 'STAGING',
    expected_result_count bigint NOT NULL CHECK (expected_result_count >= 0),
    artifact_id text NOT NULL CHECK (btrim(artifact_id) <> ''),
    feature_generation_version integer NOT NULL
        CHECK (feature_generation_version > 0)
);

CREATE TABLE repurchase.repurchase_predictions (
    publication_id text NOT NULL
        REFERENCES repurchase.prediction_batches(publication_id),
    user_id text NOT NULL CHECK (btrim(user_id) <> ''),
    pet_id bigint,
    target_scope text NOT NULL CHECK (target_scope IN ('PRODUCT_GROUP', 'CATEGORY')),
    target_id text NOT NULL CHECK (btrim(target_id) <> ''),
    window_days integer NOT NULL CHECK (window_days > 0),
    prediction_status text NOT NULL
        CHECK (prediction_status IN ('READY', 'INSUFFICIENT_DATA', 'SUPPRESSED')),
    conditional_repurchase_probability double precision,
    -- 미지정 반려동물도 같은 대상이므로 NULL 키 중복을 허용하지 않습니다.
    UNIQUE NULLS NOT DISTINCT (
        publication_id, user_id, pet_id, target_scope, target_id, window_days
    ),
    -- 결측을 SQL CHECK가 통과시키지 않도록 READY의 NOT NULL도 명시합니다.
    CHECK (
        (prediction_status = 'READY'
            AND conditional_repurchase_probability IS NOT NULL
            AND conditional_repurchase_probability >= 0
            AND conditional_repurchase_probability <= 1)
        OR (prediction_status <> 'READY'
            AND conditional_repurchase_probability IS NULL)
    )
);

-- 결과 변경과 발행 전환 모두 부모 행을 잠가 동시 적재와 발행을 직렬화합니다.
CREATE FUNCTION repurchase.guard_prediction_write() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    batch_status text;
BEGIN
    IF TG_OP = 'UPDATE' AND NEW.publication_id <> OLD.publication_id THEN
        RAISE EXCEPTION '예측 결과의 publication_id는 변경할 수 없습니다';
    END IF;
    SELECT publication_status INTO batch_status
    FROM repurchase.prediction_batches
    WHERE publication_id = CASE WHEN TG_OP = 'DELETE' THEN OLD.publication_id
                                ELSE NEW.publication_id END
    FOR UPDATE;
    -- 없는 부모는 기존 외래키가 진단하도록 둡니다.
    IF NOT FOUND THEN
        IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
        RETURN NEW;
    END IF;
    IF batch_status IS DISTINCT FROM 'STAGING' THEN
        RAISE EXCEPTION 'STAGING이 아닌 배치의 예측 결과는 변경할 수 없습니다';
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END $$;

CREATE TRIGGER guard_prediction_write
BEFORE INSERT OR UPDATE OR DELETE ON repurchase.repurchase_predictions
FOR EACH ROW EXECUTE FUNCTION repurchase.guard_prediction_write();

CREATE FUNCTION repurchase.guard_publication_status() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    actual_count bigint;
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.publication_status <> 'STAGING' THEN
            RAISE EXCEPTION '종료된 발행 배치는 삭제할 수 없습니다';
        END IF;
        RETURN OLD;
    END IF;
    IF TG_OP = 'INSERT' THEN
        IF NEW.publication_status <> 'STAGING' THEN
            RAISE EXCEPTION '새 발행 배치는 STAGING으로 시작해야 합니다';
        END IF;
        RETURN NEW;
    END IF;
    IF OLD.publication_status <> 'STAGING' THEN
        RAISE EXCEPTION '종료된 발행 배치는 수정할 수 없습니다';
    END IF;
    IF NEW.publication_id IS DISTINCT FROM OLD.publication_id
        OR NEW.idempotency_key IS DISTINCT FROM OLD.idempotency_key
        OR NEW.as_of_timestamp IS DISTINCT FROM OLD.as_of_timestamp
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
        OR NEW.expected_result_count IS DISTINCT FROM OLD.expected_result_count
        OR NEW.artifact_id IS DISTINCT FROM OLD.artifact_id
        OR NEW.feature_generation_version IS DISTINCT FROM OLD.feature_generation_version
    THEN
        RAISE EXCEPTION 'STAGING 배치에서는 상태만 변경할 수 있습니다';
    END IF;
    IF NEW.publication_status = 'PUBLISHED' THEN
        SELECT count(*) INTO actual_count
        FROM repurchase.repurchase_predictions
        WHERE publication_id = NEW.publication_id;
        IF actual_count <> NEW.expected_result_count THEN
            RAISE EXCEPTION '발행 결과 수 불일치: expected %, actual %',
                NEW.expected_result_count, actual_count;
        END IF;
    END IF;
    RETURN NEW;
END $$;

CREATE TRIGGER guard_publication_status
BEFORE INSERT OR UPDATE OR DELETE ON repurchase.prediction_batches
FOR EACH ROW EXECUTE FUNCTION repurchase.guard_publication_status();

-- READY로 먼저 거르지 않아 최신의 제공 불가 상태도 유지합니다.
CREATE VIEW repurchase.latest_predictions AS
SELECT DISTINCT ON (
    r.user_id, r.pet_id, r.target_scope, r.target_id, r.window_days
)
    r.*, b.as_of_timestamp, b.created_at, b.artifact_id,
    b.feature_generation_version
FROM repurchase.repurchase_predictions AS r
JOIN repurchase.prediction_batches AS b USING (publication_id)
WHERE b.publication_status = 'PUBLISHED'
ORDER BY r.user_id, r.pet_id, r.target_scope, r.target_id, r.window_days,
    b.as_of_timestamp DESC, b.created_at DESC, b.publication_id COLLATE "C" DESC;

COMMIT;
