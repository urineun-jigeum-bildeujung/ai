-- 내부 검증 배치만 별도 종료 상태로 보존합니다. latest_predictions에는 노출하지 않습니다.
-- 001_prediction_storage.sql 적용 후 한 번만 실행합니다.
BEGIN;

ALTER TABLE repurchase.prediction_batches
    DROP CONSTRAINT prediction_batches_publication_status_check;
ALTER TABLE repurchase.prediction_batches
    ADD CONSTRAINT prediction_batches_publication_status_check
    CHECK (publication_status IN ('STAGING', 'PUBLISHED', 'FAILED', 'SHADOW'));

CREATE OR REPLACE FUNCTION repurchase.guard_publication_status() RETURNS trigger
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
    IF NEW.publication_status IN ('PUBLISHED', 'SHADOW') THEN
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

COMMIT;
