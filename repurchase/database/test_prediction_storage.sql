-- 일회용 로컬 DB에서 제약과 조회 정책을 검증하고 입력 행은 롤백합니다.
BEGIN;
INSERT INTO repurchase.prediction_batches VALUES
    ('old', 'old', '2026-01-01Z', '2026-01-01Z', 'STAGING', 1, 'model', 1),
    ('new', 'new', '2026-01-02Z', '2026-01-02Z', 'STAGING', 1, 'model', 1),
    ('draft', 'draft', '2026-01-03Z', '2026-01-03Z', 'STAGING', 1, 'model', 1);
INSERT INTO repurchase.repurchase_predictions VALUES
    ('old', '1', NULL, 'PRODUCT_GROUP', '2', 30, 'READY', 0.7),
    ('new', '1', NULL, 'PRODUCT_GROUP', '2', 30, 'INSUFFICIENT_DATA', NULL),
    ('draft', '1', NULL, 'PRODUCT_GROUP', '2', 30, 'READY', 0.9);
UPDATE repurchase.prediction_batches SET publication_status = 'PUBLISHED'
WHERE publication_id IN ('old', 'new');

DO $$
DECLARE
    invalid_probability double precision;
BEGIN
    -- 작성 중 결과와 과거 READY가 최신 제공 불가 상태를 가리지 않습니다.
    IF (SELECT count(*) FROM repurchase.latest_predictions) <> 1
        OR NOT EXISTS (SELECT 1 FROM repurchase.latest_predictions
            WHERE publication_id = 'new' AND prediction_status = 'INSUFFICIENT_DATA')
    THEN
        RAISE EXCEPTION '최신 발행 결과 선택 실패';
    END IF;

    -- NULL 반려동물 키도 같은 배치 안에서는 중복을 거절합니다.
    BEGIN
        INSERT INTO repurchase.repurchase_predictions VALUES
            ('draft', '1', NULL, 'PRODUCT_GROUP', '2', 30, 'READY', 0.7);
        RAISE EXCEPTION 'NULL 키 중복 허용';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;

    -- NaN과 무한대, 결측 및 범위 밖 확률을 모두 거절합니다.
    FOREACH invalid_probability IN ARRAY ARRAY[
        NULL, '-0.1', '1.1', 'NaN', 'Infinity', '-Infinity'
    ]::double precision[] LOOP
        BEGIN
            UPDATE repurchase.repurchase_predictions
            SET conditional_repurchase_probability = invalid_probability
            WHERE publication_id = 'draft';
            RAISE EXCEPTION '잘못된 READY 확률 허용';
        EXCEPTION WHEN check_violation THEN NULL;
        END;
    END LOOP;

    BEGIN
        INSERT INTO repurchase.repurchase_predictions VALUES
            ('draft', '2', NULL, 'PRODUCT_GROUP', '2', 30,
             'INSUFFICIENT_DATA', 0.5);
        RAISE EXCEPTION '제공 불가 상태의 확률 허용';
    EXCEPTION WHEN check_violation THEN NULL;
    END;

    BEGIN
        INSERT INTO repurchase.repurchase_predictions VALUES
            ('missing', '1', NULL, 'PRODUCT_GROUP', '2', 30, 'READY', 0.7);
        RAISE EXCEPTION '존재하지 않는 배치 참조 허용';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;

    -- 결과를 하나도 적재하지 않은 배치는 완료로 표시할 수 없습니다.
    BEGIN
        INSERT INTO repurchase.prediction_batches VALUES
            ('incomplete', 'incomplete', '2026-01-04Z', '2026-01-04Z',
             'STAGING', 1, 'model', 1);
        UPDATE repurchase.prediction_batches
        SET publication_status = 'PUBLISHED' WHERE publication_id = 'incomplete';
        RAISE EXCEPTION '미완성 배치 발행 허용';
    EXCEPTION WHEN raise_exception THEN
        IF SQLERRM NOT LIKE '발행 결과 수 불일치:%' THEN RAISE; END IF;
    END;

    -- 발행 후 결과를 바꾸거나 더 넣을 수 없습니다.
    BEGIN
        UPDATE repurchase.repurchase_predictions
        SET conditional_repurchase_probability = 0.8 WHERE publication_id = 'old';
        RAISE EXCEPTION '발행된 결과 변경 허용';
    EXCEPTION WHEN raise_exception THEN
        IF SQLERRM <> 'STAGING이 아닌 배치의 예측 결과는 변경할 수 없습니다' THEN
            RAISE;
        END IF;
    END;

    -- 같은 멱등키로 다른 배치를 만들 수 없습니다.
    BEGIN
        INSERT INTO repurchase.prediction_batches VALUES
            ('duplicate', 'old', '2026-01-05Z', '2026-01-05Z',
             'STAGING', 0, 'model', 1);
        RAISE EXCEPTION '중복 멱등키 허용';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;

    -- STAGING과 결과 검사를 건너뛰는 직접 발행은 거절합니다.
    BEGIN
        INSERT INTO repurchase.prediction_batches VALUES
            ('direct', 'direct', '2026-01-05Z', '2026-01-05Z',
             'PUBLISHED', 0, 'model', 1);
        RAISE EXCEPTION '직접 발행 허용';
    EXCEPTION WHEN raise_exception THEN
        IF SQLERRM <> '새 발행 배치는 STAGING으로 시작해야 합니다' THEN
            RAISE;
        END IF;
    END;
END $$;
ROLLBACK;
