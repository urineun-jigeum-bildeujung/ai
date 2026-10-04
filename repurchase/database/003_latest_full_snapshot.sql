-- 현재 배치는 전체 대상 스냅샷입니다. 빠진 키에 과거 예측을 이어 붙이지 않습니다.
BEGIN;

CREATE OR REPLACE VIEW repurchase.latest_predictions AS
SELECT r.*, b.as_of_timestamp, b.created_at, b.artifact_id,
    b.feature_generation_version
FROM repurchase.repurchase_predictions AS r
JOIN (
    SELECT publication_id, as_of_timestamp, created_at, artifact_id,
        feature_generation_version
    FROM repurchase.prediction_batches
    WHERE publication_status = 'PUBLISHED'
    ORDER BY as_of_timestamp DESC, created_at DESC,
        publication_id COLLATE "C" DESC
    LIMIT 1
) AS b USING (publication_id);

COMMIT;
