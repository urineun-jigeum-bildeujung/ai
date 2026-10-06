"""Service 프로필 상태가 DB 조회부터 추천 API까지 보존되는지 검증한다.

모델 점수와 DB 연결만 합성 대역으로 바꾸며 모델 품질/운영 E2E를 주장하지 않는다.
"""
from datetime import date
from pathlib import Path
import sys

import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_access import pet_repository as repository
from src.recommend.allergy_filter import evaluate_recommendation_allergy, resolve_allergy_profile
from src.api import main as api
import pipeline


@pytest.mark.parametrize("status,codes,expected", [
    (None, [], "UNKNOWN"), (None, ["CHICKEN"], "KNOWN_LIST"),
    ("UNKNOWN", [], "UNKNOWN"), ("UNKNOWN", ["CHICKEN"], "UNKNOWN"),
    ("KNOWN_NONE", [], "KNOWN_NONE"), ("KNOWN_NONE", ["CHICKEN"], "UNKNOWN"),
    ("KNOWN_LIST", [], "UNKNOWN"), ("KNOWN_LIST", ["CHICKEN"], "KNOWN_LIST"),
    ("INVALID", [], "UNKNOWN"), ("KNOWN_NONE", None, "UNKNOWN"),
    ("KNOWN_LIST", [" "], "UNKNOWN"), ("KNOWN_LIST", [None], "UNKNOWN"),
])
def test_profile_consistency(status, codes, expected):
    assert resolve_allergy_profile(status, codes) == expected


def pet(status="UNKNOWN", codes=None):
    return {"pet_id": 1, "user_id": 2, "species": "DOG", "breed": "POODLE",
            "birth_date": "2023-01-01", "sex": "MALE", "neutered": True,
            "weight": 10.0, "bcs": 3, "concerns": [],
            "allergy_codes": [] if codes is None else codes, "allergy_profile_status": status}


def product(product_id=10, flags=None):
    return {"product_id": product_id, "product_name": "Synthetic food", "status": "ON_SALE",
            "target_species": ["DOG"], "category_code": "FOOD", "subcategory_code": "DRY",
            "target_age_group": None, "ingredients": ["BEEF"],
            "allergen_flags": ["BEEF"] if flags is None else flags}


@pytest.mark.parametrize("status,codes,flags,expected", [
    ("UNKNOWN", [], ["BEEF"], "PENDING"),
    ("KNOWN_NONE", [], ["BEEF"], "SAFE"),
    ("KNOWN_LIST", ["CHICKEN"], ["BEEF"], "SAFE"),
    ("KNOWN_LIST", ["CHICKEN"], ["chicken"], "PENALIZED"),
    ("UNKNOWN", ["CHICKEN"], ["CHICKEN"], "PENALIZED"),
    ("KNOWN_NONE", ["CHICKEN"], ["CHICKEN"], "PENALIZED"),
    ("KNOWN_NONE", ["CHICKEN"], ["BEEF"], "PENDING"),
    ("KNOWN_NONE", [], None, "PENDING"),
    ("KNOWN_LIST", ["CHICKEN"], [None], "PENDING"),
    (None, [], [], "PENDING"),
    (None, ["CHICKEN"], [], "SAFE"),
])
def test_recommendation_status(status, codes, flags, expected):
    assert evaluate_recommendation_allergy(pet(status, codes), flags)["allergy_status"] == expected


@pytest.mark.parametrize("stored,codes,expected", [
    (None, [], "UNKNOWN"), (None, ["CHICKEN"], "KNOWN_LIST"),
    ("KNOWN_NONE", [], "KNOWN_NONE"), ("UNKNOWN", [], "UNKNOWN"),
    ("KNOWN_LIST", ["CHICKEN"], "KNOWN_LIST"),
    ("KNOWN_NONE", ["CHICKEN"], "UNKNOWN"),
])
def test_repository_reads_profile_for_single_and_batch(monkeypatch, stored, codes, expected):
    row = (1, 2, "DOG", "POODLE", date(2023, 1, 1), "MALE", True, 10, 3, codes, [], stored)
    queries = []
    closed = []

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def execute(self, sql, params): queries.append((sql, params))
        def fetchone(self): return row
        def fetchall(self): return [row]

    class Connection:
        def cursor(self): return Cursor()
        def close(self): closed.append(True)

    monkeypatch.setattr(repository, "get_connection", lambda _: Connection())
    monkeypatch.setattr(repository, "USE_DUMMY_DATA", False)
    assert repository.get_pet_by_id(1)["allergy_profile_status"] == expected
    assert repository.get_pets_by_ids([1])[1]["allergy_profile_status"] == expected
    assert all("to_jsonb(p)->>'allergy_profile_status'" in sql for sql, _ in queries)
    assert [params for _, params in queries] == [(1,), ([1],)]
    assert len(closed) == 2


@pytest.fixture
def synthetic_api(monkeypatch):
    monkeypatch.setenv("INTERNAL_GATEWAY_SECRET", "synthetic-test-secret")
    class Encoder:
        def encode(self, features): return features
        def collate(self, encoded): return encoded

    monkeypatch.setattr(api, "_get_model", lambda: (Encoder(), lambda _: torch.tensor(0.8)))
    monkeypatch.setattr(api, "load_reviews_with_reviewer_pet", lambda: [])
    monkeypatch.setattr(api, "get_purchased_product_ids_for_user", lambda _: [])
    monkeypatch.setattr(api, "get_product_embeddings", lambda ids: {key: [1.0, 0.0] for key in ids})
    monkeypatch.setattr(pipeline, "USE_DUMMY_DATA", False)
    monkeypatch.setattr(api, "list_products", lambda **_: [product()])
    monkeypatch.setattr(api, "get_product_by_id", lambda _: product(9))
    with TestClient(api.app, headers={"X-Internal-Secret": "synthetic-test-secret", "X-Member-Id": "2"}) as client:
        yield client


@pytest.mark.parametrize("status,codes,expected,score", [
    ("UNKNOWN", [], "PENDING", 56),
    ("KNOWN_NONE", [], "SAFE", 80),
    ("KNOWN_LIST", ["BEEF"], "PENALIZED", 24),
    ("KNOWN_LIST", ["CHICKEN"], "SAFE", 80),
    ("KNOWN_NONE", ["BEEF"], "PENALIZED", 24),
    ("KNOWN_LIST", [], "PENDING", 56),
])
def test_home_api_status_score_and_reason(monkeypatch, synthetic_api, status, codes, expected, score):
    monkeypatch.setattr(api, "get_owned_pet_by_id", lambda *_: pet(status, codes))
    response = synthetic_api.post("/recommend/home", json={"pet_id": 1})
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["allergy_status"] == expected
    assert item["score"] == score
    assert item["matched_allergen"] == (["BEEF"] if expected == "PENALIZED" else [])
    if expected == "PENDING":
        assert "반려동물의 알레르기 정보를 확인" in item["reason_text"]
        assert "성분 정보 확인 중인 상품" not in item["reason_text"]


@pytest.mark.parametrize("status,codes,expected", [
    ("UNKNOWN", [], "PENDING"), ("KNOWN_NONE", [], "SAFE"),
    ("KNOWN_LIST", ["CHICKEN"], "SAFE"), ("KNOWN_LIST", [], "PENDING"),
])
def test_substitute_api_preserves_profile_status(monkeypatch, synthetic_api, status, codes, expected):
    monkeypatch.setattr(api, "get_owned_pet_by_id", lambda *_: pet(status, codes))
    response = synthetic_api.post("/recommend/substitute", json={"pet_id": 1, "base_product_id": 9})
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["allergy_status"] == expected
    assert item["matched_allergen"] == []
    if expected == "PENDING":
        assert "반려동물의 알레르기 정보를 확인" in item["reason_text"]


def test_substitute_still_excludes_explicit_conflicts(monkeypatch, synthetic_api):
    monkeypatch.setattr(api, "get_owned_pet_by_id", lambda *_: pet("KNOWN_LIST", ["BEEF"]))
    response = synthetic_api.post("/recommend/substitute", json={"pet_id": 1, "base_product_id": 9})
    assert response.status_code == 200
    assert response.json()["items"] == []


def test_api_missing_status_defaults_to_pending():
    assert api._item_from_recommendation({"product_id": 10}, {})["allergy_status"] == "PENDING"


@pytest.mark.parametrize("path,body", [
    ("/recommend/home", {"pet_id": 1}),
    ("/recommend/substitute", {"pet_id": 1, "base_product_id": 9}),
])
def test_unknown_product_keeps_product_pending_message(monkeypatch, synthetic_api, path, body):
    monkeypatch.setattr(api, "get_owned_pet_by_id", lambda *_: pet("KNOWN_NONE"))
    monkeypatch.setattr(api, "list_products", lambda **_: [{**product(), "allergen_flags": None}])
    response = synthetic_api.post(path, json=body)
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["allergy_status"] == "PENDING"
    assert "성분 정보 확인 중인 상품" in item["reason_text"]


def test_home_still_filters_species_and_ranks_by_penalized_score(monkeypatch, synthetic_api):
    monkeypatch.setattr(api, "get_owned_pet_by_id", lambda *_: pet("KNOWN_LIST", ["BEEF"]))
    monkeypatch.setattr(api, "list_products", lambda **_: [
        product(10), product(11, ["CHICKEN"]),
        {**product(12), "allergen_flags": None},
        {**product(13), "target_species": ["CAT"]},
    ])
    response = synthetic_api.post("/recommend/home", json={"pet_id": 1})
    assert response.status_code == 200
    assert [(item["product_id"], item["rank"], item["score"]) for item in response.json()["items"]] == [
        (11, 1, 80), (12, 2, 56), (10, 3, 24),
    ]
