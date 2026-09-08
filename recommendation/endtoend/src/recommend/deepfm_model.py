# -*- coding: utf-8 -*-
"""
DeepFM 모델 (raw PyTorch 직접 구현) + FeatureEncoder.

[변경 이력]
bcs_norm, age_group_ordinal을 DENSE_FIELDS에 추가 (deepfm_features.py에서
순서형 값을 0~1 정규화한 dense feature로 새로 만든 것과 짝을 맞춤).
dense 벡터 차원이 10 -> 12로 늘어나지만, dense_linear/DNN 입력 차원이
len(DENSE_FIELDS) 기준으로 동적으로 계산되므로 이 리스트만 수정하면 된다.
"""

import os
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature 스키마 정의
# ---------------------------------------------------------------------------

SPARSE_FIELD_VOCABS = {
    "species": ["DOG", "CAT"],
    "sex": ["MALE", "FEMALE"],
    "age_group": ["GROWTH", "ADULT", "SENIOR"],
    "breed_size": ["SMALL", "MEDIUM", "LARGE"],
    "bcs": ["1", "2", "3", "4", "5"],
    "breed": None,  # 품종 마스터 미확정 -> 학습 데이터에서 동적으로 구축 + UNKNOWN 처리
    "category_code": ["FOOD", "SUPPLEMENT", "TREAT"],
    "subcategory_code": [
        "DRY_FOOD", "WET_FOOD", "FREEZE_DRIED_FOOD", "BAKED_FOOD",
        "POWDER_SUPPLEMENT", "LIQUID_SUPPLEMENT", "CHEWABLE_SUPPLEMENT", "TABLET_SUPPLEMENT",
        "JERKY_TREAT", "WET_TREAT", "FREEZE_DRIED_TREAT", "BISCUIT_TREAT",
    ],
    "target_breed_size": ["SMALL", "MEDIUM", "LARGE"],
    "target_age_group": ["GROWTH", "ADULT", "SENIOR"],
}

MULTIHOT_FIELD_SIZES = {
    "allergy_codes": 120,
    "concerns": 95,
    "target_species": 2,
    "allergen_flags": 120,
}

# dense 필드 (pet_features.dense + product_features.dense 순서 고정)
# bcs_norm, age_group_ordinal -- 순서형 값의 순서/거리 정보를 명시적으로 전달하기 위해 추가
DENSE_FIELDS = [
    "age_months_norm", "weight_norm", "neutered", "bcs_norm", "age_group_ordinal",
    "price_norm", "palatability_score", "digestion_score",
    "skin_coat_score", "vitality_weight_score", "allergic_reaction_score", "price_value_score",
]

UNKNOWN_TOKEN = "<UNK>"


class FeatureEncoder:
    def __init__(self):
        self.sparse_vocabs = {}
        for field, vocab in SPARSE_FIELD_VOCABS.items():
            if vocab is not None:
                self.sparse_vocabs[field] = {UNKNOWN_TOKEN: 0, **{v: i + 1 for i, v in enumerate(vocab)}}
            else:
                self.sparse_vocabs[field] = {UNKNOWN_TOKEN: 0}

    def fit_dynamic_vocab(self, field: str, values: list):
        vocab = self.sparse_vocabs[field]
        for v in values:
            v = str(v)
            if v not in vocab:
                vocab[v] = len(vocab)

    def vocab_size(self, field: str) -> int:
        return len(self.sparse_vocabs[field])

    def _encode_sparse(self, field: str, value) -> int:
        value = str(value) if value is not None else UNKNOWN_TOKEN
        return self.sparse_vocabs[field].get(value, 0)

    def encode(self, sample: dict) -> dict:
        pet_f = sample["pet_features"]
        prod_f = sample["product_features"]

        sparse_out = {}
        for field, value in pet_f["sparse"].items():
            if field in self.sparse_vocabs:
                sparse_out[field] = self._encode_sparse(field, value)
        for field, value in prod_f["sparse"].items():
            if field in self.sparse_vocabs:
                sparse_out[field] = self._encode_sparse(field, value)

        multihot_out = {}
        for field, vec in pet_f["multi_hot"].items():
            multihot_out[field] = vec
        for field, vec in prod_f["multi_hot"].items():
            multihot_out[field] = vec

        dense_vals = []
        for field in DENSE_FIELDS:
            if field in pet_f["dense"]:
                dense_vals.append(float(pet_f["dense"][field]))
            elif field in prod_f["dense"]:
                dense_vals.append(float(prod_f["dense"][field]))
            else:
                dense_vals.append(0.0)

        return {
            "sparse": sparse_out,
            "multi_hot": multihot_out,
            "dense": dense_vals,
            "label": sample.get("label"),
        }

    @classmethod
    def from_saved_vocab(cls, vocab_path: str) -> "FeatureEncoder":
        import json
        encoder = cls.__new__(cls)
        with open(vocab_path, "r", encoding="utf-8") as f:
            encoder.sparse_vocabs = json.load(f)
        return encoder

    def collate(self, encoded_samples: list) -> dict:
        batch = {"sparse": {}, "multi_hot": {}, "dense": None, "label": None}

        for field in SPARSE_FIELD_VOCABS:
            batch["sparse"][field] = torch.tensor(
                [s["sparse"].get(field, 0) for s in encoded_samples], dtype=torch.long
            )

        for field, size in MULTIHOT_FIELD_SIZES.items():
            batch["multi_hot"][field] = torch.tensor(
                [s["multi_hot"][field] for s in encoded_samples], dtype=torch.float32
            )

        batch["dense"] = torch.tensor(
            [s["dense"] for s in encoded_samples], dtype=torch.float32
        )

        labels = [s["label"] for s in encoded_samples]
        if all(l is not None for l in labels):
            batch["label"] = torch.tensor(labels, dtype=torch.float32)

        return batch


class DeepFM(nn.Module):
    def __init__(self, encoder: FeatureEncoder, embed_dim: int = 8, dnn_hidden: tuple = (64, 32), dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim

        self.sparse_linear = nn.ModuleDict({
            field: nn.Embedding(encoder.vocab_size(field), 1)
            for field in SPARSE_FIELD_VOCABS
        })
        self.sparse_embed = nn.ModuleDict({
            field: nn.Embedding(encoder.vocab_size(field), embed_dim)
            for field in SPARSE_FIELD_VOCABS
        })

        self.multihot_linear = nn.ModuleDict({
            field: nn.Linear(size, 1, bias=False)
            for field, size in MULTIHOT_FIELD_SIZES.items()
        })
        self.multihot_embed_table = nn.ParameterDict({
            field: nn.Parameter(torch.randn(size, embed_dim) * 0.01)
            for field, size in MULTIHOT_FIELD_SIZES.items()
        })

        n_dense = len(DENSE_FIELDS)
        self.dense_linear = nn.Linear(n_dense, 1)

        self.bias = nn.Parameter(torch.zeros(1))

        # 임베딩 초기화: 기본값(표준편차 1)이면 FM 2차항에서 값이 크게 증폭되어
        # sigmoid가 포화되고 학습이 안 되는 문제가 생기므로 작은 표준편차로 재초기화.
        for emb in list(self.sparse_linear.values()) + list(self.sparse_embed.values()):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for lin in self.multihot_linear.values():
            nn.init.normal_(lin.weight, mean=0.0, std=0.01)

        n_fields = len(SPARSE_FIELD_VOCABS) + len(MULTIHOT_FIELD_SIZES)
        dnn_input_dim = n_fields * embed_dim + n_dense

        layers = []
        prev_dim = dnn_input_dim
        for hidden in dnn_hidden:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, batch: dict) -> torch.Tensor:
        sparse = batch["sparse"]
        multi_hot = batch["multi_hot"]
        dense = batch["dense"]

        field_embeds = []
        linear_terms = []

        for field in SPARSE_FIELD_VOCABS:
            idx = sparse[field]
            linear_terms.append(self.sparse_linear[field](idx).squeeze(-1))
            field_embeds.append(self.sparse_embed[field](idx))

        for field in MULTIHOT_FIELD_SIZES:
            vec = multi_hot[field]
            linear_terms.append(self.multihot_linear[field](vec).squeeze(-1))
            active_count = vec.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (vec @ self.multihot_embed_table[field]) / active_count
            field_embeds.append(pooled)

        first_order = torch.stack(linear_terms, dim=1).sum(dim=1, keepdim=True)
        first_order = first_order + self.dense_linear(dense) + self.bias

        stacked = torch.stack(field_embeds, dim=1)
        sum_then_square = stacked.sum(dim=1) ** 2
        square_then_sum = (stacked ** 2).sum(dim=1)
        fm_second_order = 0.5 * (sum_then_square - square_then_sum).sum(dim=1, keepdim=True)

        flat_embeds = stacked.view(stacked.size(0), -1)
        dnn_input = torch.cat([flat_embeds, dense], dim=1)
        deep_out = self.dnn(dnn_input)

        logit = first_order + fm_second_order + deep_out
        return torch.sigmoid(logit).squeeze(-1)


def load_deepfm(model_dir: str) -> tuple:
    import json

    encoder = FeatureEncoder.from_saved_vocab(os.path.join(model_dir, "feature_encoder.json"))

    with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    model = DeepFM(encoder, embed_dim=config["embed_dim"], dnn_hidden=tuple(config["dnn_hidden"]))
    model.load_state_dict(torch.load(os.path.join(model_dir, "deepfm_model.pt"), map_location="cpu"))
    model.eval()

    return encoder, model