# -*- coding: utf-8 -*-
"""
DeepFM 모델 (raw PyTorch 직접 구현) + FeatureEncoder.

deepctr-torch 대신 직접 구현한 이유:
- 최신 pandas/numpy 환경에서의 호환성 리스크 회피
- src/features/deepfm_features.py가 만드는 sparse/dense/multi-hot 딕셔너리 구조를
  그대로 받아 쓰기 위함 (라이브러리 전용 Feat 객체로 재변환하는 과정 불필요)

구조:
- 1차항(Linear): 모든 필드의 1차 가중치 합
- 2차항(FM): 필드 임베딩 간 pairwise interaction (표준 FM 공식)
- Deep항(DNN): 전체 필드 임베딩 concat + dense 값을 MLP에 통과
- 최종: sigmoid(Linear + FM + Deep)

multi-hot 필드(allergy_codes, concerns, target_species, allergen_flags)는
각 항목을 개별 임베딩하지 않고, 활성화된 항목들의 임베딩을 평균 풀링해서
하나의 필드 임베딩으로 압축한다 (표준적인 방식).
"""

import os
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature 스키마 정의
# (src/features/deepfm_features.py의 build_interaction_features() 출력 구조와 정확히 대응)
# ---------------------------------------------------------------------------

# sparse 필드: field_name -> 고정 vocab (알고 있는 경우) 또는 None (데이터에서 동적으로 구축)
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
    "target_breed_size": ["SMALL", "MEDIUM", "LARGE"],  # None -> UNKNOWN 처리
    "target_age_group": ["GROWTH", "ADULT", "SENIOR"],  # None -> UNKNOWN 처리
}

# multi-hot 필드: field_name -> vocab 크기 (allergen_master/concern_master 참조)
MULTIHOT_FIELD_SIZES = {
    "allergy_codes": 120,
    "concerns": 95,
    "target_species": 2,
    "allergen_flags": 120,
}

# dense 필드 (pet_features.dense + product_features.dense 순서 고정)
DENSE_FIELDS = [
    "age_months_norm", "weight_norm", "neutered",
    "price_norm", "palatability_score", "digestion_score",
    "skin_coat_score", "vitality_weight_score", "allergic_reaction_score", "price_value_score",
]

UNKNOWN_TOKEN = "<UNK>"


class FeatureEncoder:
    """
    build_interaction_features() 출력(중첩 딕셔너리)을 모델 입력 텐서로 변환.
    sparse 필드는 정수 인덱스로, multi-hot은 0/1 벡터 그대로, dense는 float 벡터로.
    """

    def __init__(self):
        self.sparse_vocabs = {}  # field_name -> {value: index}
        for field, vocab in SPARSE_FIELD_VOCABS.items():
            if vocab is not None:
                # 0번 인덱스는 UNKNOWN용으로 예약
                self.sparse_vocabs[field] = {UNKNOWN_TOKEN: 0, **{v: i + 1 for i, v in enumerate(vocab)}}
            else:
                self.sparse_vocabs[field] = {UNKNOWN_TOKEN: 0}  # 동적 구축 대상

    def fit_dynamic_vocab(self, field: str, values: list):
        """breed처럼 고정 vocab이 없는 필드를 학습 데이터에서 구축."""
        vocab = self.sparse_vocabs[field]
        for v in values:
            v = str(v)
            if v not in vocab:
                vocab[v] = len(vocab)

    def vocab_size(self, field: str) -> int:
        return len(self.sparse_vocabs[field])

    def _encode_sparse(self, field: str, value) -> int:
        value = str(value) if value is not None else UNKNOWN_TOKEN
        return self.sparse_vocabs[field].get(value, 0)  # 미확인 값은 UNKNOWN(0)

    def encode(self, sample: dict) -> dict:
        """
        sample: build_interaction_features()의 반환값 1건.
        반환: {"sparse": {field: int}, "multi_hot": {field: [0/1,...]}, "dense": [float,...], "label": int or None}
        """
        pet_f = sample["pet_features"]
        prod_f = sample["product_features"]

        sparse_out = {}
        for field, value in pet_f["sparse"].items():
            sparse_out[field] = self._encode_sparse(field, value)
        for field, value in prod_f["sparse"].items():
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
        """학습 시 저장된 feature_encoder.json으로부터 인코더를 복원 (추론 시 사용)."""
        import json
        encoder = cls.__new__(cls)  # __init__을 건너뛰고 vocab을 직접 주입
        with open(vocab_path, "r", encoding="utf-8") as f:
            encoder.sparse_vocabs = json.load(f)
        return encoder

    def collate(self, encoded_samples: list) -> dict:
        """encode()로 변환된 샘플 여러 개를 배치 텐서로 묶는다."""
        batch = {"sparse": {}, "multi_hot": {}, "dense": None, "label": None}

        for field in SPARSE_FIELD_VOCABS:
            batch["sparse"][field] = torch.tensor(
                [s["sparse"][field] for s in encoded_samples], dtype=torch.long
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


# ---------------------------------------------------------------------------
# DeepFM 모델
# ---------------------------------------------------------------------------

class DeepFM(nn.Module):
    def __init__(self, encoder: FeatureEncoder, embed_dim: int = 8, dnn_hidden: tuple = (64, 32), dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim

        # --- sparse 필드: 1차항(dim=1) + 2차/deep용 임베딩(dim=embed_dim) ---
        self.sparse_linear = nn.ModuleDict({
            field: nn.Embedding(encoder.vocab_size(field), 1)
            for field in SPARSE_FIELD_VOCABS
        })
        self.sparse_embed = nn.ModuleDict({
            field: nn.Embedding(encoder.vocab_size(field), embed_dim)
            for field in SPARSE_FIELD_VOCABS
        })

        # --- multi-hot 필드: 1차항(vocab_size -> 1, bias 없는 Linear) + 임베딩(평균 풀링) ---
        self.multihot_linear = nn.ModuleDict({
            field: nn.Linear(size, 1, bias=False)
            for field, size in MULTIHOT_FIELD_SIZES.items()
        })
        self.multihot_embed_table = nn.ParameterDict({
            field: nn.Parameter(torch.randn(size, embed_dim) * 0.01)
            for field, size in MULTIHOT_FIELD_SIZES.items()
        })

        # --- dense 필드: 1차항(선형) ---
        n_dense = len(DENSE_FIELDS)
        self.dense_linear = nn.Linear(n_dense, 1)

        self.bias = nn.Parameter(torch.zeros(1))

        # 임베딩 초기화: nn.Embedding 기본값(표준정규분포, std=1)을 그대로 두면
        # FM 2차항(여러 필드 임베딩의 합을 제곱)에서 값이 크게 증폭되어
        # sigmoid가 0/1에 극단적으로 포화되고 학습이 안 되는 문제가 생긴다.
        # 표준 DeepFM 구현처럼 작은 표준편차로 재초기화한다.
        for emb in list(self.sparse_linear.values()) + list(self.sparse_embed.values()):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for lin in self.multihot_linear.values():
            nn.init.normal_(lin.weight, mean=0.0, std=0.01)

        # --- Deep 파트 입력 차원: (sparse 필드 수 + multi-hot 필드 수) * embed_dim + dense 원본값 ---
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
        dense = batch["dense"]  # (B, n_dense)

        field_embeds = []  # 각 원소 shape: (B, embed_dim)
        linear_terms = []  # 각 원소 shape: (B, 1)

        for field in SPARSE_FIELD_VOCABS:
            idx = sparse[field]  # (B,)
            linear_terms.append(self.sparse_linear[field](idx).squeeze(-1))  # (B,)
            field_embeds.append(self.sparse_embed[field](idx))  # (B, embed_dim)

        for field in MULTIHOT_FIELD_SIZES:
            vec = multi_hot[field]  # (B, vocab_size), 0/1
            linear_terms.append(self.multihot_linear[field](vec).squeeze(-1))  # (B,)
            # 평균 풀링: 활성화된 항목의 임베딩 평균 (활성 항목 없으면 0벡터)
            active_count = vec.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
            pooled = (vec @ self.multihot_embed_table[field]) / active_count  # (B, embed_dim)
            field_embeds.append(pooled)

        # --- 1차항 합산: 모든 필드의 (B,) 선형항 + dense 선형항 + bias ---
        first_order = torch.stack(linear_terms, dim=1).sum(dim=1, keepdim=True)  # (B, 1)
        first_order = first_order + self.dense_linear(dense) + self.bias  # (B, 1)

        # --- 2차항(FM): 0.5 * [(sum v_i)^2 - sum(v_i^2)], 필드 임베딩 기준 ---
        stacked = torch.stack(field_embeds, dim=1)  # (B, n_fields, embed_dim)
        sum_then_square = stacked.sum(dim=1) ** 2          # (B, embed_dim)
        square_then_sum = (stacked ** 2).sum(dim=1)         # (B, embed_dim)
        fm_second_order = 0.5 * (sum_then_square - square_then_sum).sum(dim=1, keepdim=True)  # (B,1)

        # --- Deep 파트 ---
        flat_embeds = stacked.view(stacked.size(0), -1)  # (B, n_fields*embed_dim)
        dnn_input = torch.cat([flat_embeds, dense], dim=1)
        deep_out = self.dnn(dnn_input)  # (B,1)

        logit = first_order + fm_second_order + deep_out
        return torch.sigmoid(logit).squeeze(-1)  # (B,)


def load_deepfm(model_dir: str) -> tuple:
    """
    저장된 모델(deepfm_model.pt) + vocab(feature_encoder.json) + config(model_config.json)로부터
    FeatureEncoder와 DeepFM 모델을 복원한다.
    반환: (encoder, model)
    """
    import json

    encoder = FeatureEncoder.from_saved_vocab(os.path.join(model_dir, "feature_encoder.json"))

    with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    model = DeepFM(encoder, embed_dim=config["embed_dim"], dnn_hidden=tuple(config["dnn_hidden"]))
    model.load_state_dict(torch.load(os.path.join(model_dir, "deepfm_model.pt"), map_location="cpu"))
    model.eval()

    return encoder, model