"""다나와 → AI 입력 변환 어댑터 (1차 구현, 9/7)

목적:
- 다나와 CSV 1행 → api_nutrition.ProductIn 호환 dict
- DMB 변환 (수분 있을 때)
- 영양소 단위 UNKNOWN (수분 없거나 결측)
- Life-stage 한국어 keyword → AAFCO code
- Name 오염 처리 (가격만 → missing)
- pcode 보존
- 임의 상품명 생성 금지

기존 시드/Rule Engine/API/schema 무수정.

사용:
    from danawa_adapter import danawa_to_product
    p = danawa_to_product(pcode="32133548", name="238 280 원",
                          brand="한국마즈", description="...")
    product_in = p.to_product_in()
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any


# ============== 상수 ==============

# 영양소별 state (plan vocab; 1차 검증 보고서의 Q3 권장안과 정합)
STATE_KNOWN = "KNOWN"
STATE_UNKNOWN = "UNKNOWN"

# 분석 가능 여부 (overall)
# READY: 적어도 1개 이상의 영양소가 KNOWN → Rule Engine이 의미있는 score 산출 가능
# INSUFFICIENT_DATA: 모든 영양소 UNKNOWN → score=0, FE는 '데이터 부족'으로 표시
FEASIBILITY_READY = "READY"
FEASIBILITY_INSUFFICIENT = "INSUFFICIENT_DATA"

# 다나와 한국어 영양소명 → 내부 키 매핑
# NIAS에 매핑 있는 키: 단백질_g, 지방_g, 칼슘_g, 인_g
# NIAS에 매핑 없는 키 (현재 Rule Engine은 silent skip): 조회분_g, 조섬유_g, 수분_g
# → raw_guaranteed_analysis에 보존 (향후 시드 확장 시 활용)
DANAWA_NUTRIENT_MAP = {
    "조단백": "단백질_g",
    "조지방": "지방_g",
    "조섬유": "조섬유_g",
    "조회분": "조회분_g",
    "칼슘": "칼슘_g",
    "수분": "수분_g",
}

# '인'은 단어 경계 필요 (음식점, 인분, 임산부 등 오매칭 방지)
_IN_PATTERN = re.compile(r"(?<![가-힣A-Za-z])인\s*[:]?\s*([\d.]+)\s*%?")

# Life-stage 한국어 keyword → AAFCO code
# 우선순위: all > growth(임신/수유) > puppy/kitten > adult > senior
# (전연령은 모든 케이스에 매칭되므로 최우선)
LIFE_STAGE_RULES: list[tuple[str, list[str]]] = [
    ("all", ["전연령", "올라이프", "all life stages"]),
    ("growth", ["임신", "수유"]),
    ("puppy", ["퍼피", "자견"]),
    ("kitten", ["키튼"]),
    ("adult", ["어덜트", "성견", "성묘"]),
    ("senior", ["시니어", "노령"]),
]

AAFCO_LIFE_STAGE = {
    "puppy": "GROWTH_REPRODUCTION",
    "kitten": "GROWTH_REPRODUCTION",
    "growth": "GROWTH_REPRODUCTION",
    "adult": "MAINTENANCE",
    "senior": "MAINTENANCE",
    "all": "ALL_LIFE_STAGES",
}

# Name 오염 패턴 (가격만 있는 경우)
_NAME_PRICE_ONLY = re.compile(r"^[\s\d,\s원]+$")


# ============== 유틸 함수 ==============

def _parse_pct(text: str, keyword: str) -> float | None:
    """상세설명에서 '{keyword}: 숫자%' 추출 (단일 값)."""
    if not text:
        return None
    if keyword == "인":
        m = _IN_PATTERN.search(text)
    else:
        m = re.search(re.escape(keyword) + r"\s*[:]?\s*([\d.]+)\s*%?", text)
    if not m:
        return None
    try:
        v = float(m.group(1))
        return v
    except (ValueError, TypeError):
        return None


def _to_dmb(as_fed_pct: float, moisture_pct: float) -> float | None:
    """as-fed % → g/100g DMB (건물 기준)

    DMB = as_fed × 100 / (100 - moisture)
    수분이 0 이하거나 100 이상이면 환산 불가(None). 원래 as-fed 값을
    건물 기준 값처럼 반환하면 안 된다.
    """
    if moisture_pct <= 0 or moisture_pct >= 100:
        return None
    return as_fed_pct * 100.0 / (100.0 - moisture_pct)


def _is_valid_moisture(m: float | None) -> bool:
    return m is not None and 0 < m < 100


def _extract_ingredients(description: str) -> list[str]:
    """'주 단백질원: 닭고기, 칠면조고기' → ['닭고기', '칠면조고기']"""
    if not description:
        return []
    m = re.search(r"주\s*단백질원\s*[:]\s*([^/]+)", description)
    if not m:
        return []
    raw = m.group(1)
    parts = re.split(r"[,、·]+", raw)
    return [p.strip() for p in parts if p.strip()]


def _extract_category(description: str) -> str:
    """상세설명 → api_nutrition.ProductIn.category
    - '간식' → 'treat'
    - '영양제' / '보충제' / '분유' → 'supplement'
    - 기본 → 'food'
    """
    if not description:
        return "food"
    # segment 구분자 기준 위치 확인 (앞쪽 5개 segment 안에 키워드)
    # 잘못된 매칭 방지: 영양 정보의 '수분' 등은 제외하고 segment 헤더만
    segments = [s.strip() for s in description.split("/")[:6] if s.strip()]
    text = " / ".join(segments)
    if "간식" in text:
        return "treat"
    if "영양제" in text or "보충제" in text or "분유" in text:
        return "supplement"
    return "food"


def _extract_life_stage(description: str) -> tuple[str | None, str]:
    """한국어 keyword → AAFCO code + source.
    Returns: (aafco_code | None, source)
    source ∈ {"keyword", "none"}
    결정되지 않으면 (None, "none") — 기본값은 호출자에서 결정.
    """
    if not description:
        return None, "none"
    text = description
    for canonical, kws in LIFE_STAGE_RULES:
        for kw in kws:
            if kw.lower() in text.lower():
                return AAFCO_LIFE_STAGE[canonical], "keyword"
    return None, "none"


def _normalize_name(name: str, pcode: str) -> tuple[str, str]:
    """가격-only name → status="missing", pcode 보존.
    임의 상품명 생성 금지 — 원본 문자열 그대로 유지하고 status로만 표시.
    Returns: (name_kept, status) where status ∈ {"ok", "missing"}
    """
    if not name or not name.strip():
        # 빈 name → 원본 보존(빈 문자열), missing 마킹
        return name or "", "missing"
    if _NAME_PRICE_ONLY.match(name.strip()):
        return name, "missing"
    return name.strip(), "ok"


# ============== 메인 결과 ==============

@dataclass
class DanawaProduct:
    """변환 결과. to_product_in()으로 ProductIn dict 생성."""
    pcode: str
    name: str
    name_status: str  # "ok" | "missing"
    brand: str | None
    category: str

    ingredient_list: list[str] = field(default_factory=list)
    aafco_life_stage: str | None = None  # MAINTENANCE/GROWTH_REPRODUCTION/ALL_LIFE_STAGES/None
    aafco_life_stage_source: str = "none"  # "keyword" | "none"
    life_stage_status: str = STATE_UNKNOWN  # "KNOWN" | "UNKNOWN" (N1 수정)

    # Rule Engine 입력 (NIAS 호환 키 + DMB 적용)
    guaranteed_analysis: dict[str, float] = field(default_factory=dict)
    # 원본 (as-fed) — 보존용
    raw_guaranteed_analysis: dict[str, float] = field(default_factory=dict)
    # 영양소 단위 state
    nutrient_states: dict[str, str] = field(default_factory=dict)
    # 수분 (% or None)
    moisture_percent: float | None = None
    # 변환 중 발생한 주의사항
    warnings: list[str] = field(default_factory=list)

    # 분석 가능 여부 (READY / INSUFFICIENT_DATA) — score=0과 분리
    analysis_feasibility: str = FEASIBILITY_INSUFFICIENT

    def to_product_in(self) -> dict[str, Any]:
        """api_nutrition.ProductIn 호환 dict + Adapter metadata.

        N1 수정: aafco_life_stage가 None일 때 더 이상 MAINTENANCE로 강제하지 않음.
        - aafco_life_stage = None (실제 값 없음)
        - life_stage_status = "UNKNOWN" (다운스트림 인지용)
        - warnings에 명시 (PR-C API에서 이 status로 FE 표시 결정)

        생애주기가 비어 있으면 None을 유지한다. 임시 MAINTENANCE 기본값은
        UNKNOWN을 정상 판정처럼 보이게 하므로 사용하지 않는다.
        """
        warnings = list(self.warnings)
        if self.life_stage_status == STATE_UNKNOWN:
            warnings.append("AAFCO life stage 미상 — 적합성 판정 보류")

        nutrient_code_map = {
            "단백질_g": "CRUDE_PROTEIN",
            "지방_g": "CRUDE_FAT",
            "조섬유_g": "CRUDE_FIBER",
            "수분_g": "MOISTURE",
            "칼슘_g": "CALCIUM",
            "인_g": "PHOSPHORUS",
        }
        nutrition_items = [
            {
                "nutrient_code": nutrient_code,
                "value": self.raw_guaranteed_analysis.get(legacy_key),
                "unit": "PERCENT",
                "basis": "AS_FED",
                "source": "DANAWA_LABEL",
            }
            for legacy_key, nutrient_code in nutrient_code_map.items()
            if legacy_key in self.raw_guaranteed_analysis
        ]

        return {
            "id": self.pcode,
            "name": self.name,
            "category": self.category,
            "ingredient_list": self.ingredient_list,
            "guaranteed_analysis": self.guaranteed_analysis,
            "aafco_life_stage": self.aafco_life_stage,
            "nutrition_items": nutrition_items,
            "calorie_kcal_per_kg": None,
            "product_attributes": {},
            # Adapter metadata (PR-C에서 활용)
            "life_stage_status": self.life_stage_status,
            "analysis_feasibility": self.analysis_feasibility,
            "nutrient_states": dict(self.nutrient_states),
            "warnings": warnings,
        }


def danawa_to_product(pcode: str, name: str, brand: str, description: str) -> DanawaProduct:
    """다나와 row → DanawaProduct 변환 (메인 진입점).

    Args:
        pcode: 다나와 pcode (unique 식별자)
        name: 다나와 name (오염 가능)
        brand: 다나와 brand
        description: 다나와 상세설명 (한국어 자유 텍스트)

    Returns:
        DanawaProduct (to_product_in()으로 API 입력 변환 가능)
    """
    p = DanawaProduct(
        pcode=pcode,
        name="",
        name_status="",
        brand=brand or None,
        category="food",
    )

    # 1) name 정규화
    p.name, p.name_status = _normalize_name(name, pcode)

    # 2) category
    p.category = _extract_category(description)

    # 3) ingredient_list
    p.ingredient_list = _extract_ingredients(description)

    # 4) life-stage (N1 수정: None 유지, status 명시)
    p.aafco_life_stage, p.aafco_life_stage_source = _extract_life_stage(description)
    p.life_stage_status = STATE_KNOWN if p.aafco_life_stage is not None else STATE_UNKNOWN

    # 5) 수분 (DMB denominator)
    moisture = _parse_pct(description, "수분")
    p.moisture_percent = moisture
    moisture_valid = _is_valid_moisture(moisture)

    if not moisture_valid:
        p.warnings.append("수분 정보 없음 — DMB 변환 불가, 영양소 단위 UNKNOWN")

    # 6) 7개 영양소
    # '수분'은 DMB 변환의 denominator이지 변환 대상이 아님
    DMB_TARGETS = {k for k in DANAWA_NUTRIENT_MAP.values() if k != "수분_g"}
    for kw, key in DANAWA_NUTRIENT_MAP.items():
        val = _parse_pct(description, kw)
        if val is None:
            p.nutrient_states[key] = STATE_UNKNOWN
            continue
        # 원본 보존
        p.raw_guaranteed_analysis[key] = val
        # DMB 변환 (수분 자체는 변환 제외)
        if key == "수분_g":
            # 수분은 as-fed 그대로 (DMB 기준도 같은 %이므로 변환 불필요)
            p.guaranteed_analysis[key] = val
            p.nutrient_states[key] = STATE_KNOWN
        elif moisture_valid:
            dmb = _to_dmb(val, moisture)  # type: ignore[arg-type]
            p.guaranteed_analysis[key] = round(dmb, 2)
            p.nutrient_states[key] = STATE_KNOWN
        else:
            p.nutrient_states[key] = STATE_UNKNOWN

    # 7) '인' (단어 경계 별도 처리)
    in_val = _parse_pct(description, "인")
    if in_val is not None:
        p.raw_guaranteed_analysis["인_g"] = in_val
        if moisture_valid:
            dmb = _to_dmb(in_val, moisture)  # type: ignore[arg-type]
            p.guaranteed_analysis["인_g"] = round(dmb, 2)
            p.nutrient_states["인_g"] = STATE_KNOWN
        else:
            p.nutrient_states["인_g"] = STATE_UNKNOWN
    else:
        p.nutrient_states["인_g"] = STATE_UNKNOWN

    # 8) 분석 가능 여부 (N1 수정 + score=0 분리)
    # - guaranteed_analysis이 1개 이상 비어있지 않으면 READY
    # - 모두 비어있으면 INSUFFICIENT_DATA (score=0이 될 예정)
    if p.guaranteed_analysis:
        p.analysis_feasibility = FEASIBILITY_READY
    else:
        p.analysis_feasibility = FEASIBILITY_INSUFFICIENT
        p.warnings.append("분석 가능한 영양소 없음 — INSUFFICIENT_DATA")

    return p


# ============== batch loader (P0-1, 9/10) =============

def danawa_load_products(json_path: str) -> list[DanawaProduct]:
    """seed_25_danawa JSON 1개 → DanawaProduct list 변환 (batch entry point).

    입력 JSON 구조 (v1, 실측):
        {
          "source": "Danawa dog food crawler",
          "count": 1,
          "items": [
            {
              "category_code": 19238134,
              "total": 298,
              "success": 298,
              "failed": 0,
              "products": [
                {"pcode": "...", "name": "...", "brand": "...",
                 "description": "...", ...}, ...
              ]
            }
          ]
        }

    처리:
    - JSON load → items[*] 순회 → 각 item의 products[*]를 danawa_to_product()로 변환
    - pcode 누락 row는 skip (skipped 카운트는 호출자가 검증)
    - 변환 중 예외는 row 단위로 skip (전체 batch 중단 X)

    예외:
    - FileNotFoundError: json_path 부재 시 open()이 그대로 발생
    - ValueError: 'items' 키 부재/빈 리스트
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items")
    if not items:
        raise ValueError(
            f"Invalid seed JSON: 'items' missing or empty in {json_path}"
        )

    products_out: list[DanawaProduct] = []
    skipped = 0
    for item in items:
        products = item.get("products")
        if not products:
            # 'products' 키 부재/빈 리스트 → 해당 item skip
            continue
        for p in products:
            pcode = p.get("pcode")
            if not pcode:
                skipped += 1
                continue
            try:
                products_out.append(
                    danawa_to_product(
                        pcode=str(pcode),
                        name=str(p.get("name") or ""),
                        brand=str(p.get("brand") or ""),
                        description=str(p.get("description") or ""),
                    )
                )
            except Exception:
                # row 단위 실패는 batch 전체를 중단시키지 않음
                skipped += 1
                continue
    return products_out


# ============== self-test ==============

if __name__ == "__main__":
    # 빠른 자체 확인
    samples = [
        # 정상 dog food (수분 있음)
        ("32133548", "238 280 원", "한국마즈",
         "강아지 전용 / 사료 / 어덜트 / 건식 / 주 단백질원: 닭고기, 칠면조고기 / 조단백: 19% / 조지방: 12% / 조회분: 6.8% / 수분: 11%"),
        # 정상 cat food (수분 있음)
        ("1802793", "로얄캐닌 캣 인도어 10kg", "로얄캐닌",
         "고양이 전용 / 사료 / 어덜트 / 건식 / 주 단백질원: 닭고기 / 조단백: 25% / 조지방: 11% / 조회분: 8.2% / 수분: 5.5%"),
        # 습식 사료 (시저)
        ("7238965", "시저 11세이상 쇠고기 100g", "마즈",
         "강아지 전용 / 사료 / 시니어 / 습식 / 주 단백질원: 쇠고기 / 조단백: 5.5% / 조지방: 3.5% / 수분: 89%"),
        # 전연령
        ("5664799", "캐츠랑 전연령 20kg", "캐츠랑",
         "고양이 전용 / 사료 / 전연령 / 건식 / 조단백: 31% / 조지방: 11% / 조회분: 12% / 수분: 10%"),
        # 퍼피
        ("sample_puppy", "퍼피 사료 1kg", "X",
         "강아지 전용 / 사료 / 퍼피 / 건식 / 조단백: 28% / 지방: 16% / 수분: 10%"),
        # 키튼
        ("sample_kitten", "키튼 사료 1kg", "Y",
         "고양이 전용 / 사료 / 키튼 / 건식 / 조단백: 32% / 지방: 14% / 수분: 8%"),
        # 임신/수유
        ("sample_preg", "임신 수유 사료 1kg", "Z",
         "강아지 전용 / 사료 / 임신·수유 / 건식 / 조단백: 30% / 지방: 18% / 수분: 10%"),
        # 영양소 거의 없음
        ("108425402", "시저 미티 팩", "마즈",
         "강아지 전용 / 사료 / 습식 / 캔 / 파우치"),
        # 빈 description
        ("empty", "Test 1kg", "X", ""),
    ]

    print("=== danawa_adapter self-test ===\n")
    for pcode, name, brand, desc in samples:
        p = danawa_to_product(pcode, name, brand, desc)
        print(f"[{pcode}] {name[:25]}")
        print(f"  name_status={p.name_status}, category={p.category}")
        print(f"  aafco_life_stage={p.aafco_life_stage} (source={p.aafco_life_stage_source}, status={p.life_stage_status})")
        print(f"  analysis_feasibility={p.analysis_feasibility}")
        print(f"  ingredients={p.ingredient_list}")
        print(f"  moisture={p.moisture_percent}")
        print(f"  DMB guaranteed: {p.guaranteed_analysis}")
        print(f"  raw (as-fed): {p.raw_guaranteed_analysis}")
        print(f"  states: {p.nutrient_states}")
        if p.warnings:
            print(f"  warnings: {p.warnings}")
        # to_product_in() check
        pi = p.to_product_in()
        print(f"  to_product_in: aafco={pi['aafco_life_stage']}, life_stage_status={pi['life_stage_status']}, feasibility={pi['analysis_feasibility']}")
        print()
