"""danawa_adapter 단위 테스트 (9/7)

18개 케이스:
- 파싱 7 (영양소 7종)
- DMB 3 (정상, 결측, 0%)
- Life-stage 5 (어덜트/퍼피/키튼/전연령/임신·수유/시니어)
- Name 3 (정상, 가격만, 빈문자열)
- Integration 2 (전체 파이프라인)

실행: python3 scripts/nutrition/test_danawa_adapter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 동일 폴더의 danawa_adapter 임포트
sys.path.insert(0, str(Path(__file__).resolve().parent))
from danawa_adapter import (  # noqa: E402
    danawa_to_product,
    _parse_pct,
    _to_dmb,
    _extract_life_stage,
    _extract_ingredients,
    _extract_category,
    _normalize_name,
    STATE_KNOWN,
    STATE_UNKNOWN,
    FEASIBILITY_READY,
    FEASIBILITY_INSUFFICIENT,
)


PASS = 0
FAIL = 0
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        RESULTS.append((name, True, detail))
    else:
        FAIL += 1
        RESULTS.append((name, False, detail))


# ============ 1) 파싱 테스트 ============
def test_parsing():
    print("\n--- 1) 파싱 (7 영양소) ---")
    desc = "조단백: 19% / 조지방: 12% / 조섬유: 2.9% / 조회분: 6.8% / 칼슘: 0.55% / 인: 0.3% / 수분: 11%"
    check("조단백 파싱", _parse_pct(desc, "조단백") == 19.0)
    check("조지방 파싱", _parse_pct(desc, "조지방") == 12.0)
    check("조섬유 파싱", _parse_pct(desc, "조섬유") == 2.9)
    check("조회분 파싱", _parse_pct(desc, "조회분") == 6.8)
    check("칼슘 파싱", _parse_pct(desc, "칼슘") == 0.55)
    check("인 파싱 (단어 경계)", _parse_pct(desc, "인") == 0.3)
    check("수분 파싱", _parse_pct(desc, "수분") == 11.0)
    # '인' 단어 경계 테스트
    check("'인분' 오매칭 안 됨",
          _parse_pct("닭고기, 인분 0.5%", "인") is None)
    check("'음식점' 오매칭 안 됨",
          _parse_pct("음식점 특가", "인") is None)


# ============ 2) DMB 변환 ============
def test_dmb():
    print("\n--- 2) DMB 변환 ---")
    # 정상: 19%, 수분 11% → 21.35
    check("DMB 정상 (19% / 11%)",
          abs(_to_dmb(19.0, 11.0) - 21.35) < 0.01)
    # 경계: moisture 0/100 → 환산 불가 (원래 값 fallback 금지)
    check("DMB moisture=0 is unavailable",
          _to_dmb(20.0, 0) is None)
    check("DMB moisture=100 is unavailable",
          _to_dmb(20.0, 100) is None)
    # 습식: 5.5%, 수분 89% → 50
    check("DMB 습식 (5.5% / 89%)",
          abs(_to_dmb(5.5, 89.0) - 50.0) < 0.01)


# ============ 3) Life-stage ============
def test_lifestage():
    print("\n--- 3) Life-stage ---")
    check("'어덜트' → MAINTENANCE",
          _extract_life_stage("강아지 전용 / 사료 / 어덜트 / 건식") == ("MAINTENANCE", "keyword"))
    check("'퍼피' → GROWTH_REPRODUCTION",
          _extract_life_stage("강아지 전용 / 사료 / 퍼피 / 건식") == ("GROWTH_REPRODUCTION", "keyword"))
    check("'키튼' → GROWTH_REPRODUCTION",
          _extract_life_stage("고양이 전용 / 사료 / 키튼 / 건식") == ("GROWTH_REPRODUCTION", "keyword"))
    check("'전연령' → ALL_LIFE_STAGES (우선순위 1위)",
          _extract_life_stage("고양이 전용 / 사료 / 전연령 / 건식") == ("ALL_LIFE_STAGES", "keyword"))
    check("'시니어' → MAINTENANCE",
          _extract_life_stage("강아지 전용 / 사료 / 시니어 / 건식") == ("MAINTENANCE", "keyword"))
    check("'임신·수유' → GROWTH_REPRODUCTION",
          _extract_life_stage("강아지 전용 / 사료 / 임신·수유 / 건식") == ("GROWTH_REPRODUCTION", "keyword"))
    check("'성견' → MAINTENANCE",
          _extract_life_stage("강아지 전용 / 사료 / 성견 / 건식") == ("MAINTENANCE", "keyword"))
    check("키워드 없음 → (None, 'none')",
          _extract_life_stage("강아지 전용 / 사료 / 건식") == (None, "none"))


# ============ 4) Name ============
def test_name():
    print("\n--- 4) Name 정규화 ---")
    check("정상 name",
          _normalize_name("로얄캐닌 독 어덜트 8kg", "12345") == ("로얄캐닌 독 어덜트 8kg", "ok"))
    check("가격-only name → missing",
          _normalize_name("238 280 원", "12345") == ("238 280 원", "missing"))
    check("빈 name → missing",
          _normalize_name("", "12345") == ("", "missing"))
    check("숫자만 name → missing",
          _normalize_name("123456", "12345") == ("123456", "missing"))


# ============ 5) Category ============
def test_category():
    print("\n--- 5) Category ---")
    check("'간식' → treat",
          _extract_category("강아지 전용 / 사료 / 어덜트 / 간식") == "treat")
    check("'영양제' → supplement",
          _extract_category("강아지 전용 / 사료 / 영양제 / 어덜트") == "supplement")
    check("'분유' → supplement",
          _extract_category("강아지 전용 / 분유·우유") == "supplement")
    check("기본 → food",
          _extract_category("강아지 전용 / 사료 / 어덜트 / 건식") == "food")


# ============ 6) Integration ============
def test_integration():
    print("\n--- 6) Integration (전체 파이프라인) ---")
    # 정상 dog food
    p = danawa_to_product(
        pcode="32133548", name="238 280 원", brand="한국마즈",
        description="강아지 전용 / 사료 / 어덜트 / 건식 / 주 단백질원: 닭고기, 칠면조고기 / 조단백: 19% / 조지방: 12% / 수분: 11%"
    )
    check("Integration: name missing",
          p.name_status == "missing")
    check("Integration: aafco_life_stage 어덜트→MAINTENANCE",
          p.aafco_life_stage == "MAINTENANCE")
    check("Integration: DMB 단백질 19→21.35",
          abs(p.guaranteed_analysis["단백질_g"] - 21.35) < 0.01)
    check("Integration: ingredients 2개",
          len(p.ingredient_list) == 2)
    check("Integration: 영양소 5개 UNKNOWN (조섬유/조회분/칼슘/인/수분 등 데이터 없음)",
          p.nutrient_states["조섬유_g"] == STATE_UNKNOWN)

    # 수분 없는 케이스 → 모든 영양소 UNKNOWN
    p2 = danawa_to_product(
        pcode="4266771", name="오리젠", brand="오리젠",
        description="강아지 전용 / 사료 / 어덜트 / 건식 / 조단백: 38% / 지방: 18%"
    )
    check("Integration2: 수분 없음 → 단백질 UNKNOWN",
          p2.nutrient_states["단백질_g"] == STATE_UNKNOWN)
    check("Integration2: 수분 없음 → warnings 1개 이상",
          len(p2.warnings) >= 1)

    # ProductIn dict 변환
    p3 = danawa_to_product(
        pcode="1802793", name="로얄캐닌 캣 인도어 10kg", brand="로얄캐닌",
        description="고양이 전용 / 사료 / 어덜트 / 건식 / 조단백: 25% / 지방: 11% / 수분: 5.5%"
    )
    pi = p3.to_product_in()
    check("Integration3: to_product_in() id=pcode",
          pi["id"] == "1802793")
    check("Integration3: to_product_in() aafco_life_stage=MAINTENANCE",
          pi["aafco_life_stage"] == "MAINTENANCE")
    check("Integration3: to_product_in() category=food",
          pi["category"] == "food")
    check("Integration3: to_product_in() 단백질 DMB 적용",
          abs(pi["guaranteed_analysis"]["단백질_g"] - 26.46) < 0.01)


# ============ 7) N1 수정 (life_stage_status) ============
def test_n1_life_stage_status():
    print("\n--- 7) N1 수정: life_stage_status ---")
    # keyword 있음 → KNOWN
    p = danawa_to_product(
        pcode="k1", name="Test 1kg", brand="X",
        description="강아지 전용 / 사료 / 어덜트 / 건식 / 조단백: 25% / 수분: 10%"
    )
    check("N1: keyword 있음 → life_stage_status=KNOWN",
          p.life_stage_status == STATE_KNOWN)
    check("N1: keyword 있음 → aafco_life_stage=MAINTENANCE",
          p.aafco_life_stage == "MAINTENANCE")

    # keyword 없음 → UNKNOWN (N1 핵심: None 유지, status로만 표시)
    p2 = danawa_to_product(
        pcode="k2", name="Test 2kg", brand="Y",
        description="강아지 전용 / 사료 / 건식 / 조단백: 25% / 수분: 10%"
    )
    check("N1: keyword 없음 → life_stage_status=UNKNOWN",
          p2.life_stage_status == STATE_UNKNOWN)
    check("N1: keyword 없음 → aafco_life_stage=None (강제 fallback 없음)",
          p2.aafco_life_stage is None)
    # to_product_in()도 UNKNOWN을 유지해야 한다.
    pi2 = p2.to_product_in()
    check("N1: to_product_in() aafco_life_stage=None 유지",
          pi2["aafco_life_stage"] is None)
    check("N1: to_product_in()에 life_stage_status 포함",
          pi2["life_stage_status"] == STATE_UNKNOWN)
    check("N1: to_product_in()에 analysis_feasibility 포함",
          "analysis_feasibility" in pi2)


# ============ 8) score=0 → analysis_feasibility ============
def test_feasibility():
    print("\n--- 8) score=0 → analysis_feasibility ---")
    # 수분 있는 정상 row → READY
    p1 = danawa_to_product(
        pcode="f1", name="Test 1kg", brand="X",
        description="강아지 전용 / 사료 / 어덜트 / 건식 / 조단백: 25% / 지방: 12% / 수분: 10%"
    )
    check("Feasibility: 정상 영양소 → READY",
          p1.analysis_feasibility == FEASIBILITY_READY)
    check("Feasibility: 정상 to_product_in()에 READY 포함",
          p1.to_product_in()["analysis_feasibility"] == FEASIBILITY_READY)

    # 수분 없는 row → INSUFFICIENT_DATA
    p2 = danawa_to_product(
        pcode="f2", name="Test 2kg", brand="Y",
        description="강아지 전용 / 사료 / 어덜트 / 건식 / 조단백: 25% / 지방: 12%"
    )
    check("Feasibility: 수분 없음 → INSUFFICIENT_DATA",
          p2.analysis_feasibility == FEASIBILITY_INSUFFICIENT)

    # 영양소 거의 없는 row → INSUFFICIENT_DATA
    p3 = danawa_to_product(
        pcode="f3", name="Test 3kg", brand="Z",
        description="강아지 전용 / 사료 / 습식 / 캔"
    )
    check("Feasibility: 영양소 0개 → INSUFFICIENT_DATA",
          p3.analysis_feasibility == FEASIBILITY_INSUFFICIENT)


# ============ main ============
def main():
    test_parsing()
    test_dmb()
    test_lifestage()
    test_name()
    test_category()
    test_integration()
    test_n1_life_stage_status()
    test_feasibility()

    print("\n" + "=" * 60)
    print(f"총 {PASS + FAIL}개: PASS={PASS}, FAIL={FAIL}")
    print("=" * 60)
    if FAIL > 0:
        print("\n[FAIL 상세]")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
