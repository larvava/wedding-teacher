"""graph.py 의 노드·도구·헬퍼 단위 테스트 — LLM 호출 없이 결정적 검증."""

from __future__ import annotations

import json

import pytest
from langgraph.types import Send

from graph import (
    _candidates_for_category,
    _looks_like_real_vendor,
    _normalize_name,
    _strip_existing_tip,
    lookup_vendor,
    route_after_intent,
)


# --- @tool lookup_vendor ---------------------------------------------------


@pytest.mark.unit
def test_lookup_vendor_finds_known_shop():
    payload = json.loads(lookup_vendor.invoke({"query": "벨에포크"}))

    assert isinstance(payload, list) and payload
    assert "벨에포크" in payload[0]["name"]
    assert payload[0]["category"] == "드레스"


@pytest.mark.unit
def test_lookup_vendor_returns_error_for_unknown():
    payload = json.loads(lookup_vendor.invoke({"query": "존재하지않는샵_xyz_998"}))

    assert isinstance(payload, dict)
    assert "error" in payload


@pytest.mark.unit
@pytest.mark.parametrize(
    "query, expected_name",
    [
        ("엘리자베스 (럭스)", "엘리자베스럭스"),
        ("엘리자베스 럭스", "엘리자베스럭스"),
        ("엘리자베스럭스", "엘리자베스럭스"),
        ("더화이트 엘리자베스", "더화이트엘리자베스"),
    ],
)
def test_lookup_vendor_matches_spacing_and_parens_variants(query, expected_name):
    """괄호·공백 표기차로도 동일 vendor 가 매칭되어야 한다."""
    payload = json.loads(lookup_vendor.invoke({"query": query}))

    assert isinstance(payload, list)
    names = [m["name"] for m in payload]
    assert expected_name in names, f"{query!r} → {names}"


# --- _normalize_name -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw, normed",
    [
        ("엘리자베스 (럭스)", "엘리자베스럭스"),
        ("엘리자베스 럭스", "엘리자베스럭스"),
        ("엘리자베스럭스", "엘리자베스럭스"),
        ("더청담스튜디오", "더청담"),
        ("더화이트 엘리자베스", "더화이트엘리자베스"),
    ],
)
def test_normalize_name(raw, normed):
    assert _normalize_name(raw) == normed


# --- _candidates_for_category ---------------------------------------------


@pytest.mark.unit
def test_candidates_for_category_filters_budget_and_popularity():
    out = _candidates_for_category(
        category="dress",
        styles=["실크"],
        budget_max=200,
        pop_min=2,
        limit=10,
    )

    for c in out:
        assert c["category"] == "dress"
        assert c["popularity"] >= 2
        prices_under_budget = [
            a["price_man"] for a in c["agencies"] if a.get("price_man") and a["price_man"] <= 200
        ]
        assert prices_under_budget, f"budget 필터링 실패: {c['vendor']} → {c['agencies']}"


@pytest.mark.unit
def test_candidates_for_category_sorted_by_popularity_desc():
    out = _candidates_for_category(
        category="dress", styles=[], budget_max=None, pop_min=0, limit=20
    )

    pops = [c["popularity"] for c in out]
    assert pops == sorted(pops, reverse=True), pops


@pytest.mark.unit
def test_candidates_for_category_empty_when_impossible():
    out = _candidates_for_category("dress", [], budget_max=1, pop_min=3, limit=10)

    assert out == []


# --- route_after_intent ----------------------------------------------------


@pytest.mark.unit
def test_route_with_real_vendor_mention_goes_to_tool_lookup():
    state = {"intent": {"vendor_mention": "벨에포크", "intent_type": "explain"}}

    sends = route_after_intent(state)

    assert len(sends) == 1
    assert sends[0].node == "tool_lookup"


@pytest.mark.unit
def test_route_with_fake_vendor_mention_falls_through_to_retrieve():
    # "비즈 드레스 BEST" 같은 일반어는 vendor 가 아니므로 retrieve_per_category 로 가야 함
    state = {"intent": {"vendor_mention": "비즈 드레스 BEST", "category": "dress"}}

    sends = route_after_intent(state)

    assert all(s.node == "retrieve_per_category" for s in sends)


@pytest.mark.unit
def test_route_category_all_fans_out_to_three():
    state = {"intent": {"category": "all"}}

    sends = route_after_intent(state)

    assert len(sends) == 3
    assert {s.arg["category"] for s in sends} == {"studio", "dress", "makeup"}
    assert all(s.node == "retrieve_per_category" for s in sends)


@pytest.mark.unit
def test_route_category_single_dispatches_one():
    state = {"intent": {"category": "dress"}}

    sends = route_after_intent(state)

    assert len(sends) == 1
    assert sends[0].arg["category"] == "dress"
    assert sends[0].arg["limit"] == 10


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("본문\n\n---\n**오늘의 웨딩 지식**\n드레스 헬퍼비는 25만원...", "본문"),
        ("본문 마지막 줄.\n\n오늘의 웨딩 지식 하트넥은 여성스러운 느낌...", "본문 마지막 줄."),
        ("본문에 팁 블록 없음", "본문에 팁 블록 없음"),
    ],
)
def test_strip_existing_tip_removes_trailing_tip_block(raw, expected):
    """LLM 이 본문에 만들어 넣었거나 코드가 첨부한 '오늘의 웨딩 지식' 블록을 제거."""
    assert _strip_existing_tip(raw) == expected


@pytest.mark.unit
def test_route_general_intent_goes_to_style_advice():
    """샵 요청 없이 스타일/체형 조언만 묻는 경우 style_advice 로 라우팅."""
    state = {"intent": {"intent_type": "general", "category": "dress"}}

    sends = route_after_intent(state)

    assert len(sends) == 1
    assert sends[0].node == "style_advice"


# --- _looks_like_real_vendor ----------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, expected",
    [
        ("벨에포크", True),
        ("엔조최재훈", True),
        ("시작바이이명순", False),  # 데이터셋에서 삭제됨
        ("비즈 드레스 BEST", False),
        ("인기 스튜디오", False),
        ("유명한 메이크업", False),
        ("", False),
        (None, False),
    ],
)
def test_looks_like_real_vendor(name, expected):
    assert _looks_like_real_vendor(name) is expected
