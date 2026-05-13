import json
import operator
import re
from pathlib import Path
from typing import TypedDict
from typing_extensions import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from education import (
    body_dress_brief,
    daily_tip,
    season_timing_brief,
)


llm = init_chat_model("openai:gpt-4o-mini")

DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


PRICING = _load("sdm_pricing.json")
DRESS_STYLES = _load("dress_styles.json")
STUDIO_STYLES = _load("studio_styles.json")
MAKEUP_STYLES = _load("makeup_styles.json")

CATEGORY_STYLES = {
    "studio": STUDIO_STYLES,
    "dress": DRESS_STYLES,
    "makeup": MAKEUP_STYLES,
}

CATEGORY_LABEL_KO = {"studio": "스튜디오", "dress": "드레스", "makeup": "메이크업"}

ALL_STYLE_TAGS = sorted({
    tag
    for data in CATEGORY_STYLES.values()
    for vendor in data["vendors"].values()
    for tag in vendor.get("styles", [])
})


def _extract_min_price(entry: dict, category: str):
    if category == "studio":
        price = entry.get("price") or entry.get("base")
        if isinstance(price, (int, float)):
            return float(price)
        if isinstance(price, str):
            nums = re.findall(r"\d+(?:\.\d+)?", price)
            if nums:
                return min(float(n) for n in nums)
    elif category == "dress":
        for key in ("main_only", "main", "main_plus_shoot", "main_plus_shoot_total", "shoot"):
            v = entry.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    elif category == "makeup":
        tiers = entry.get("tiers") or {}
        nums = [v for v in tiers.values() if isinstance(v, (int, float))]
        if nums:
            return float(min(nums))
        for key in ("main_only", "main_plus_shoot"):
            v = entry.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _normalize_name(name: str) -> str:
    """표기차 통일 — 괄호 내용은 본 이름에 합치고 공백 제거.

    "엘리자베스 (럭스)" / "엘리자베스 럭스" / "엘리자베스럭스" → "엘리자베스럭스".
    """
    n = re.sub(r"\(([^)]*)\)", r"\1", name)
    n = re.sub(r"\s+", "", n)
    for suffix in ("스튜디오", "메이크업", "헤어", "웨딩"):
        if n.endswith(suffix) and len(n) > len(suffix) + 1:
            n = n[: -len(suffix)]
    return n


def _build_price_index() -> dict:
    idx = {"studio": {}, "dress": {}, "makeup": {}}
    for agency in PRICING["agencies"]:
        agency_name = agency["name"]
        for cat in ("studio", "dress", "makeup"):
            for entry in agency.get(cat, []):
                vendor = entry["vendor"]
                norm = _normalize_name(vendor)
                price = _extract_min_price(entry, cat)
                idx[cat].setdefault(norm, []).append({
                    "agency": agency_name,
                    "vendor_as_listed": vendor,
                    "price_man": price,
                    "raw": entry,
                })
    return idx


PRICE_INDEX = _build_price_index()


@tool
def lookup_vendor(query: str) -> str:
    """업체명·별칭·키워드로 스드메 데이터베이스에서 vendor 상세를 조회합니다.

    Args:
        query: 찾을 업체명 또는 키워드 (예: '벨에포크', '꾸띠원', '라포레').

    Returns:
        해당 vendor 의 카테고리·스타일·인기도·메모와 모든 대행사별 가격을 JSON 으로.
    """
    q_norm = _normalize_name(query).lower()
    matches = []
    for category, data in CATEGORY_STYLES.items():
        for vendor_name, info in data["vendors"].items():
            v_norm = _normalize_name(vendor_name).lower()
            if q_norm in v_norm or v_norm in q_norm or query.lower() in vendor_name.lower():
                norm = _normalize_name(vendor_name)
                prices = PRICE_INDEX.get(category, {}).get(norm, [])
                price_lines = []
                for p in prices:
                    if p.get("price_man") is not None:
                        price_lines.append(f"{p['agency']}: {p['price_man']:g}만원")
                    else:
                        price_lines.append(f"{p['agency']}: 가격정보없음")
                matches.append({
                    "name": vendor_name,
                    "category": CATEGORY_LABEL_KO[category],
                    "styles": info.get("styles", []),
                    "popularity": info.get("popularity", 0),
                    "notes": info.get("notes", ""),
                    "agency_prices": price_lines,
                })
    if not matches:
        return json.dumps({"error": f"'{query}' 와 일치하는 vendor 를 찾지 못했습니다."}, ensure_ascii=False)
    return json.dumps(matches, ensure_ascii=False, indent=2)


llm_with_tools = llm.bind_tools([lookup_vendor])


class State(TypedDict):
    user_input: str
    chat_history: list[dict]
    body_info: dict
    intent: dict
    candidates: Annotated[list[dict], operator.add]
    response: str


INTENT_PROMPT = """당신은 한국 결혼 준비 상담 챗봇의 의도 파서입니다.
사용자 메시지를 보고 다음 JSON 스키마로만 출력하세요. 코드 블록 없이 순수 JSON.

{{
  "category": "studio" | "dress" | "makeup" | "all",
  "styles": [...],
  "budget_max_man": 정수 또는 null,
  "popularity_min": 0-3 정수,
  "intent_type": "recommend" | "compare" | "explain" | "general",
  "vendor_mention": 사용자가 언급한 특정 업체명 또는 null
}}

가능한 styles 값 (반드시 이 목록에서만 선택):
{style_tags}

규칙:
- vendor_mention 은 **명확한 고유명사(개별 샵 이름)** 일 때만 채우세요. 일반어·스타일명·카테고리·수식어는 제외.
  ❌ 잘못된 예: "비즈 드레스 BEST", "인기 스튜디오", "유명한 메이크업", "실크 드레스 맛집", "BEST", "추천샵"
  ✅ 올바른 예: "벨에포크", "시작바이이명순", "누벨드블랑", "더청담스튜디오"
  — 의심스러우면 null 로 두세요.
- vendor_mention 이 채워졌을 때만 intent_type="explain". 비어있으면 intent_type 은 "recommend" 또는 "general".
- 카테고리가 명시 안되면 "all".
- "인기있는", "유명한", "BEST", "맛집" 표현 → popularity_min 2 또는 3.
- "200만원" → 200, "5천만원" → 5000.
- styles 는 사용자 표현과 가장 가까운 것만. 없으면 빈 배열.

이전 대화:
{history}

현재 사용자 메시지:
{user_msg}

JSON:"""


def _format_history(history: list[dict], n: int = 4) -> str:
    if not history:
        return "(없음)"
    last = history[-n:]
    return "\n".join(f"[{m['role']}] {m['content']}" for m in last)


def parse_intent(state: State):
    """노드 1: LLM 으로 사용자 의도(JSON)를 파싱."""
    prompt = INTENT_PROMPT.format(
        style_tags=", ".join(ALL_STYLE_TAGS),
        history=_format_history(state.get("chat_history", [])),
        user_msg=state["user_input"],
    )
    raw = llm.invoke(prompt).content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        intent = json.loads(text)
    except json.JSONDecodeError:
        intent = {}
    intent.setdefault("category", "all")
    intent.setdefault("styles", [])
    intent.setdefault("budget_max_man", None)
    intent.setdefault("popularity_min", 0)
    intent.setdefault("intent_type", "general")
    intent.setdefault("vendor_mention", None)
    return {"intent": intent}


def _candidates_for_category(
    category: str,
    styles: list[str],
    budget_max: int | None,
    pop_min: int,
    limit: int = 6,
) -> list[dict]:
    styles_data = CATEGORY_STYLES[category]
    results = []
    for vendor_name, info in styles_data["vendors"].items():
        if info.get("popularity", 0) < pop_min:
            continue
        if styles:
            v_styles = info.get("styles", [])
            if not any(s in v_styles for s in styles):
                continue
        norm = _normalize_name(vendor_name)
        prices = list(PRICE_INDEX.get(category, {}).get(norm, []))
        if not prices:
            for k, v in PRICE_INDEX.get(category, {}).items():
                if k and (norm in k or k in norm):
                    prices.extend(v)
        if budget_max is not None:
            valid = [p for p in prices if p.get("price_man") and p["price_man"] <= budget_max]
            if not valid:
                continue
            prices = valid
        results.append({
            "vendor": vendor_name,
            "category": category,
            "category_ko": CATEGORY_LABEL_KO[category],
            "styles": info.get("styles", []),
            "popularity": info.get("popularity", 0),
            "notes": info.get("notes", ""),
            "agencies": [
                {"agency": p["agency"], "price_man": p["price_man"]}
                for p in prices
            ],
        })

    def sort_key(r):
        prices = [a["price_man"] for a in r["agencies"] if a.get("price_man")]
        return (-r["popularity"], min(prices) if prices else 9999)

    results.sort(key=sort_key)
    return results[:limit]


def retrieve_per_category(args: dict):
    """노드 2: Send 로 dispatch 되어 한 카테고리의 후보를 필터링.

    Send 로 fan-out 되면 병렬 실행되며 candidates 에 누적된다.
    """
    category = args["category"]
    intent = args["intent"]
    candidates = _candidates_for_category(
        category,
        intent.get("styles", []) or [],
        intent.get("budget_max_man"),
        intent.get("popularity_min", 0) or 0,
        limit=args.get("limit", 6),
    )
    return {"candidates": candidates}


TEACHER_SYSTEM = (
    "당신은 'Wedding Teacher' 입니다. 결혼 준비가 처음인 예비부부에게 "
    "베테랑 선배가 차근차근 가르쳐 주듯 따뜻하고 차분한 어투로 답변하세요. "
    "정보를 던지기보다 이해하도록 설명하고, 추천에는 항상 이유를 함께 알려 주세요. "
    "이모지는 사용하지 마세요."
)


_TILDE_RANGE_RE = re.compile(r"(\d)\s*~\s*(\d)")


def _normalize_ranges(text: str) -> str:
    """LLM 답변에 남은 '20~35%' 같은 표기를 '20-35%' 로 강제 변환."""
    return _TILDE_RANGE_RE.sub(r"\1-\2", text)


def _append_daily_tip(text: str) -> str:
    cleaned = _normalize_ranges(text or "")
    tip = _normalize_ranges(daily_tip())
    return f"{cleaned}\n\n---\n**오늘의 웨딩 지식**\n{tip}"


def _top_popular_vendors(category: str, n: int = 5) -> list[str]:
    vendors = CATEGORY_STYLES[category]["vendors"]
    items = sorted(vendors.items(), key=lambda kv: -kv[1].get("popularity", 0))
    return [name for name, _ in items[:n]]


def _format_fallback_pool() -> str:
    return "\n".join(
        f"  - {CATEGORY_LABEL_KO[cat]}: {', '.join(_top_popular_vendors(cat, 5))}"
        for cat in ("studio", "dress", "makeup")
    )


def tool_lookup(state: State):
    """노드 3: vendor 조회. vendor_mention 이 있으면 도구를 결정적으로 호출,
    없으면 bind_tools 한 LLM 이 호출 여부를 판단."""
    user_msg = state["user_input"]
    vendor_hint = (state.get("intent") or {}).get("vendor_mention")

    if vendor_hint:
        tool_results = [lookup_vendor.invoke({"query": vendor_hint})]
    else:
        first = llm_with_tools.invoke([
            ("system", TEACHER_SYSTEM + " 사용자가 특정 샵을 묻거나 정보가 필요할 땐 lookup_vendor 도구를 사용해 데이터베이스에서 조회한 뒤 답변하세요."),
            ("user", user_msg),
        ])
        tool_results = []
        for tc in (first.tool_calls or []):
            if tc["name"] == "lookup_vendor":
                result = lookup_vendor.invoke(tc["args"])
                tool_results.append(result)
        if not tool_results:
            return {"response": _append_daily_tip(first.content or "(응답이 비어있어요. 다시 질문해 주세요.)")}

    final = llm.invoke(
        f"""{TEACHER_SYSTEM}

사용자 질문:
{user_msg}

도구(lookup_vendor) 조회 결과:
{chr(10).join(tool_results)}

[추천 가능 vendor 풀 — 비슷한 샵 제안 시 반드시 이 안에서만 고르세요]
{_format_fallback_pool()}

[중요 규칙 — 절대 위반 금지]
1. **vendor 이름은 위 도구 조회 결과 또는 [추천 가능 vendor 풀] 에 있는 이름만 사용하세요.**
   존재하지 않는 가짜 이름(예: '메이크업바이소영', '뷰티풀리' 등 데이터에 없는 이름) 절대 만들지 마세요.

2. 조회 결과 분기:
   - **vendor 가 매칭되고 agency_prices 가 채워져 있으면**: 대행사별 견적 마크다운 표 작성
     | 대행사 | 가격 | 비고 |
     | --- | --- | --- |
     표 위에 가장 저렴/비싼 대행사 한 줄 코멘트.
   - **vendor 는 매칭됐지만 agency_prices 가 비어있으면**: "본 데이터셋(5개 대행사 가격표 기준)에는 가격이 미연계된 업체입니다" 안내. notes 에 적힌 스타일/특징 설명. 그 다음 위 [추천 가능 vendor 풀] 에서 같은 카테고리 1-2 곳 이름만 짧게 제안.
   - **조회 결과에 error 가 포함되면**: "본 데이터셋에 등록되지 않은 업체입니다" 솔직히 안내. 일반 결혼 정보(웨딩홀 비용, 식대 등) 절대 만들지 말 것. 위 풀에서 1-2 곳 제안.

3. 추천 시 가짜 가격 만들지 마세요. 가격 모르면 이름·스타일만.
4. 가격 단위 만원, 범위는 하이픈(-), 물결(~) 금지, 이모지 금지.
5. 마지막에 "이런 점도 알아두세요" 짧은 가르침 한 줄."""
    )
    return {"response": _append_daily_tip(final.content)}


RECOMMEND_PROMPT = """{system}

당신은 결혼 선생님으로서 학생(예비부부)에게 결혼 준비를 가르치고 있습니다.
샵 추천뿐 아니라, 선택 이유와 함께 알아두면 좋은 지식을 한두 마디 곁들여 주세요.

[규칙]
- 추천 샵은 아래 후보 목록에서만. 다른 곳을 만들어내지 마세요.
- 후보가 비어있으면 솔직히 "조건에 맞는 데이터가 부족합니다" 말하고 조건 완화를 부드럽게 제안.
- 각 추천 샵: 이름, 한 줄 스타일 요약, 인기도(0-3), 어느 대행사에서 얼마인지(만원), 추천 이유.
- 같은 샵이 여러 대행사에 있으면 가격 차이를 짚어 비교 포인트로 알려주세요.
- 사용자가 체형(키·어깨·고민 부위) 정보를 줬다면 그에 맞는 실루엣 추천 사유에 반영.
- 사용자가 시즌·시간대를 언급하면 가격 영향(성수기 +15-30%, 평일 20-30% 할인 등) 짧게 부연.
- 가격은 모두 "만원" 단위. 1-2 문장 도입 후 추천 이어가기.
- 가격 범위는 '20-35%' 처럼 하이픈(-) 으로 표기. 물결(~)은 사용 금지.
- 마크다운 가능. 이모지 금지.

[참고 — 시즌·시간대 가격 영향]
{season_timing}

[참고 — 체형별 드레스 가이드]
{body_dress}

[사용자 체형 정보]
{body_info}

[사용자 의도 (JSON)]
{intent}

[필터된 후보 목록]
{candidates}

[이전 대화]
{history}

[이번 사용자 메시지]
{user_msg}

답변:"""


def _format_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return "(없음 — 조건을 완화하거나 다른 카테고리로 안내해 주세요)"
    out = []
    for c in candidates:
        agencies = []
        for a in c.get("agencies", []):
            price = a.get("price_man")
            agencies.append(
                f"{a['agency']} {price:g}만원" if price is not None else f"{a['agency']} (가격정보없음)"
            )
        styles = ", ".join(c.get("styles", [])) or "(태그없음)"
        out.append(
            f"- [{c['category_ko']}] {c['vendor']} | 스타일: {styles} | 인기도: {c.get('popularity', 0)}/3\n"
            f"  대행사: {' / '.join(agencies) if agencies else '연계 정보 없음'}\n"
            f"  메모: {c.get('notes', '')}"
        )
    return "\n".join(out)


def _format_body_info(body_info: dict) -> str:
    if not body_info:
        return "(입력 없음)"
    parts = []
    if body_info.get("height"):
        parts.append(f"키: {body_info['height']}")
    if body_info.get("bust"):
        parts.append(f"가슴 사이즈: {body_info['bust']}")
    if body_info.get("concerns"):
        parts.append(f"고민 부위: {', '.join(body_info['concerns'])}")
    if body_info.get("highlights"):
        parts.append(f"강조하고 싶은 부분: {', '.join(body_info['highlights'])}")
    return " / ".join(parts) if parts else "(입력 없음)"


def recommend(state: State):
    """노드 4: 후보 목록과 체형/시즌 가이드를 함께 LLM 에 넘겨 추천 응답 생성."""
    intent = state.get("intent", {})
    candidates = state.get("candidates", [])
    body_info = state.get("body_info") or {}
    prompt = RECOMMEND_PROMPT.format(
        system=TEACHER_SYSTEM,
        season_timing=season_timing_brief(),
        body_dress=body_dress_brief(),
        body_info=_format_body_info(body_info),
        intent=json.dumps(intent, ensure_ascii=False),
        candidates=_format_candidates(candidates),
        history=_format_history(state.get("chat_history", [])),
        user_msg=state["user_input"],
    )
    response = llm.invoke(prompt)
    return {"response": _append_daily_tip(response.content)}


_GENERIC_NON_VENDOR_TOKENS = (
    "best", "맛집", "추천", "인기", "유명",
    "드레스", "스튜디오", "메이크업", "스드메",
    "비즈", "잔잔비즈", "실크", "레이스", "튤", "머메이드",
    "감성", "모던", "자연광", "청순", "내추럴",
)


def _looks_like_real_vendor(name: str) -> bool:
    """vendor_mention 이 일반어·스타일·카테고리 단어로만 구성된 경우 False."""
    if not name:
        return False
    n_lower = name.lower().strip()
    # generic word 만으로 구성됐는지 체크
    cleaned = n_lower
    for tok in _GENERIC_NON_VENDOR_TOKENS:
        cleaned = cleaned.replace(tok, "")
    if not cleaned.strip():
        return False
    # 우리 데이터의 실제 vendor 와 substring 매칭이 되면 진짜 vendor 로 본다
    n_norm = _normalize_name(name).lower()
    for category in CATEGORY_STYLES.values():
        for vendor in category["vendors"]:
            v_norm = _normalize_name(vendor).lower()
            if len(v_norm) >= 2 and (n_norm in v_norm or v_norm in n_norm):
                return True
    return False


def route_after_intent(state: State):
    """Conditional edge: intent_type / category 에 따라 다음 노드를 분기.

    - vendor_mention 이 실제 vendor 와 매칭될 때만  → tool_lookup
    - category == "all"                            → retrieve_per_category 를 3 카테고리에 fan-out (Send)
    - category 단일                                → retrieve_per_category 1 회
    """
    intent = state.get("intent") or {}
    vendor_mention = intent.get("vendor_mention")

    if vendor_mention and _looks_like_real_vendor(vendor_mention):
        return [Send("tool_lookup", state)]

    category = intent.get("category", "all")
    if category == "all":
        return [
            Send("retrieve_per_category", {"category": c, "intent": intent, "limit": 3})
            for c in ("studio", "dress", "makeup")
        ]
    if category in ("studio", "dress", "makeup"):
        return [Send("retrieve_per_category", {"category": category, "intent": intent, "limit": 10})]
    return [Send("retrieve_per_category", {"category": "dress", "intent": intent, "limit": 10})]


graph_builder = StateGraph(State)
graph_builder.add_node("parse_intent", parse_intent)
graph_builder.add_node("tool_lookup", tool_lookup)
graph_builder.add_node("retrieve_per_category", retrieve_per_category)
graph_builder.add_node("recommend", recommend)

graph_builder.add_edge(START, "parse_intent")
graph_builder.add_conditional_edges(
    "parse_intent",
    route_after_intent,
    ["tool_lookup", "retrieve_per_category"],
)
graph_builder.add_edge("tool_lookup", END)
graph_builder.add_edge("retrieve_per_category", "recommend")
graph_builder.add_edge("recommend", END)

graph = graph_builder.compile(name="wedding_planner")
