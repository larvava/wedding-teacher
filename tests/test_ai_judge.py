"""AI-as-judge 평가.

`gpt-4o-mini` (피평가자) 응답을 별도 `gpt-4o` 심판이 4 차원 1-5 점으로 채점:
  - faithfulness : 데이터셋에 없는 vendor·가짜 가격을 만들지 않았는가
  - relevance    : 사용자 질문에 직접 답했는가
  - tone         : 'Wedding Teacher' 의 차분한 선생님 톤·이모지 없음
  - formatting   : "만원" 단위·하이픈(-) 범위 표기·물결(~) 미사용

API 비용/시간 때문에 기본 collection 에서 빠지도록 `judge` 마커만 단다.
실행: `uv run pytest -m judge`
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import pytest
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from graph import CATEGORY_STYLES, PRICING, graph_builder


pytestmark = [pytest.mark.judge]


@dataclass(frozen=True)
class JudgeCase:
    name: str
    user_input: str
    must_mention: tuple[str, ...] = ()


CASES: list[JudgeCase] = [
    JudgeCase(
        name="vendor_known",
        user_input="벨에포크 어떤 샵이야?",
        must_mention=("벨에포크",),
    ),
    JudgeCase(
        name="recommend_budget_style",
        user_input="200만원 안쪽 실크 드레스 인기샵 추천",
    ),
    JudgeCase(
        name="recommend_all_categories",
        user_input="스드메 풀패키지 추천해줘",
    ),
    JudgeCase(
        name="vendor_partial_mention",
        user_input="엔조 알려줘",
        must_mention=("엔조",),  # LLM 이 풀네임을 줄여 쓰기도 함 — 부분명 매칭만 검증
    ),
]


JUDGE_PROMPT = """당신은 한국 결혼 정보 챗봇 응답을 평가하는 깐깐한 심사위원입니다.
아래 응답을 4개 차원에서 1-5 점으로 채점하세요. 1=매우 나쁨, 5=완벽.

[차원]
1. faithfulness: 데이터셋에 없는 vendor 이름·가짜 가격을 만들지 않았는가.
   - 응답에 등장한 vendor 가 아래 KNOWN_VENDORS 목록 안에 있으면 신뢰 가능.
   - 부분 매칭(별칭·약칭) 도 OK. 예: "더청담스튜디오" 와 "더청담" 은 동일 vendor 로 본다.
   - 응답에 vendor 가 없을 수도 있고(일반 안내) 그건 정상이며 5점.
   - 응답에 등장한 vendor 가 KNOWN_VENDORS 에 없거나 가격을 데이터셋 출처 없이 추측한
     흔적이 보이면 1-2 점.
2. relevance: 사용자 질문에 직접·구체적으로 답했는가.
3. tone: 차분하고 따뜻한 '선생님' 톤인가. 이모지 사용은 감점.
4. formatting: 가격 언급 시 단위는 '만원', 범위는 하이픈(-)만 (물결 ~ 금지).
   가격이 응답에 전혀 등장하지 않아도 형식 위반은 아니므로 5점.

[KNOWN_VENDORS — 데이터셋 등재 vendor (개별 샵) 전체 목록]
{vendor_list}

[KNOWN_AGENCIES — 데이터셋 등재 결혼준비 대행사 목록]
{agency_list}

주의: '대행사' 는 견적을 주는 결혼준비 대행사이고, 'vendor' 는 각 대행사가 연결해주는
실제 스튜디오/드레스/메이크업 샵입니다. 둘은 별개로 평가하세요.

[사용자 질문]
{user_input}

[응답]
{response}

JSON 으로만 출력. 코드블록 없이.
{{
  "faithfulness": 1-5,
  "relevance": 1-5,
  "tone": 1-5,
  "formatting": 1-5,
  "rationale": "한 줄 사유"
}}
"""


_judge_llm = None


def _judge():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = init_chat_model("openai:gpt-4o", temperature=0)
    return _judge_llm


def _vendor_list() -> str:
    names: set[str] = set()
    for data in CATEGORY_STYLES.values():
        names.update(data["vendors"].keys())
    return ", ".join(sorted(names))


def _agency_list() -> str:
    return ", ".join(a["name"] for a in PRICING["agencies"])


def _score(user_input: str, response: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        vendor_list=_vendor_list(),
        agency_list=_agency_list(),
        user_input=user_input,
        response=response,
    )
    raw = _judge().invoke(prompt).content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(text)


def _has_no_tilde_range(text: str) -> bool:
    return re.search(r"\d\s*~\s*\d", text) is None


def _has_no_emoji(text: str) -> bool:
    return re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text) is None


@pytest.fixture(scope="module", autouse=True)
def _require_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY 가 없으면 judge 테스트를 건너뜁니다.")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_ai_judge_minimum_quality(case: JudgeCase):
    graph = graph_builder.compile(checkpointer=InMemorySaver())
    result = graph.invoke(
        {
            "user_input": case.user_input,
            "chat_history": [],
            "body_info": {},
            "intent": {},
            "candidates": [],
            "response": "",
        },
        config={"configurable": {"thread_id": f"judge-{case.name}"}},
    )
    response = result.get("response") or ""
    assert response, "응답이 비어있으면 평가 불가"

    # 정형 검사 — 결정적
    assert _has_no_tilde_range(response), "가격 범위는 하이픈(-) 만 사용해야 합니다 (물결 ~ 금지)"
    assert _has_no_emoji(response), "이모지가 응답에 등장했습니다"
    for token in case.must_mention:
        assert token in response, f"필수 토큰 '{token}' 누락"

    # LLM 심판 점수 — 각 차원 ≥ 3
    scores = _score(case.user_input, response)
    failures = [
        k for k in ("faithfulness", "relevance", "tone", "formatting") if scores[k] < 3
    ]
    assert not failures, f"AI-judge 평가 미달 차원={failures}, scores={scores}"
