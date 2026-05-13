# Wedding Teacher

결혼 준비 첫걸음을 함께 짚어주는 LangGraph 챗 에이전트.
청담·강남 결혼준비대행사 5곳의 스드메 가격(594 entries) 과 400+ 샵의 스타일/인기도 데이터를 자연어로 추천·비교·해설해 준다.

## 과제 요구사항 체크리스트

### 필수

| # | 요구사항 | 구현 | 위치 |
|---|---|---|---|
| 1 | 최소 3개의 노드 | ✅ **4개** — `parse_intent`, `tool_lookup`, `retrieve_per_category`, `recommend` | `graph.py` |
| 2 | 최소 1개의 Conditional Edge (사용자 입력에 따라 다른 경로) | ✅ `route_after_intent` 함수가 의도/카테고리에 따라 **3 갈래**로 분기 | `graph.py:route_after_intent` |
| 3 | 최소 1개의 Tool 연동 | ✅ **커스텀 Tool** — `@tool lookup_vendor` (vendor 데이터베이스 조회), `llm.bind_tools()` 로 LLM에 바인딩 | `graph.py:lookup_vendor` |

### 추가 기능 (선택)

| # | 항목 | 구현 | 설명 |
|---|---|---|---|
| 1 | 병렬 실행 (Send API) | ✅ | `category == "all"` 일 때 `retrieve_per_category` 를 스튜디오·드레스·메이크업 3 카테고리에 **Send 로 fan-out**, `Annotated[list[dict], operator.add]` 누적 reducer 로 결과 합침 |
| 2 | 메모리 기능 | ✅ | Streamlit `session_state` 에 대화 기록 보관, 매 턴 `chat_history` 로 그래프에 전달 → 후속 질문("그 중에 가장 저렴한건?") 처리 가능 |
| 3 | 여러 개의 Tool 연동 | ❌ | 현재 `lookup_vendor` 1개 (필수 요건은 충족) |
| 4 | PyTest 노드 테스트 + AI-as-judge 평가 | ✅ | `tests/test_nodes.py` (도구·헬퍼·라우팅 26 케이스) + `tests/test_ai_judge.py` (`gpt-4o` 심판이 faithfulness/relevance/tone/formatting 4 차원 채점, 4 케이스) |

### Conditional Edge 분기 상세

`graph.py:route_after_intent` 가 `parse_intent` 결과의 `intent_type` 과 `category` 에 따라 다음과 같이 분기:

| 사용자 입력 예시 | 추출된 의도 | 라우팅 결과 |
|---|---|---|
| "벨에포크 어떤 샵이야?" | `intent_type="explain"`, `vendor_mention="벨에포크"` | `tool_lookup` (도구 호출 경로) |
| "스드메 패키지 추천해줘" | `category="all"` | `retrieve_per_category` × 3 (병렬) |
| "200만원 실크 드레스 추천" | `category="dress"` | `retrieve_per_category` × 1 |

## 주요 기능

- **자연어 추천**: "200만원 안쪽 실크 드레스 인기샵 추천해줘" → 후보 필터 + 가격 비교 + 추천 이유 + **오늘의 웨딩 지식** 한 줄 자동 첨부.
- **vendor 비교**: 같은 샵이 여러 대행사에 등록돼 있으면 가격 차이를 자동으로 짚어줌 (예: 누벨드블랑 — 베리굳 125만원 vs 케이앤엠 162만원).
- **vendor 해설**: 사용자가 특정 샵을 언급하면 LangChain `@tool` 을 호출해 데이터베이스 lookup 후 답변.
- **카테고리 통합 추천**: "스드메 풀 패키지 견적" 류 질문은 스튜디오·드레스·메이크업 3 카테고리 동시 fan-out (Send API 병렬).
- **체형 가산점**: 사이드바에 키·고민 부위 입력 시 추천 사유에 체형별 드레스 가이드 자동 반영.
- **선생님의 노트**: 사이드바 expander 5종 — 상담 체크리스트 / 결혼 준비 타임라인 / 시즌·시간대 가격 가이드 / 체형별 드레스 가이드 / 추가금 함정 모음.

## 데이터

| 파일 | 내용 | 출처 |
|---|---|---|
| `data/sdm_pricing.json` | 5개 대행사 × 594 entries (Studio 247 / Dress 224 / Makeup 123) | 결혼준비대행 업체 가격정보 (2025-10-10, 5개사) |
| `data/dress_styles.json` | 135 vendor 스타일·인기도 (BEST 26개) | 다이렉트결혼준비, 아이웨딩, 웨딩북, 웨딩랩더하기, 결직웨딩 |
| `data/studio_styles.json` | 174 vendor 스타일·인기도 (BEST 20개) | 동상 + keyzard, 웨딩21 |
| `data/makeup_styles.json` | 95 vendor 스타일·인기도 (BEST 19개) | 동상 + 마리끌레르, 얼루어 |

가격은 **만원 단위, VAT 포함, 성수기 기준**. 스튜디오 기본은 앨범 20p / 액자 20R.

## 실행

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY 채우기

uv run streamlit run streamlit_app.py   # 메인 UI (http://localhost:8501)
uv run langgraph dev                    # LangGraph Studio (그래프 시각화)
```

## 테스트

```bash
uv run pytest -m unit    # 노드/도구 단위 테스트 (API 호출 없음)
uv run pytest -m judge   # AI-as-judge 품질 평가 (OpenAI API 호출)
uv run pytest            # 전체
```
