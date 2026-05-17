# Wedding Teacher

결혼 준비 첫걸음을 함께 짚어주는 LangGraph 챗 에이전트.
청담·강남 결혼준비대행사 5곳의 스드메 가격(594 entries) 과 400+ 샵의 스타일/인기도 데이터를 자연어로 추천·비교·해설해 준다.

## 주요 기능

- **자연어 추천**: "200만원 안쪽 실크 드레스 인기샵 추천해줘" → 후보 필터 + 가격 비교 + 추천 이유 + **오늘의 웨딩 지식** 한 줄 자동 첨부.
- **vendor 비교**: 같은 샵이 여러 대행사에 등록돼 있으면 가격 차이를 자동으로 짚어줌 (예: 누벨드블랑 — 베리굳 125만원 vs 케이앤엠 162만원).
- **vendor 해설**: 사용자가 특정 샵을 언급하면 LangChain `@tool` 을 호출해 데이터베이스 lookup 후 답변.
- **카테고리 통합 추천**: "스드메 풀 패키지 견적" 류 질문은 스튜디오·드레스·메이크업 3 카테고리 동시 fan-out (Send API 병렬).

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
