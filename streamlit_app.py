import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.memory import InMemorySaver

from graph import graph_builder
from education import (
    BODY_DRESS_GUIDE,
    CHECKLISTS,
    SEASON_TIMING,
    TIMELINE,
    WARNINGS,
)


st.set_page_config(
    page_title="Wedding Teacher",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.css');
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #FAF6F0;
}
html, body {
    font-family: 'Pretendard', 'Apple SD Gothic Neo', sans-serif;
    color: #2C2825;
}
[data-testid="stMarkdownContainer"],
[data-testid="stChatMessageContent"],
.stChatInput textarea,
.stTextInput input,
.stSelectbox label,
.stMultiSelect label,
.stExpander summary {
    font-family: 'Pretendard', 'Apple SD Gothic Neo', sans-serif;
    color: #2C2825;
}
h1, h2, h3 {
    font-family: 'Noto Serif KR', serif !important;
    color: #2C2825;
    letter-spacing: -0.01em;
}
.brand-tag {
    color: #B76E79;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    font-size: 0.7rem;
    text-align: center;
    margin: 8px 0 4px;
}
.brand-title {
    text-align: center;
    margin: 0 0 12px;
}
.brand-sub {
    text-align: center;
    color: #5b524d;
    margin-bottom: 28px;
    line-height: 1.7;
}
[data-testid="stChatMessage"] {
    background-color: white;
    border: 1px solid #E8DECF;
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(60,50,40,0.04), 0 6px 18px rgba(60,50,40,0.04);
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #F8F2E8;
}
.stChatInputContainer {
    border-top: 1px solid #E8DECF;
}
button[kind="secondary"] {
    background-color: white !important;
    border: 1px solid #E8DECF !important;
    color: #2C2825 !important;
    border-radius: 999px !important;
    font-size: 0.85rem !important;
}
button[kind="secondary"]:hover {
    border-color: #B76E79 !important;
    color: #B76E79 !important;
}
.disclaimer {
    color: #8a7f78;
    font-size: 0.75rem;
    text-align: center;
    line-height: 1.6;
    margin-top: 20px;
}
[data-testid="stSidebar"] {
    background-color: #F8F2E8;
}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    font-family: 'Noto Serif KR', serif !important;
    color: #2C2825;
}
.note-card {
    background-color: white;
    border: 1px solid #E8DECF;
    border-radius: 12px;
    padding: 12px 14px;
    margin: 6px 0;
}
.phase-tag {
    display: inline-block;
    background: #B76E79;
    color: white;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 999px;
    margin-right: 6px;
}
.body-tip {
    color: #7E4951;
    font-size: 0.85rem;
    font-style: italic;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_graph():
    checkpointer = InMemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)


graph = get_graph()


# === Sidebar: 결혼 선생님의 노트 + 체형 입력 ===

with st.sidebar:
    st.markdown("### Teacher's Notes")
    st.caption("수업 자료를 옆에 두고 참고하세요. 상담 가기 전에 한번 훑어보면 좋아요.")

    with st.expander("내 체형 정보 (드레스 추천에 반영)"):
        height_choice = st.selectbox(
            "키",
            ["선택 안 함", "160cm 이하", "160-170cm", "170cm 이상"],
            key="body_height",
        )
        bust_choice = st.selectbox(
            "가슴 사이즈",
            ["선택 안 함", "작은 편", "평균", "큰 편"],
            key="body_bust",
        )
        concerns_choice = st.multiselect(
            "고민 부위",
            ["어깨가 넓어요", "팔뚝이 신경 쓰여요", "배·복부가 신경 쓰여요"],
            key="body_concerns",
        )
        highlights_choice = st.multiselect(
            "강조하고 싶은 부분",
            ["허리 라인", "쇄골·어깨선", "다리 길이"],
            key="body_highlights",
        )
        st.caption("입력하면 추천 사유에 체형 가이드가 자연스럽게 반영돼요.")

    with st.expander("상담 전 체크리스트"):
        for section, items in CHECKLISTS.items():
            st.markdown(f"**{section}**")
            for it in items:
                st.markdown(f"- {it}")
            st.markdown("")

    with st.expander("결혼 준비 타임라인"):
        for phase in TIMELINE:
            st.markdown(
                f"<span class='phase-tag'>{phase['phase']}</span>",
                unsafe_allow_html=True,
            )
            for t in phase["tasks"]:
                st.markdown(f"- {t}")
            st.markdown("")

    with st.expander("시즌·시간대 가격 가이드"):
        for k in ("성수기", "비수기"):
            d = SEASON_TIMING[k]
            st.markdown(
                f"**{k}** ({d['months']}) — 가격 {d['extra_cost']}\n\n_{d['note']}_"
            )
        st.markdown("**타임별**")
        for t, v in SEASON_TIMING["타임"].items():
            st.markdown(f"- {t}: {v}")
        st.markdown("**요일별**")
        for t, v in SEASON_TIMING["요일"].items():
            st.markdown(f"- {t}: {v}")

    with st.expander("체형별 드레스 가이드"):
        for g in BODY_DRESS_GUIDE:
            st.markdown(f"**{g['body_type']}**")
            st.markdown(f"- 추천: {', '.join(g['recommended'])}")
            st.markdown(f"- 피하면 좋음: {', '.join(g['avoid'])}")
            st.markdown(
                f"<div class='body-tip'>{g['tip']}</div>", unsafe_allow_html=True
            )
            st.markdown("")

    with st.expander("추가금 함정 모음"):
        for cat, items in WARNINGS.items():
            st.markdown(f"**{cat}**")
            for it in items:
                st.markdown(f"- {it}")
            st.markdown("")

    st.markdown("---")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        "데이터: 5개 결혼준비 대행사 가격표 (2025-10-10) + 다이렉트결혼준비·아이웨딩·웨딩북·웨딩랩 큐레이션."
    )


# === Main: 챗 영역 ===

st.markdown('<div class="brand-tag">Wedding Teacher</div>', unsafe_allow_html=True)
st.markdown('<h1 class="brand-title">결혼은 처음이라</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="brand-sub">청담·강남 스드메 추천부터 상담 전 알아둘 지식까지,<br/>'
    '결혼 준비 첫걸음을 함께 짚어 드릴게요.</p>',
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo-session"

EXAMPLES = [
    "200만원 안쪽 실크 드레스 인기샵 추천해줘",
    "비즈 드레스 BEST는 어디야?",
    "감성 자연광 스튜디오 추천",
    "시작바이이명순에 대해 알려줘",
]

def _collect_body_info() -> dict:
    height = st.session_state.get("body_height", "선택 안 함")
    bust = st.session_state.get("body_bust", "선택 안 함")
    concerns = st.session_state.get("body_concerns", []) or []
    highlights = st.session_state.get("body_highlights", []) or []
    info = {}
    if height and height != "선택 안 함":
        info["height"] = height
    if bust and bust not in ("선택 안 함", "평균"):
        info["bust"] = bust
    if concerns:
        info["concerns"] = concerns
    if highlights:
        info["highlights"] = highlights
    return info


# 1) 기존 대화 렌더 (이전 턴들)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# 2) 입력 수집 — chat_input 또는 예시 클릭에서 큐잉된 pending_input
typed = st.chat_input("어떤 스타일·예산·고민이 있나요?")
pending = st.session_state.pop("pending_input", None)
input_text = typed or pending


# 3) 입력 처리 — 사용자 메시지와 응답을 인라인 렌더 + 세션 적재
if input_text:
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.markdown(input_text)

    with st.chat_message("assistant"):
        with st.spinner("Wedding Teacher가 자료를 펼쳐보는 중이에요..."):
            try:
                result = graph.invoke(
                    {
                        "user_input": input_text,
                        "chat_history": st.session_state.messages[:-1],
                        "body_info": _collect_body_info(),
                        "intent": {},
                        "candidates": [],
                        "response": "",
                    },
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                response = result.get("response", "(응답 없음)")
            except Exception as e:
                response = (
                    f"오류가 발생했어요: `{e}`\n\n"
                    "`.env` 의 `OPENAI_API_KEY` 가 올바른지 확인해 주세요."
                )
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


# 4) 예시 버튼 — 대화 비어있을 때만. 클릭 시 pending_input 큐잉 후 rerun.
#    (rerun 으로 위쪽 처리 블록이 즉시 실행되며, 다음 렌더부터 messages 가 차서 버튼이 사라진다)
if not st.session_state.messages:
    st.markdown("**예시 질문**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_input = ex
            st.rerun()


st.markdown(
    '<div class="disclaimer">이 결과는 공개된 일반적인 가격·스타일 정보를 바탕으로 한 교육용 추천입니다. '
    '실제 견적은 시즌·시간대·옵션에 따라 달라질 수 있습니다.</div>',
    unsafe_allow_html=True,
)
