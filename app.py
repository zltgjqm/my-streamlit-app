import streamlit as st
from openai import OpenAI
from PIL import Image
import datetime
import pandas as pd
import altair as alt
from collections import Counter, defaultdict
import base64
import hashlib
import calendar

# =====================================================
# 기본 설정
# =====================================================
st.set_page_config(page_title="하루의 단서", layout="centered")
st.title("📸 하루의 단서")

# =====================================================
# 사이드바 - API KEY + 사용자 설정
# =====================================================
st.sidebar.header("🔑 API 설정")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
client = OpenAI(api_key=openai_key) if openai_key else None

st.sidebar.header("⚙️ 사용 설정")
usage_mode = st.sidebar.selectbox(
    "기록 빈도 가정",
    ["매일", "주 2~3회", "생각날 때"],
    index=0
)

# =====================================================
# 세션 상태
# =====================================================
if "records" not in st.session_state:
    st.session_state.records = []  # dict list

if "custom_emotions" not in st.session_state:
    st.session_state.custom_emotions = {}

if "stickers" not in st.session_state:
    st.session_state.stickers = []  # earned badges

# =====================================================
# 맥락별 감정 풀 + 기본 감정 풀
# =====================================================
CONTEXT_EMOTIONS = {
    "식사": ["😋 맛있음", "🙂 괜찮았음", "😕 아쉬움", "💸 가격이 아까움"],
    "풍경": ["🌿 차분함", "✨ 인상 깊음", "🙂 그냥 그랬음", "😮 놀라움"],
    "휴식": ["😌 편안함", "🙂 만족스러움", "😴 나른함", "😮‍💨 회복됨"],
    "이동": ["😴 피곤함", "😐 무난함", "😤 지침", "😠 짜증"],
    "여가": ["😆 즐거움", "🙂 만족", "😐 평범함", "🤩 신남"],
    "기타": ["🙂 평범함", "😌 편안함", "😐 무덤덤", "😟 불안", "😆 즐거움", "😤 지침"]
}

DEFAULT_EMOTIONS = CONTEXT_EMOTIONS["기타"]
ALLOWED_CONTEXTS = ["식사", "풍경", "휴식", "이동", "여가", "기타"]

CONFIDENCE_THRESHOLD = 0.55  # 이 값보다 낮으면 애매 판정

# =====================================================
# 유틸
# =====================================================
def image_to_data_url(pil_img: Image.Image) -> str:
    """PIL 이미지를 data URL(base64)로 변환"""
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def make_photo_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:12]

def safe_today() -> datetime.date:
    return datetime.date.today()

# =====================================================
# OpenAI - 이미지 기반 맥락 분류 + confidence
# =====================================================
def classify_context_with_confidence(pil_img: Image.Image) -> tuple[str, float]:
    """
    return: (context, confidence)
    실패하면 ("기타", 0.0)
    """
    if client is None:
        return ("기타", 0.0)

    data_url = image_to_data_url(pil_img)

    prompt = """
너는 사용자의 일상 사진을 아래 카테고리 중 하나로 분류한다.
카테고리: [식사, 풍경, 휴식, 이동, 여가, 기타]

규칙:
- 사진이 애매하거나 여러 활동이 섞였거나 추상적이면 '기타'를 선택한다.
- 반드시 JSON 한 줄로만 출력한다.
형식: {"context":"<카테고리>","confidence":0.0~1.0}
"""

    try:
        res = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )
        text = res.output_text.strip()
        # 매우 단순 파서(안정성 위해 try)
        import json
        obj = json.loads(text)
        ctx = obj.get("context", "기타")
        conf = float(obj.get("confidence", 0.0))
        if ctx not in ALLOWED_CONTEXTS:
            ctx = "기타"
        conf = max(0.0, min(1.0, conf))
        return (ctx, conf)
    except Exception:
        return ("기타", 0.0)

# =====================================================
# Streak / Sticker 로직
# =====================================================
STICKER_RULES = [
    (3, "🥉 3일 연속 기록"),
    (7, "🥈 7일 연속 기록"),
    (14, "🥇 14일 연속 기록"),
    (30, "🏆 30일 연속 기록"),
]

def compute_streak(records_df: pd.DataFrame) -> int:
    """오늘 기준으로 '연속 기록' 계산. (하루에 기록 하나라도 있으면 기록한 날로 침)"""
    if records_df.empty:
        return 0
    days = sorted(set(records_df["date"].dt.date.tolist()))
    if not days:
        return 0

    today = safe_today()
    # 오늘 기록이 없으면 '어제'까지 streak를 보여주고 싶으면 아래를 바꾸면 됨
    # 여기서는 "마지막 기록일부터 연속"으로 계산하되, 마지막 기록이 오늘/어제인지에 따라 달라짐
    last = days[-1]

    # 마지막 기록이 오늘도 아니고 어제도 아니면 streak는 1(마지막 날만)로 처리할지 0으로 처리할지 선택인데,
    # 보통 streak는 끊겼다고 보는 게 자연스러워서 0으로 둠.
    if last not in [today, today - datetime.timedelta(days=1)]:
        return 0

    streak = 1
    cur = last
    dayset = set(days)
    while (cur - datetime.timedelta(days=1)) in dayset:
        streak += 1
        cur = cur - datetime.timedelta(days=1)
    return streak

def award_stickers(streak: int):
    """st.session_state.stickers에 중복 없이 추가"""
    earned = set(st.session_state.stickers)
    for n, label in STICKER_RULES:
        if streak >= n and label not in earned:
            st.session_state.stickers.append(label)

# =====================================================
# 달력 렌더링
# =====================================================
def build_month_calendar(year: int, month: int, day_to_emotion: dict[int, str]) -> str:
    """
    HTML 달력 생성: 각 날짜 칸에 감정 텍스트 표시
    day_to_emotion: {day: "😆 즐거움"} 형태
    """
    cal = calendar.Calendar(firstweekday=0)  # Monday=0 in python? actually 0=Monday in calendar module
    weeks = cal.monthdayscalendar(year, month)

    # 약간의 스타일
    style = """
    <style>
    .cal {border-collapse: collapse; width: 100%; table-layout: fixed;}
    .cal th {padding: 8px; border: 1px solid #ddd; background: #f7f7f7; font-size: 14px;}
    .cal td {vertical-align: top; padding: 8px; border: 1px solid #ddd; height: 86px; font-size: 13px;}
    .cal .day {font-weight: 700; margin-bottom: 6px;}
    .cal .emo {margin-top: 6px; line-height: 1.2;}
    .cal .muted {color: #999;}
    </style>
    """

    # 요일 헤더 (월화수목금토일)
    headers = ["월", "화", "수", "목", "금", "토", "일"]

    html = [style, "<table class='cal'>"]
    html.append("<thead><tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr></thead>")
    html.append("<tbody>")

    for w in weeks:
        html.append("<tr>")
        for d in w:
            if d == 0:
                html.append("<td class='muted'></td>")
            else:
                emo = day_to_emotion.get(d, "")
                emo_html = f"<div class='emo'>{emo}</div>" if emo else "<div class='emo muted'>—</div>"
                html.append(f"<td><div class='day'>{d}</div>{emo_html}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")
    return "\n".join(html)

# =====================================================
# 입력 UI
# =====================================================
st.header("🗓️ 오늘 기록")

record_date = st.date_input(
    "📅 기록할 날짜",
    value=safe_today(),
    max_value=safe_today()
)

energy = st.slider("🔋 오늘의 에너지 (1~10)", 1, 10, 5)

st.subheader("📝 사진 + 감정 (선택)")
images = st.file_uploader(
    "하루의 사진 (최대 3장)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)
images = images[:3]

daily_records = []

# =====================================================
# 리마인드 배너 (사용 빈도 기반)
# =====================================================
if st.session_state.records:
    tmpdf = pd.DataFrame(st.session_state.records)
    tmpdf["date"] = pd.to_datetime(tmpdf["date"])
    last_day = tmpdf["date"].max().date()
    gap = (safe_today() - last_day).days

    if usage_mode == "매일" and gap >= 1:
        st.warning(f"⏰ 마지막 기록이 {gap}일 전({last_day})이에요. 오늘 한 줄이라도 남겨볼까요?")
    elif usage_mode == "주 2~3회" and gap >= 4:
        st.info(f"💡 최근 기록이 조금 뜸해요. 마지막 기록: {last_day}")
else:
    if usage_mode == "매일":
        st.info("👋 첫 기록을 남겨보세요! 매일 한 번이면 충분해요.")

# =====================================================
# 사진별 입력
# =====================================================
for idx, img in enumerate(images):
    st.markdown("---")
    st.subheader(f"사진 {idx + 1}")

    file_bytes = img.getvalue()
    photo_id = make_photo_id(file_bytes)

    image = Image.open(img).convert("RGB")
    st.image(image, use_column_width=True)

    ai_ctx, ai_conf = classify_context_with_confidence(image)

    # 사용자가 맥락을 직접 수정할 수 있게
    st.caption(f"🤖 AI 추천 맥락: **{ai_ctx}** (confidence={ai_conf:.2f})")
    chosen_ctx = st.selectbox(
        "맥락 선택(수정 가능)",
        options=ALLOWED_CONTEXTS,
        index=ALLOWED_CONTEXTS.index(ai_ctx) if ai_ctx in ALLOWED_CONTEXTS else ALLOWED_CONTEXTS.index("기타"),
        key=f"context_{idx}"
    )

    # confidence 낮거나 사용자가 기타로 바꾸면 기본 감정 세트에 더 무게
    emotions = CONTEXT_EMOTIONS.get(chosen_ctx, DEFAULT_EMOTIONS).copy()
    if ai_conf < CONFIDENCE_THRESHOLD:
        # 애매하면 기본 감정도 함께 보여주기(안정성)
        emotions = list(dict.fromkeys(emotions + DEFAULT_EMOTIONS))

    # 사용자 커스텀 감정 누적
    if chosen_ctx in st.session_state.custom_emotions:
        emotions += st.session_state.custom_emotions[chosen_ctx]
        emotions = list(dict.fromkeys(emotions))

    emotion = st.radio(
        "감정 선택 (선택)",
        ["선택 안 함"] + emotions,
        key=f"emotion_{idx}"
    )

    custom = st.text_input("직접 입력 (선택)  예: 😮 뿌듯함 / 😔 아쉬움", key=f"custom_{idx}")

    if custom:
        st.session_state.custom_emotions.setdefault(chosen_ctx, [])
        if custom not in st.session_state.custom_emotions[chosen_ctx]:
            st.session_state.custom_emotions[chosen_ctx].append(custom)
        emotion = custom

    # 기록 저장용 row
    if emotion != "선택 안 함":
        daily_records.append({
            "date": record_date,
            "photo_id": photo_id,
            "context": chosen_ctx,
            "ai_context": ai_ctx,
            "ai_confidence": float(ai_conf),
            "emotion": emotion,
            "energy": energy
        })

# 사진이 없거나, 사진은 있는데 감정 선택을 안했으면 에너지만 기록 가능
if not daily_records:
    st.markdown("---")
    st.caption("사진/감정을 선택하지 않아도 에너지만 기록할 수 있어요.")
    daily_records.append({
        "date": record_date,
        "photo_id": None,
        "context": None,
        "ai_context": None,
        "ai_confidence": None,
        "emotion": None,
        "energy": energy
    })

# =====================================================
# 저장
# =====================================================
if st.button("💾 기록 저장"):
    st.session_state.records.extend(daily_records)
    st.success("기록이 저장되었습니다 ✅")

    # 저장 후 streak/sticker 계산
    df_tmp = pd.DataFrame(st.session_state.records)
    df_tmp["date"] = pd.to_datetime(df_tmp["date"])
    streak = compute_streak(df_tmp)
    award_stickers(streak)

# =====================================================
# 사이드바: streak + stickers
# =====================================================
st.sidebar.header("🔥 연속 기록 / 스티커")
if st.session_state.records:
    df_side = pd.DataFrame(st.session_state.records)
    df_side["date"] = pd.to_datetime(df_side["date"])
    cur_streak = compute_streak(df_side)
    st.sidebar.metric("현재 연속 기록", f"{cur_streak}일")
else:
    st.sidebar.metric("현재 연속 기록", "0일")

if st.session_state.stickers:
    st.sidebar.write("획득한 스티커:")
    for s in st.session_state.stickers[::-1]:
        st.sidebar.write(f"- {s}")
else:
    st.sidebar.write("아직 스티커가 없어요. 3일 연속부터 지급!")

# =====================================================
# 리포트
# =====================================================
st.header("📊 리포트")

if st.session_state.records:
    df = pd.DataFrame(st.session_state.records)
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # -------------------------------------------------
    # 월간 달력 뷰
    # -------------------------------------------------
    st.subheader("🗓️ 한 달 달력 보기")

    # 달 선택: 기록이 있으면 해당 달 우선
    months_available = sorted(df["month"].unique().tolist())
    default_month = df["month"].max()
    selected_month = st.selectbox("표시할 달", months_available, index=months_available.index(default_month))

    year, month = map(int, selected_month.split("-"))
    mdf = df[df["month"] == selected_month].copy()

    # 날짜별 대표 감정(그날 여러 감정이면 최빈값)
    day_to_emotion = {}
    m_emotion = mdf.dropna(subset=["emotion"])
    if not m_emotion.empty:
        for day, g in m_emotion.groupby(m_emotion["date"].dt.day):
            # 최빈 감정
            emo = Counter(g["emotion"].tolist()).most_common(1)[0][0]
            day_to_emotion[int(day)] = emo

    cal_html = build_month_calendar(year, month, day_to_emotion)
    st.markdown(cal_html, unsafe_allow_html=True)

    # -------------------------------------------------
    # 에너지 리포트
    # -------------------------------------------------
    st.subheader("🔋 에너지 리포트")

    for period, label in [("week", "주별"), ("month", "월별")]:
        st.markdown(f"### {label}")

        grouped = df.groupby(period)
        energy_df = grouped["energy"].mean().reset_index()

        chart = (
            alt.Chart(energy_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{period}:O", title=label),
                y=alt.Y("energy:Q", title="평균 에너지", scale=alt.Scale(domain=[1, 10]))
            )
        )
        st.altair_chart(chart, use_container_width=True)

        mode_energy = grouped["energy"].agg(lambda x: Counter(x).most_common(1)[0][0])
        st.write("📌 최빈 에너지")
        st.dataframe(mode_energy)

        max_day = grouped.apply(lambda x: x.loc[x["energy"].idxmax(), "date"])
        st.write("⚡ 가장 에너지가 높았던 날")
        st.dataframe(max_day)

    # -------------------------------------------------
    # 감정 리포트 + 활동(맥락)별 감정 비율
    # -------------------------------------------------
    st.subheader("💭 감정 리포트")

    emotion_df = df.dropna(subset=["emotion"]).copy()
    if not emotion_df.empty:
        # (1) 주/월별 감정 빈도
        for period, label in [("week", "주별"), ("month", "월별")]:
            st.markdown(f"### {label} 감정 빈도")

            freq = emotion_df.groupby([period, "emotion"]).size().reset_index(name="count")
            chart = (
                alt.Chart(freq)
                .mark_bar()
                .encode(
                    x=alt.X("emotion:N", title="감정"),
                    y=alt.Y("count:Q", title="빈도"),
                    color="emotion:N"
                )
            )
            st.altair_chart(chart, use_container_width=True)

            most_common = emotion_df.groupby(period)["emotion"].agg(lambda x: Counter(x).most_common(1)[0][0])
            st.write("📌 가장 많이 선택된 감정")
            st.dataframe(most_common)

        # (2) 활동 유형별 감정 비율(맥락이 있는 기록만)
        st.markdown("### 활동(맥락) 유형별 감정 비율")
        ctx_df = emotion_df.dropna(subset=["context"]).copy()
        if not ctx_df.empty:
            ctx_freq = ctx_df.groupby(["context", "emotion"]).size().reset_index(name="count")

            # 비율 계산
            ctx_total = ctx_freq.groupby("context")["count"].transform("sum")
            ctx_freq["ratio"] = ctx_freq["count"] / ctx_total

            chart2 = (
                alt.Chart(ctx_freq)
                .mark_bar()
                .encode(
                    x=alt.X("context:N", title="활동(맥락)"),
                    y=alt.Y("ratio:Q", title="비율"),
                    color="emotion:N",
                    tooltip=["context", "emotion", "count", alt.Tooltip("ratio:Q", format=".0%")]
                )
            )
            st.altair_chart(chart2, use_container_width=True)
        else:
            st.info("맥락이 저장된 감정 기록이 아직 없어요.")

    else:
        st.info("감정 기록이 없어 에너지 리포트만 표시됩니다.")

else:
    st.info("아직 저장된 기록이 없습니다.")
