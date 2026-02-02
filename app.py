import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
import datetime
import pandas as pd

# =====================================================
# 기본 설정
# =====================================================
st.set_page_config(page_title="하루의 단서", layout="centered")
st.title("📸 하루의 단서")

# =====================================================
# 사이드바 - API KEY
# =====================================================
st.sidebar.header("🔑 API 설정")

openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
weather_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")

client = OpenAI(api_key=openai_key) if openai_key else None

# =====================================================
# 세션 상태 (데이터 저장용)
# =====================================================
if "records" not in st.session_state:
    st.session_state.records = []

if "custom_emotions" not in st.session_state:
    st.session_state.custom_emotions = {}

# =====================================================
# 맥락별 감정 풀 (오류 방지 핵심)
# =====================================================
CONTEXT_EMOTIONS = {
    "식사": ["😋 맛있음", "🙂 괜찮았음", "😕 아쉬움", "💸 가격이 아까움"],
    "풍경": ["🌿 차분함", "✨ 인상 깊음", "🙂 그냥 그랬음"],
    "휴식": ["😌 편안함", "🙂 만족스러움", "😴 나른함"],
    "이동": ["😴 피곤함", "😐 무난함", "😤 지침"],
    "여가": ["😆 즐거움", "🙂 만족", "😐 평범함"]
}

# =====================================================
# OpenAI - 사진 맥락 분류만
# =====================================================
def classify_context():
    if client is None:
        return "일상"
    prompt = """
    이 이미지는 사용자의 일상 사진이다.
    다음 중 하나로만 분류하라:
    [식사, 풍경, 휴식, 이동, 여가]
    단어 하나만 출력하라.
    """
    res = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    return res.output_text.strip()

# =====================================================
# 기록 날짜 (과거만 허용)
# =====================================================
record_date = st.date_input(
    "📅 기록할 날짜",
    value=datetime.date.today(),
    max_value=datetime.date.today()
)

# =====================================================
# 하루 에너지 점수 (1회)
# =====================================================
energy = st.slider(
    "🔋 오늘의 에너지 수준",
    min_value=1,
    max_value=10,
    value=5
)

# =====================================================
# 사진 기록
# =====================================================
st.header("📝 오늘의 기록")

images = st.file_uploader(
    "하루의 사진 (최대 3장)",
    type=["jpg", "png"],
    accept_multiple_files=True
)

images = images[:3]
daily_records = []

for idx, img in enumerate(images):
    st.subheader(f"사진 {idx + 1}")
    image = Image.open(img)
    st.image(image, use_column_width=True)

    context = classify_context()
    emotions = CONTEXT_EMOTIONS.get(context, ["🙂 평범함"])

    # 사용자 주관식 감정 재사용
    if context in st.session_state.custom_emotions:
        emotions += st.session_state.custom_emotions[context]

    emotion = st.radio(
        "감정 선택 (1개)",
        emotions,
        key=f"emotion_{idx}"
    )

    custom = st.text_input(
        "직접 입력 (선택)",
        key=f"custom_{idx}"
    )

    if custom:
        st.session_state.custom_emotions.setdefault(context, [])
        if custom not in st.session_state.custom_emotions[context]:
            st.session_state.custom_emotions[context].append(custom)
        emotion = custom

    daily_records.append({
        "date": record_date,
        "context": context,
        "emotion": emotion,
        "energy": energy
    })

# =====================================================
# 저장
# =====================================================
if st.button("💾 기록 저장"):
    st.session_state.records.extend(daily_records)
    st.success("기록이 저장되었습니다")

# =====================================================
# 리포트
# =====================================================
st.header("📊 리포트")

if st.session_state.records:
    df = pd.DataFrame(st.session_state.records)
    df["date"] = pd.to_datetime(df["date"])

    # 주별 리포트
    st.subheader("🗓️ 주별 리포트")
    df["week"] = df["date"].dt.isocalendar().week
    weekly = df.groupby(["week", "emotion"]).size().unstack(fill_value=0)
    st.bar_chart(weekly)

    st.line_chart(df.groupby("week")["energy"].mean())

    # 월별 리포트
    st.subheader("📆 월별 리포트")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby(["month", "emotion"]).size().unstack(fill_value=0)
    st.bar_chart(monthly)

    st.line_chart(df.groupby("month")["energy"].mean())

else:
    st.info("아직 저장된 기록이 없습니다")
