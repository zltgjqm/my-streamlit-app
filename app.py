import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
import datetime

# =====================================================
# 기본 설정
# =====================================================
st.set_page_config(page_title="하루의 단서", layout="centered")
st.title("📸 하루의 단서")
st.caption("사진과 선택으로 하루의 감정 흐름을 기록합니다")

# =====================================================
# 사이드바 - API KEY 입력
# =====================================================
st.sidebar.header("🔑 API 설정")

openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
weather_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")

client = OpenAI(api_key=openai_key) if openai_key else None

# =====================================================
# 세션 상태
# =====================================================
if "custom_emotions" not in st.session_state:
    st.session_state.custom_emotions = []

# =====================================================
# 날씨 API (맥락 정보용)
# =====================================================
def get_weather(city="Seoul"):
    if not weather_key:
        return None
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={weather_key}&units=metric&lang=kr"
    )
    res = requests.get(url).json()
    if "weather" in res:
        return {
            "desc": res["weather"][0]["description"],
            "temp": res["main"]["temp"]
        }
    return None

weather = get_weather()

# =====================================================
# OpenAI - 감정 선택지 생성
# =====================================================
def generate_emotions(context_text):
    """
    감정을 추론하지 않고
    선택 가능한 표현만 생성
    """
    # API 키 없을 때 기본값
    if client is None:
        return ["🙂 평범함", "😐 그냥 그랬음", "😌 차분함", "😴 피곤함"]

    prompt = f"""
    사용자의 일상 기록을 위한 감정 선택지를 생성하라.

    규칙:
    - 감정을 추론하거나 판단하지 말 것
    - 중립적인 표현 사용
    - 아이콘 + 짧은 텍스트
    - 4~6개만 제시

    상황 설명:
    {context_text}
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    text = response.output_text
    emotions = []

    for line in text.split("\n"):
        line = line.strip()
        if line:
            emotions.append(line)

    return emotions[:6]

# =====================================================
# 기록 입력 UI
# =====================================================
st.header("📝 오늘의 기록")

mode = st.radio(
    "기록 방식 선택",
    ["사진으로 기록", "사진 없이 감정만 기록"]
)

records = []

# -------------------------------
# 📸 사진으로 기록
# -------------------------------
if mode == "사진으로 기록":
    images = st.file_uploader(
        "하루의 사진 (최대 3장)",
        type=["jpg", "png"],
        accept_multiple_files=True
    )

    images = images[:3]

    for idx, img in enumerate(images):
        st.subheader(f"사진 {idx + 1}")
        image = Image.open(img)
        st.image(image, use_column_width=True)

        emotions = generate_emotions("일상 기록 사진")
        emotions += st.session_state.custom_emotions

        choice = st.radio(
            "감정 선택 (1개)",
            emotions,
            key=f"emotion_{idx}"
        )

        custom = st.text_input(
            "직접 입력 (선택)",
            key=f"custom_{idx}"
        )

        if custom:
            if custom not in st.session_state.custom_emotions:
                st.session_state.custom_emotions.append(custom)
            choice = custom

        records.append({
            "type": "photo",
            "emotion": choice
        })

# -------------------------------
# ✏️ 사진 없이 기록
# -------------------------------
else:
    context = st.selectbox(
        "오늘의 상황",
        ["일상", "휴식", "이동", "여가", "기타"]
    )

    emotions = generate_emotions(context)
    emotions += st.session_state.custom_emotions

    choice = st.radio("감정 선택 (1개)", emotions)

    custom = st.text_input("직접 입력 (선택)")
    if custom:
        if custom not in st.session_state.custom_emotions:
            st.session_state.custom_emotions.append(custom)
        choice = custom

    records.append({
        "type": "text_only",
        "emotion": choice
    })

# =====================================================
# 저장
# =====================================================
if st.button("💾 오늘의 단서 저장"):
    st.success("오늘의 기록을 저장했어요")

    st.markdown("### 📌 기록 요약")
    for r in records:
        st.write(f"- 감정: {r['emotion']}")

    if weather:
        st.write(f"🌤️ 날씨: {weather['desc']} / {weather['temp']}°C")

    st.caption(f"📅 날짜: {datetime.date.today()}")

    st.markdown("---")
    st.caption(
        "AI는 감정을 판단하지 않으며, "
        "사용자가 선택한 표현만을 데이터로 저장합니다."
    )
