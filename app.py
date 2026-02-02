import streamlit as st
from openai import OpenAI
from PIL import Image
import datetime
import pandas as pd
from collections import Counter

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
client = OpenAI(api_key=openai_key) if openai_key else None

# =====================================================
# 세션 상태
# =====================================================
if "records" not in st.session_state:
    st.session_state.records = []

if "custom_emotions" not in st.session_state:
    st.session_state.custom_emotions = {}

# =====================================================
# 맥락별 감정 풀
# =====================================================
CONTEXT_EMOTIONS = {
    "식사": ["😋 맛있음", "🙂 괜찮았음", "😕 아쉬움", "💸 가격이 아까움"],
    "풍경": ["🌿 차분함", "✨ 인상 깊음", "🙂 그냥 그랬음"],
    "휴식": ["😌 편안함", "🙂 만족스러움", "😴 나른함"],
    "이동": ["😴 피곤함", "😐 무난함", "😤 지침"],
    "여가": ["😆 즐거움", "🙂 만족", "😐 평범함"]
}

# =====================================================
# OpenAI - 맥락 분류 (단어 하나)
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
# 날짜 (과거만 허용)
# =====================================================
record_date = st.date_input(
    "📅 기록할 날짜",
    value=datetime.date.today(),
    max_value=datetime.date.today()
)

# =====================================================
# 에너지 체크 (필수)
# =====================================================
energy = st.slider(
    "🔋 오늘의 에너지 (1~10)",
    1, 10, 5
)

# =====================================================
# 사진 + 감정 기록 (선택)
# =====================================================
st.header("📝 오늘의 기록 (선택)")

images = st.file_uploader(
    "하루의 사진 (최대 3장, 선택)",
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

    if context in st.session_state.custom_emotions:
        emotions += st.session_state.custom_emotions[context]

    emotion = st.radio(
        "감정 선택 (선택)",
        ["선택 안 함"] + emotions,
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

    if emotion != "선택 안 함":
        daily_records.append({
            "date": record_date,
            "context": context,
            "emotion": emotion,
            "energy": energy
        })

# =====================================================
# 에너지 단독 기록도 저장
# =====================================================
if not daily_records:
    daily_records.append({
        "date": record_date,
        "context": None,
        "emotion": None,
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
    df["week"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # -----------------------------
    # 에너지 리포트
    # -----------------------------
    st.subheader("🔋 에너지 리포트")

    for period, label in [("week", "주별"), ("month", "월별")]:
        st.markdown(f"### {label}")

        grouped = df.groupby(period)

        avg_energy = grouped["energy"].mean()
        mode_energy = grouped["energy"].agg(lambda x: Counter(x).most_common(1)[0][0])
        max_day = grouped.apply(lambda x: x.loc[x["energy"].idxmax(), "date"])

        st.write("📈 평균 에너지")
        st.line_chart(avg_energy)

        st.write("📌 최빈 에너지")
        st.dataframe(mode_energy)

        st.write("⚡ 가장 에너지가 높았던 날")
        st.dataframe(max_day)

    # -----------------------------
    # 감정 리포트
    # -----------------------------
    st.subheader("💭 감정 리포트")

    emotion_df = df.dropna(subset=["emotion"])

    if not emotion_df.empty:
        for period, label in [("week", "주별"), ("month", "월별")]:
            st.markdown(f"### {label}")

            freq = emotion_df.groupby([period, "emotion"]).size().unstack(fill_value=0)
            st.bar_chart(freq)

            most_common = emotion_df.groupby(period)["emotion"].agg(
                lambda x: Counter(x).most_common(1)[0][0]
            )
            st.write("📌 가장 많이 선택된 감정")
            st.dataframe(most_common)

    else:
        st.info("아직 감정 기록이 없습니다. 에너지 리포트만 표시됩니다.")

else:
    st.info("아직 저장된 기록이 없습니다.")
