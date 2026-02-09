import streamlit as st
from openai import OpenAI
from PIL import Image
import datetime
import pandas as pd
import altair as alt
from collections import Counter
import base64
import hashlib
import calendar
import random
import requests

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

# 포켓몬 도감: {pokemon_id: {"id":..,"name":..,"sprite":..}}
if "pokedex" not in st.session_state:
    st.session_state.pokedex = {}

# 날짜별 포켓몬 획득 여부(하루 1마리 제한): set(["YYYY-MM-DD", ...])
if "pokemon_claimed_dates" not in st.session_state:
    st.session_state.pokemon_claimed_dates = set()

# 달력에서 선택한 날짜
if "selected_calendar_date" not in st.session_state:
    st.session_state.selected_calendar_date = None

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
CONFIDENCE_THRESHOLD = 0.55

# =====================================================
# 유틸
# =====================================================
def safe_today() -> datetime.date:
    return datetime.date.today()

def image_to_data_url(pil_img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def pil_to_b64_png(pil_img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_pil(b64_str: str) -> Image.Image:
    import io
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw))

def make_photo_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:12]

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
# 포켓몬 (PokeAPI) - 1세대 랜덤 획득
# =====================================================
def get_pokemon() -> dict:
    """
    PokeAPI에서 1~151 랜덤 포켓몬 가져오기
    return: {"id": int, "name": str, "sprite": str|None}
    """
    poke_id = random.randint(1, 151)
    url = f"https://pokeapi.co/api/v2/pokemon/{poke_id}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    sprite = None
    # 기본 스프라이트(정면)
    if data.get("sprites"):
        sprite = data["sprites"].get("front_default")
    return {"id": data["id"], "name": data["name"], "sprite": sprite}

def claim_pokemon_for_date(date_obj: datetime.date) -> tuple[bool, dict | None, str | None]:
    """
    해당 날짜에 포켓몬을 아직 안 받았으면 지급.
    return: (claimed_now, pokemon_dict, error_msg)
    """
    date_key = date_obj.isoformat()
    if date_key in st.session_state.pokemon_claimed_dates:
        return (False, None, None)

    try:
        p = get_pokemon()
        st.session_state.pokemon_claimed_dates.add(date_key)
        # 도감에 등록(중복 포켓몬이면 이미 있던 걸 유지)
        if p["id"] not in st.session_state.pokedex:
            st.session_state.pokedex[p["id"]] = p
        return (True, p, None)
    except Exception as e:
        return (False, None, f"포켓몬 지급 중 오류: {e}")

# =====================================================
# 리마인드 배너 (사용 빈도 기반)
# =====================================================
def show_reminder_banner():
    if not st.session_state.records:
        if usage_mode == "매일":
            st.info("👋 첫 기록을 남겨보세요! 사진 없어도 에너지 기록만으로 포켓몬을 얻을 수 있어요.")
        return

    tmpdf = pd.DataFrame(st.session_state.records)
    tmpdf["date"] = pd.to_datetime(tmpdf["date"])
    last_day = tmpdf["date"].max().date()
    gap = (safe_today() - last_day).days

    if usage_mode == "매일" and gap >= 1:
        st.warning(f"⏰ 마지막 기록이 {gap}일 전({last_day})이에요. 오늘 에너지라도 남기고 포켓몬 받자!")
    elif usage_mode == "주 2~3회" and gap >= 4:
        st.info(f"💡 최근 기록이 조금 뜸해요. 마지막 기록: {last_day}")

# =====================================================
# 달력 UI (버튼 클릭)
# =====================================================
def render_month_calendar_buttons(year: int, month: int, day_to_label: dict[int, str]):
    """
    7열 그리드 버튼 달력.
    day_to_label: 각 날짜에 표시할 감정 라벨(짧게)
    """
    st.write(f"**{year}년 {month}월**")

    headers = ["월", "화", "수", "목", "금", "토", "일"]
    cols = st.columns(7)
    for i, h in enumerate(headers):
        cols[i].markdown(f"**{h}**")

    cal = calendar.Calendar(firstweekday=0)  # 월요일 시작
    weeks = cal.monthdayscalendar(year, month)

    for w in weeks:
        row = st.columns(7)
        for i, d in enumerate(w):
            if d == 0:
                row[i].markdown(" ")
                continue

            label = day_to_label.get(d, "—")
            # 버튼 텍스트를 너무 길게 하지 않기 위해 줄바꿈
            btn_text = f"{d}\n{label}"
            key = f"calbtn_{year}_{month}_{d}"

            if row[i].button(btn_text, key=key):
                st.session_state.selected_calendar_date = datetime.date(year, month, d)

# =====================================================
# 입력 UI
# =====================================================
st.header("🗓️ 오늘 기록")
show_reminder_banner()

record_date = st.date_input(
    "📅 기록할 날짜",
    value=safe_today(),
    max_value=safe_today()
)

energy = st.slider("🔋 오늘의 에너지 (1~10)", 1, 10, 5)

st.subheader("📝 사진 + 감정 (선택)")
images = st.file_uploader(
    "하루의 사진 (최대 3장) - 없어도 OK",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)
images = (images or [])[:3]

daily_records = []

# 사진별 입력
for idx, img in enumerate(images):
    st.markdown("---")
    st.subheader(f"사진 {idx + 1}")

    file_bytes = img.getvalue()
    photo_id = make_photo_id(file_bytes)

    image = Image.open(img).convert("RGB")
    st.image(image, use_column_width=True)

    ai_ctx, ai_conf = classify_context_with_confidence(image)

    st.caption(f"🤖 AI 추천 맥락: **{ai_ctx}** (confidence={ai_conf:.2f})")
    chosen_ctx = st.selectbox(
        "맥락 선택(수정 가능)",
        options=ALLOWED_CONTEXTS,
        index=ALLOWED_CONTEXTS.index(ai_ctx) if ai_ctx in ALLOWED_CONTEXTS else ALLOWED_CONTEXTS.index("기타"),
        key=f"context_{idx}"
    )

    emotions = CONTEXT_EMOTIONS.get(chosen_ctx, DEFAULT_EMOTIONS).copy()
    if ai_conf < CONFIDENCE_THRESHOLD:
        emotions = list(dict.fromkeys(emotions + DEFAULT_EMOTIONS))

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

    # 사진 자체도 저장(달력 상세 보기용)
    image_b64 = pil_to_b64_png(image)

    # 감정 선택 안 해도 사진 기록은 남길지 여부는 취향인데,
    # 여기서는 "감정 선택한 경우만" 감정 row로 저장하고,
    # 사진은 저장하되 emotion=None으로 저장해도 상세 보기에는 보일 수 있게 하자.
    daily_records.append({
        "date": record_date,
        "photo_id": photo_id,
        "image_b64": image_b64,
        "context": chosen_ctx,
        "ai_context": ai_ctx,
        "ai_confidence": float(ai_conf),
        "emotion": None if emotion == "선택 안 함" else emotion,
        "energy": energy
    })

# 사진이 없으면 에너지만 기록
if not images:
    st.caption("사진을 올리지 않아도 에너지 기록만으로 포켓몬을 얻을 수 있어요.")
    daily_records.append({
        "date": record_date,
        "photo_id": None,
        "image_b64": None,
        "context": None,
        "ai_context": None,
        "ai_confidence": None,
        "emotion": None,
        "energy": energy
    })

# =====================================================
# 저장 (저장 시 포켓몬 지급)
# =====================================================
if st.button("💾 기록 저장"):
    st.session_state.records.extend(daily_records)
    st.success("기록이 저장되었습니다 ✅")

    claimed, p, err = claim_pokemon_for_date(record_date)
    if err:
        st.error(err)
    elif claimed:
        st.balloons()
        st.success(f"🎁 오늘의 포켓몬 GET!  #{p['id']}  {p['name']}")
        if p.get("sprite"):
            st.image(p["sprite"], width=120)
    else:
        st.info("오늘은 이미 포켓몬을 받았어요! (하루 1마리)")

# =====================================================
# 사이드바: 포켓몬 진행도
# =====================================================
st.sidebar.header("🧡 포켓몬 도감")
st.sidebar.metric("획득", f"{len(st.session_state.pokedex)}/151")
if st.session_state.pokedex:
    # 최근 획득 몇 개 보여주기(최대 5)
    recent = sorted(st.session_state.pokedex.values(), key=lambda x: x["id"], reverse=True)[:5]
    st.sidebar.write("최근 도감 등록:")
    for p in recent:
        if p.get("sprite"):
            st.sidebar.image(p["sprite"], width=60)
        st.sidebar.write(f"#{p['id']} {p['name']}")

# =====================================================
# 리포트
# =====================================================
st.header("📊 리포트")

if not st.session_state.records:
    st.info("아직 저장된 기록이 없습니다.")
    st.stop()

df = pd.DataFrame(st.session_state.records)
df["date"] = pd.to_datetime(df["date"])
df["week"] = df["date"].dt.isocalendar().week
df["month"] = df["date"].dt.to_period("M").astype(str)

# -------------------------------------------------
# 월간 달력 (클릭 가능)
# -------------------------------------------------
st.subheader("🗓️ 한 달 달력 보기 (날짜 클릭 → 상세)")
months_available = sorted(df["month"].unique().tolist())
default_month = df["month"].max()
selected_month = st.selectbox("표시할 달", months_available, index=months_available.index(default_month))

year, month = map(int, selected_month.split("-"))
mdf = df[df["month"] == selected_month].copy()

# 날짜별 대표 감정(그날 여러 감정이면 최빈)
day_to_label = {}
m_emotion = mdf.dropna(subset=["emotion"])
if not m_emotion.empty:
    for day, g in m_emotion.groupby(m_emotion["date"].dt.day):
        emo = Counter(g["emotion"].tolist()).most_common(1)[0][0]
        # 달력 라벨은 너무 길면 보기 힘드니 앞쪽만
        day_to_label[int(day)] = emo

render_month_calendar_buttons(year, month, day_to_label)

# -------------------------------------------------
# 날짜 클릭 시 상세 보기
# -------------------------------------------------
st.markdown("---")
st.subheader("🔎 선택한 날짜 상세 보기")

if st.session_state.selected_calendar_date is None:
    st.info("달력에서 날짜를 클릭해 주세요.")
else:
    sel = st.session_state.selected_calendar_date
    st.write(f"**선택 날짜:** {sel.isoformat()}")

    day_df = df[df["date"].dt.date == sel].copy()
    if day_df.empty:
        st.info("해당 날짜 기록이 없습니다.")
    else:
        # 에너지(그날 여러 row면 동일하다고 가정하지만 안전하게 최빈/평균)
        st.metric("에너지", float(day_df["energy"].mean()))

        # 포켓몬 획득 여부
        if sel.isoformat() in st.session_state.pokemon_claimed_dates:
            st.success("🎁 이 날은 포켓몬을 획득한 날이에요!")
        else:
            st.warning("이 날은 포켓몬을 아직 못 받았어요(기록 저장하면 받을 수 있음).")

        # 사진/감정 리스트
        # photo_id 없는(에너지만) row는 별도 표시
        photo_rows = day_df[day_df["image_b64"].notna()].copy()
        energy_only = day_df[day_df["image_b64"].isna()].copy()

        if not energy_only.empty and photo_rows.empty:
            st.info("사진 없이 에너지만 기록한 날이에요.")

        if not photo_rows.empty:
            st.write("**사진 기록**")
            for i, r in photo_rows.iterrows():
                cols = st.columns([1, 2])
                # 이미지
                try:
                    pil = b64_to_pil(r["image_b64"])
                    cols[0].image(pil, use_column_width=True)
                except Exception:
                    cols[0].write("(이미지 표시 실패)")

                # 메타
                ctx = r.get("context")
                emo = r.get("emotion")
                ai_ctx = r.get("ai_context")
                ai_conf = r.get("ai_confidence")

                meta_lines = []
                if ctx:
                    meta_lines.append(f"- 맥락: **{ctx}**")
                if emo:
                    meta_lines.append(f"- 감정: **{emo}**")
                else:
                    meta_lines.append(f"- 감정: (선택 안 함)")
                if ai_ctx:
                    meta_lines.append(f"- AI 추천: {ai_ctx} (conf={ai_conf:.2f})" if ai_conf is not None else f"- AI 추천: {ai_ctx}")
                cols[1].markdown("\n".join(meta_lines))

# -------------------------------------------------
# 에너지 리포트
# -------------------------------------------------
st.markdown("---")
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

# -------------------------------------------------
# 감정 + 활동(맥락) 리포트
# -------------------------------------------------
st.markdown("---")
st.subheader("💭 감정 리포트")

emotion_df = df.dropna(subset=["emotion"]).copy()
if emotion_df.empty:
    st.info("감정 기록이 없어 에너지 리포트만 표시됩니다.")
else:
    # 주/월별 감정 빈도
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

    # 활동(맥락) 유형별 감정 비율
    st.markdown("### 활동(맥락) 유형별 감정 비율")
    ctx_df = emotion_df.dropna(subset=["context"]).copy()
    if ctx_df.empty:
        st.info("맥락이 저장된 감정 기록이 아직 없어요.")
    else:
        ctx_freq = ctx_df.groupby(["context", "emotion"]).size().reset_index(name="count")
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

# =====================================================
# 맨 아래: 내 포켓몬 도감
# =====================================================
st.markdown("---")
st.header("📚 나의 포켓몬 도감 (획득한 포켓몬)")

if not st.session_state.pokedex:
    st.info("아직 획득한 포켓몬이 없어요. 기록 저장하면 하루 1마리씩 얻을 수 있어요!")
else:
    pokes = sorted(st.session_state.pokedex.values(), key=lambda x: x["id"])
    st.write(f"총 **{len(pokes)} / 151** 마리")

    # 그리드 표시(4열)
    cols_per_row = 4
    for i in range(0, len(pokes), cols_per_row):
        row = st.columns(cols_per_row)
        chunk = pokes[i:i+cols_per_row]
        for j in range(cols_per_row):
            if j >= len(chunk):
                row[j].write("")
                continue
            p = chunk[j]
            if p.get("sprite"):
                row[j].image(p["sprite"], width=120)
            row[j].markdown(f"**#{p['id']} {p['name']}**")

    st.caption("포켓몬 이름은 PokeAPI 원문(영문)입니다. 원하면 한글 이름 매핑도 붙여줄게요.")
