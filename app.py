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
    st.session_state.records = []

if "custom_emotions" not in st.session_state:
    st.session_state.custom_emotions = {}

# 도감: {pokemon_id: {"id":..,"name_ko":..,"name_en":..,"sprite":..,"caught":int}}
if "pokedex" not in st.session_state:
    st.session_state.pokedex = {}

# 날짜별 포켓몬 지급 여부
if "pokemon_claimed_dates" not in st.session_state:
    st.session_state.pokemon_claimed_dates = set()

# 달력에서 선택한 날짜
if "selected_calendar_date" not in st.session_state:
    st.session_state.selected_calendar_date = None

# =====================================================
# 감정 풀
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
def safe_today():
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
# streak 계산
# =====================================================
def compute_streak(records_df: pd.DataFrame) -> int:
    if records_df.empty:
        return 0

    days = sorted(set(records_df["date"].dt.date.tolist()))
    if not days:
        return 0

    today = safe_today()
    last = days[-1]

    if last not in [today, today - datetime.timedelta(days=1)]:
        return 0

    streak = 1
    cur = last
    dayset = set(days)

    while (cur - datetime.timedelta(days=1)) in dayset:
        streak += 1
        cur = cur - datetime.timedelta(days=1)

    return streak

# =====================================================
# 포켓몬 API
# =====================================================
def get_pokemon_name_ko(species_url: str) -> str:
    """
    species API에서 한국어 이름 가져오기
    """
    r = requests.get(species_url, timeout=10)
    r.raise_for_status()
    data = r.json()

    for name_obj in data.get("names", []):
        if name_obj["language"]["name"] == "ko":
            return name_obj["name"]

    return data.get("name", "???")

def get_pokemon() -> dict:
    """
    1세대 랜덤 포켓몬
    return: {"id":int, "name_ko":str, "name_en":str, "sprite":str|None}
    """
    poke_id = random.randint(1, 151)
    url = f"https://pokeapi.co/api/v2/pokemon/{poke_id}"

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    sprite = None
    if data.get("sprites"):
        sprite = data["sprites"].get("front_default")

    name_en = data.get("name", "???")
    species_url = data.get("species", {}).get("url")

    name_ko = name_en
    if species_url:
        name_ko = get_pokemon_name_ko(species_url)

    return {"id": data["id"], "name_ko": name_ko, "name_en": name_en, "sprite": sprite}

def add_to_pokedex(p: dict):
    """
    도감 등록 + caught count 증가
    """
    pid = p["id"]
    if pid not in st.session_state.pokedex:
        st.session_state.pokedex[pid] = {
            "id": pid,
            "name_ko": p["name_ko"],
            "name_en": p["name_en"],
            "sprite": p["sprite"],
            "caught": 1
        }
    else:
        st.session_state.pokedex[pid]["caught"] += 1

def claim_pokemon(date_obj: datetime.date, count: int = 1):
    """
    count 마리 지급
    """
    results = []
    for _ in range(count):
        p = get_pokemon()
        add_to_pokedex(p)
        results.append(p)
    return results

# =====================================================
# 리마인드 배너
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
# 달력 렌더링 (버튼)
# =====================================================
def render_month_calendar_buttons(year: int, month: int, day_to_label: dict[int, str]):
    st.write(f"**{year}년 {month}월**")

    headers = ["월", "화", "수", "목", "금", "토", "일"]
    cols = st.columns(7)
    for i, h in enumerate(headers):
        cols[i].markdown(f"**{h}**")

    cal = calendar.Calendar(firstweekday=0)
    weeks = cal.monthdayscalendar(year, month)

    for w in weeks:
        row = st.columns(7)
        for i, d in enumerate(w):
            if d == 0:
                row[i].markdown(" ")
                continue

            label = day_to_label.get(d, "—")
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
    st.image(image, use_container_width=True)

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

    image_b64 = pil_to_b64_png(image)

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
# 저장 (포켓몬 지급)
# =====================================================
if st.button("💾 기록 저장"):
    st.session_state.records.extend(daily_records)
    st.success("기록이 저장되었습니다 ✅")

    date_key = record_date.isoformat()

    # 하루에 1번만 지급
    if date_key in st.session_state.pokemon_claimed_dates:
        st.info("오늘은 이미 포켓몬을 받았어요! (하루 1회 지급)")
    else:
        # 기록 날짜에 대해 지급 처리
        st.session_state.pokemon_claimed_dates.add(date_key)

        # streak 계산
        df_tmp = pd.DataFrame(st.session_state.records)
        df_tmp["date"] = pd.to_datetime(df_tmp["date"])
        streak = compute_streak(df_tmp)

        # 기본 1마리 + streak가 3의 배수면 추가 1마리
        bonus = 1 if (streak > 0 and streak % 3 == 0) else 0
        total = 1 + bonus

        try:
            pokes = claim_pokemon(record_date, count=total)
            st.balloons()

            if bonus:
                st.success(f"🔥 연속 {streak}일 달성! 보너스 포켓몬 포함 총 {total}마리 획득!")
            else:
                st.success(f"🎁 오늘의 포켓몬 GET! ({total}마리)")

            for p in pokes:
                st.write(f"#{p['id']} **{p['name_ko']}** ({p['name_en']})")
                if p.get("sprite"):
                    st.image(p["sprite"], width=120)

        except Exception as e:
            st.error(f"포켓몬 지급 실패: {e}")

# =====================================================
# 사이드바: 포켓몬 진행도
# =====================================================
st.sidebar.header("🧡 포켓몬 도감")
st.sidebar.metric("등록", f"{len(st.session_state.pokedex)}/151")

if st.session_state.pokedex:
    recent = sorted(st.session_state.pokedex.values(), key=lambda x: x["id"], reverse=True)[:5]
    st.sidebar.write("최근 등록:")
    for p in recent:
        if p.get("sprite"):
            st.sidebar.image(p["sprite"], width=60)
        st.sidebar.write(f"#{p['id']} {p['name_ko']} (x{p['caught']})")

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
# 달력
# -------------------------------------------------
st.subheader("🗓️ 한 달 달력 보기 (날짜 클릭 → 상세)")
months_available = sorted(df["month"].unique().tolist())
default_month = df["month"].max()
selected_month = st.selectbox("표시할 달", months_available, index=months_available.index(default_month))

year, month = map(int, selected_month.split("-"))
mdf = df[df["month"] == selected_month].copy()

day_to_label = {}
m_emotion = mdf.dropna(subset=["emotion"])
if not m_emotion.empty:
    for day, g in m_emotion.groupby(m_emotion["date"].dt.day):
        emo = Counter(g["emotion"].tolist()).most_common(1)[0][0]
        day_to_label[int(day)] = emo

render_month_calendar_buttons(year, month, day_to_label)

# -------------------------------------------------
# 날짜 클릭 상세 보기
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
        st.metric("에너지", float(day_df["energy"].mean()))

        if sel.isoformat() in st.session_state.pokemon_claimed_dates:
            st.success("🎁 이 날은 포켓몬을 획득한 날이에요!")
        else:
            st.warning("이 날은 포켓몬을 아직 못 받았어요.")

        photo_rows = day_df[day_df["image_b64"].notna()].copy()
        energy_only = day_df[day_df["image_b64"].isna()].copy()

        if not energy_only.empty and photo_rows.empty:
            st.info("사진 없이 에너지만 기록한 날이에요.")

        if not photo_rows.empty:
            st.write("**사진 기록**")
            for _, r in photo_rows.iterrows():
                cols = st.columns([1, 2])

                try:
                    pil = b64_to_pil(r["image_b64"])
                    cols[0].image(pil, use_container_width=True)
                except Exception:
                    cols[0].write("(이미지 표시 실패)")

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
# 감정 리포트 + 활동 비율
# -------------------------------------------------
st.markdown("---")
st.subheader("💭 감정 리포트")

emotion_df = df.dropna(subset=["emotion"]).copy()
if emotion_df.empty:
    st.info("감정 기록이 없어 에너지 리포트만 표시됩니다.")
else:
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
# 도감
# =====================================================
st.markdown("---")
st.header("📚 나의 포켓몬 도감")

if not st.session_state.pokedex:
    st.info("아직 획득한 포켓몬이 없어요. 기록 저장하면 하루 1마리씩 얻을 수 있어요!")
else:
    pokes = sorted(st.session_state.pokedex.values(), key=lambda x: x["id"])
    st.write(f"총 **{len(pokes)} / 151** 마리")

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

            row[j].markdown(f"**#{p['id']} {p['name_ko']}**")
            row[j].caption(f"{p['name_en']} / 잡은 횟수: x{p['caught']}")
