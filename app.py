"""AI 관상가: 얼굴 사진을 업로드하면 관상을 풀이해 주는 Streamlit 앱."""

import base64
import hashlib
import io
import json
import logging
import os
import threading
import warnings
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

from prompts import (
    DEFAULT_STYLE,
    NO_FACE_TOKEN,
    SCORE_KEYS,
    SCORES_MARKER,
    STYLES,
    USER_PROMPT,
    build_system_prompt,
)

DEFAULT_MODEL = "gpt-4o-mini"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_MB = 10
MAX_IMAGE_SIDE = 1024
MAX_IMAGE_PIXELS = 50_000_000  # 5천만 화소 (50MP 휴대폰 사진까지 허용)
MAX_ANALYSES_PER_SESSION = 5
# 서버 전체(모든 사용자 합계)의 하루 분석 상한. 새로고침으로 세션 제한을 우회해도 비용이 여기서 막힌다.
# Secrets/.env의 DAILY_ANALYSIS_LIMIT로 바꿀 수 있다.
DEFAULT_DAILY_LIMIT = 200
KST = timezone(timedelta(hours=9))
MAX_COMPLETION_TOKENS = 1500
TEMPERATURE = 0.8
API_TIMEOUT_SECONDS = 30  # PRD 성능 목표(30초)에 맞춤. 재시도는 하지 않고 실패하면 사용자가 다시 누른다
NO_FACE_SCAN_CHARS = 30  # 응답 앞부분 몇 글자 안에서 NO_FACE 토큰을 찾을지
RESULT_REQUIRED_TEXT = "첫인상"  # 정상 풀이라면 반드시 들어 있는 제목. 없으면 거부 응답으로 본다

logger = logging.getLogger(__name__)

# 로컬 개발용 .env 로드 (배포 환경에서는 파일이 없어도 무시됨)
load_dotenv()

# 정적 스타일만 담는다. 모델 응답은 절대 이 HTML 경로로 렌더링하지 않는다.
CUSTOM_CSS = """
<style>
.hero, .notice { word-break: keep-all; }
.hero { text-align: center; padding: 0.5rem 0 0.75rem; }
.hero .seal { font-size: 2.75rem; line-height: 1; }
.hero h1 { margin: 0.25rem 0 0; padding: 0; letter-spacing: 0.04em; }
.hero p { margin: 0.35rem 0 0; color: #6B5E4E; font-size: 1.05rem; }
.notice {
    border-top: 1px solid #D9CBB0; border-bottom: 1px solid #D9CBB0;
    padding: 0.5rem 0.25rem; margin: 0.25rem 0 1rem;
    text-align: center; font-size: 0.85rem; color: #6B5E4E; line-height: 1.6;
}
[data-testid="stMetricValue"] { color: #B08D57; letter-spacing: 0.08em; }
</style>
"""

HERO_HTML = """
<div class="hero">
  <div class="seal">🔮</div>
  <h1>AI 관상가</h1>
  <p>사진 한 장으로 보는 나의 관상</p>
</div>
<div class="notice">
  ⚠️ 본 서비스는 재미를 위한 것이며 과학적 근거가 없습니다.<br>
  🔒 이미지는 저장되지 않으며 분석을 위해 OpenAI API로 전송됩니다.
</div>
"""


class ImageError(Exception):
    """사용자에게 그대로 보여줄 수 있는 이미지 관련 오류."""


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def get_setting(name: str) -> str | None:
    """설정값을 st.secrets → 환경변수(.env) 순서로 읽는다."""
    try:
        # secrets.toml이 없으면 st.secrets 접근 시 예외가 날 수 있음
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def get_api_key() -> str:
    """OpenAI API 키를 읽는다. 없으면 오류를 표시하고 앱을 중단한다."""
    api_key = get_setting("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OpenAI API 키가 설정되지 않았습니다. "
            "로컬에서는 .env 파일에, Streamlit Cloud에서는 Secrets에 OPENAI_API_KEY를 등록해주세요."
        )
        st.stop()
    return api_key


def get_model() -> str:
    """사용할 OpenAI 모델명을 읽는다. 없으면 기본값을 사용한다."""
    return get_setting("OPENAI_MODEL") or DEFAULT_MODEL


def get_daily_limit() -> int:
    """서버 전체 하루 분석 상한. 설정값이 없거나 숫자가 아니면 기본값."""
    raw = get_setting("DAILY_ANALYSIS_LIMIT")
    try:
        return max(0, int(raw)) if raw not in (None, "") else DEFAULT_DAILY_LIMIT
    except (TypeError, ValueError):
        return DEFAULT_DAILY_LIMIT


# ---------------------------------------------------------------------------
# 서버 전체 하루 분석 상한 (PRD 11장 비용 리스크)
# ---------------------------------------------------------------------------
@st.cache_resource
def _daily_counter() -> dict:
    """모든 세션이 공유하는 하루 분석 횟수. 앱 프로세스가 재시작되면 0부터 다시 센다."""
    return {"date": None, "count": 0, "lock": threading.Lock()}


def reserve_daily_slot() -> bool:
    """오늘 분석 한 번을 예약한다. 상한에 도달했으면 False."""
    counter = _daily_counter()
    today = datetime.now(KST).date()
    with counter["lock"]:
        if counter["date"] != today:
            counter["date"], counter["count"] = today, 0
        if counter["count"] >= get_daily_limit():
            return False
        counter["count"] += 1
        return True


def release_daily_slot() -> None:
    """API가 결과를 주지 못한 경우 예약을 되돌린다."""
    counter = _daily_counter()
    with counter["lock"]:
        counter["count"] = max(0, counter["count"] - 1)


# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
def init_state() -> None:
    """세션 상태 기본값을 설정한다."""
    defaults = {
        "image_id": None,  # 현재 이미지의 해시. 바뀌면 이전 결과를 지운다
        "result": None,  # 마지막 분석 결과 (마크다운)
        "scores": None,  # 마지막 분석의 운세 점수 {"재물운": 4, ...} 또는 None
        "no_face": False,  # 마지막 분석에서 얼굴을 찾지 못했는지
        "refused": False,  # 마지막 분석에서 모델이 풀이를 거부했는지
        "analyses_used": 0,  # 이 세션에서 결과를 받은 API 호출 수
        "input_key": 0,  # 업로드/카메라 위젯 키. 올리면 입력이 비워진다
        "processed": None,  # (이미지 해시, 미리보기, base64) 전처리 결과 캐시
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_result() -> None:
    st.session_state.result = None
    st.session_state.scores = None
    st.session_state.no_face = False
    st.session_state.refused = False


def reset_result_if_new_image(image_id: str | None) -> None:
    """새 이미지가 들어오면 이전 결과를 초기화한다."""
    if image_id != st.session_state.image_id:
        st.session_state.image_id = image_id
        clear_result()


def start_over() -> None:
    """'다른 사진으로 다시 보기': 결과와 입력 위젯을 모두 비운다 (버튼 콜백)."""
    clear_result()
    st.session_state.image_id = None
    st.session_state.input_key += 1


def remaining_analyses() -> int:
    return max(0, MAX_ANALYSES_PER_SESSION - st.session_state.analyses_used)


# ---------------------------------------------------------------------------
# 화면: 헤더, 사이드바, 입력
# ---------------------------------------------------------------------------
def render_header() -> None:
    """제목, 부제, 면책 고지와 개인정보 고지를 표시한다 (정적 HTML)."""
    st.markdown(CUSTOM_CSS + HERO_HTML, unsafe_allow_html=True)


def render_sidebar() -> str:
    """사이드바를 그리고 선택된 분석 스타일을 반환한다."""
    with st.sidebar:
        st.header("🎴 관상 보는 법")
        style = st.radio(
            "분석 스타일",
            list(STYLES),
            index=list(STYLES).index(DEFAULT_STYLE),
            captions=["품위 있는 옛 관상가", "반말로 유쾌하게", "차분하고 논리적으로"],
        )
        st.divider()
        st.markdown(
            "**좋은 사진 고르는 법**\n"
            "- 정면을 바라본 사진\n"
            "- 밝은 조명, 그림자 없이\n"
            "- 한 사람만 나온 사진\n"
            "- 모자, 선글라스는 벗고"
        )
        st.divider()
        remaining = remaining_analyses()
        st.markdown(f"**남은 분석 횟수** · {remaining} / {MAX_ANALYSES_PER_SESSION}회")
        st.progress(remaining / MAX_ANALYSES_PER_SESSION)
    return style


def get_input_bytes() -> bytes | None:
    """업로드 탭 또는 카메라 탭에서 이미지 바이트를 받는다."""
    n = st.session_state.input_key
    upload_tab, camera_tab = st.tabs(["📁 사진 업로드", "📷 카메라 촬영"])
    with upload_tab:
        uploaded = st.file_uploader("얼굴 사진을 올려주세요", type=ALLOWED_TYPES, key=f"upload_{n}")
    with camera_tab:
        captured = st.camera_input("정면을 보고 촬영해주세요", key=f"camera_{n}")

    # 업로드 파일을 카메라 촬영보다 우선 사용
    source = uploaded or captured
    if uploaded and captured:
        st.caption("업로드한 사진과 촬영한 사진이 모두 있어, 업로드한 사진으로 분석합니다.")
    if source is None:
        return None
    if source.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"파일이 너무 큽니다. {MAX_UPLOAD_MB}MB 이하의 사진을 올려주세요.")
        return None
    return source.getvalue()


def preprocess_image(data: bytes) -> tuple[Image.Image, str]:
    """이미지를 회전 보정, RGB 변환, 리사이즈한 뒤 (미리보기 이미지, base64 JPEG)를 반환한다."""
    try:
        # open은 헤더만 읽는다. 픽셀을 메모리에 풀기 전에 크기부터 확인한다
        # (파일은 작아도 해상도가 거대한 이미지 하나로 서버 메모리가 수백 MB 늘어날 수 있음)
        # 아래에서 직접 화소 수를 검사하므로 Pillow의 대용량 경고는 로그에 남기지 않는다
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ImageError(
                f"사진 해상도가 너무 큽니다 ({width}x{height}). "
                "5천만 화소 이하의 사진을 올려주세요."
            )
        # JPEG는 축소된 크기로 바로 디코딩해 메모리를 아낀다 (다른 형식에서는 아무 일도 하지 않음)
        image.draft("RGB", (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    except Image.DecompressionBombError as e:
        # 해상도가 Pillow 안전 한도의 2배를 넘으면 open 단계에서 바로 거부된다
        raise ImageError("사진 해상도가 너무 큽니다. 5천만 화소 이하의 사진을 올려주세요.") from e
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ImageError("이미지를 열 수 없습니다. JPG, PNG, WEBP 형식의 사진을 올려주세요.") from e

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return image, base64.b64encode(buffer.getvalue()).decode("ascii")


def get_processed_image(data: bytes, image_id: str) -> tuple[Image.Image, str]:
    """같은 이미지는 한 번만 전처리하고, 화면을 다시 그릴 때는 세션에 보관한 결과를 쓴다."""
    cached = st.session_state.processed
    if cached and cached[0] == image_id:
        return cached[1], cached[2]
    preview, image_b64 = preprocess_image(data)  # 실패하면 ImageError (캐시하지 않음)
    st.session_state.processed = (image_id, preview, image_b64)
    return preview, image_b64


def render_preview(image: Image.Image) -> None:
    """미리보기를 가운데에 적당한 크기로 표시한다 (모바일에서는 컬럼이 쌓여 꽉 차게 보임)."""
    _, center, _ = st.columns([1, 1.3, 1])
    with center:
        st.image(image, caption="오늘의 얼굴", width="stretch")


# ---------------------------------------------------------------------------
# OpenAI 분석
# ---------------------------------------------------------------------------
def open_analysis_stream(api_key: str, model: str, image_b64: str, style: str):
    """관상 분석 요청을 보내고 스트리밍 응답 객체를 반환한다."""
    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT_SECONDS, max_retries=0)
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt(style)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "auto",
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )


def iter_text(stream, flag: dict) -> Iterator[str]:
    """스트리밍 응답에서 텍스트 조각만 꺼낸다. 모델의 공식 거부 신호가 오면 flag["refusal"]을 켠다."""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "refusal", None):
            flag["refusal"] = True
        if delta.content:
            yield delta.content


def filter_no_face(chunks: Iterator[str], flag: dict) -> Iterator[str]:
    """응답 앞부분에 NO_FACE 토큰이 있으면 화면에 아무것도 내보내지 않고 flag["no_face"]를 켠다.

    모델이 토큰을 굵게(**NO_FACE**)나 코드 블록으로 감싸는 경우도 잡기 위해,
    앞 NO_FACE_SCAN_CHARS 글자를 모아 그 안에 토큰이 있는지 본 뒤 흘려보낸다.
    """
    head = ""
    for piece in chunks:
        if head is None:
            yield piece
            continue
        head += piece
        if len(head) < NO_FACE_SCAN_CHARS:
            continue
        if NO_FACE_TOKEN in head[:NO_FACE_SCAN_CHARS]:
            flag["no_face"] = True
            for _ in chunks:  # 나머지 응답은 버림
                pass
            return
        yield head
        head = None
    # 응답이 검사 길이보다 짧게 끝난 경우
    if head:
        if NO_FACE_TOKEN in head:
            flag["no_face"] = True
        else:
            yield head


def split_scores(chunks: Iterator[str], flag: dict) -> Iterator[str]:
    """SCORES_MARKER 앞의 본문만 흘려보내고, 뒤의 점수 JSON 원문은 flag["scores_raw"]에 담는다.

    마커가 조각나서 올 수 있으므로, 마커의 앞부분일 수 있는 꼬리는 다음 조각이 올 때까지 붙잡아 둔다.
    """
    buffer = ""
    for piece in chunks:
        buffer += piece
        idx = buffer.find(SCORES_MARKER)
        if idx != -1:
            if idx:
                yield buffer[:idx]
            flag["scores_raw"] = buffer[idx + len(SCORES_MARKER):] + "".join(chunks)
            return
        # 버퍼 끝이 마커의 앞부분과 겹치는 만큼만 남기고 내보낸다
        keep = 0
        for k in range(min(len(SCORES_MARKER) - 1, len(buffer)), 0, -1):
            if SCORES_MARKER.startswith(buffer[-k:]):
                keep = k
                break
        if len(buffer) > keep:
            yield buffer[: len(buffer) - keep]
            buffer = buffer[len(buffer) - keep:]
    if buffer:
        yield buffer


def parse_scores(raw: str | None) -> dict[str, int] | None:
    """점수 JSON 원문을 검증해서 {운세: 1~5}로 반환한다. 형식이 어긋나면 None."""
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    scores = {}
    for key in SCORE_KEYS:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        scores[key] = min(5, max(1, round(value)))
    return scores


def run_analysis(image_b64: str, style: str) -> None:
    """분석을 실행하고 결과를 세션 상태에 저장한다. 오류는 화면에 표시하고 삼킨다."""
    api_key = get_api_key()
    model = get_model()
    flag = {"no_face": False, "refusal": False, "scores_raw": None}

    if not reserve_daily_slot():
        st.warning("오늘 준비된 관상 풀이가 모두 소진되었습니다. 내일 다시 찾아주세요. 🙏")
        return

    try:
        with st.container(border=True):
            with st.spinner("관상을 살펴보는 중입니다..."):
                stream = open_analysis_stream(api_key, model, image_b64, style)
            text = st.write_stream(split_scores(filter_no_face(iter_text(stream, flag), flag), flag))
    except Exception as e:
        # 결과를 받지 못했으므로 하루 상한 예약을 되돌린다 (세션 횟수도 차감하지 않음)
        release_daily_slot()
        show_api_error(e)
        return

    # 결과를 받은 경우에만 횟수를 차감한다 (얼굴 미검출, 거부도 응답을 받았으므로 차감)
    if flag["no_face"]:
        clear_result()
        st.session_state.no_face = True
        st.session_state.analyses_used += 1
        return

    text = text if isinstance(text, str) else "".join(map(str, text))
    if not text.strip() and not flag["refusal"]:
        release_daily_slot()
        st.error("분석 결과를 받지 못했습니다. 다시 시도해주세요.")
        return

    # 공식 거부 신호가 왔거나, 정상 풀이의 필수 제목도 점수도 없으면 거부 응답으로 본다 (PRD 11장)
    if flag["refusal"] or (RESULT_REQUIRED_TEXT not in text and flag["scores_raw"] is None):
        logger.info("모델이 관상 풀이를 거부함: %r", text[:120])
        clear_result()
        st.session_state.refused = True
        st.session_state.analyses_used += 1
        return

    st.session_state.result = text.rstrip()
    st.session_state.scores = parse_scores(flag["scores_raw"])
    st.session_state.no_face = False
    st.session_state.analyses_used += 1


def show_api_error(error: Exception) -> None:
    """API 호출 중 난 오류를 사용자용 메시지로 바꿔 보여준다. 키나 상세 내용은 노출하지 않는다."""
    if isinstance(error, openai.AuthenticationError):
        st.error("API 키가 올바르지 않습니다. 설정을 확인해주세요.")
    elif isinstance(error, openai.RateLimitError):
        st.error("요청이 많습니다. 잠시 후 다시 시도해주세요.")
    elif isinstance(error, openai.APITimeoutError):
        st.error("응답이 너무 늦어지고 있습니다. 잠시 후 다시 시도해주세요.")
    elif isinstance(error, openai.APIConnectionError):
        st.error("네트워크 연결을 확인해주세요.")
    elif isinstance(error, openai.APIError):
        logger.exception("OpenAI API 오류", exc_info=error)
        st.error("관상 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
    else:
        logger.exception("예상하지 못한 오류", exc_info=error)
        st.error("알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")


# ---------------------------------------------------------------------------
# 결과 표시
# ---------------------------------------------------------------------------
def stars(score: int) -> str:
    return "★" * score + "☆" * (5 - score)


def render_scores(scores: dict[str, int]) -> None:
    """운세 점수를 별점으로 표시한다 (PRD N-03).

    고정 폭 metric을 줄바꿈되는 가로 컨테이너에 넣어, 데스크톱은 4개 한 줄, 모바일은 2x2가 된다.
    (st.columns는 좁은 화면에서 1열로 쌓여 세로로 너무 길어진다)
    """
    icons = {"재물운": "💰", "애정운": "💞", "직업운": "💼", "건강운": "🌿"}
    with st.container(horizontal=True, wrap=True, horizontal_alignment="distribute"):
        for key in SCORE_KEYS:
            # 별 기호는 화면 낭독기에서 점수로 읽히지 않으므로 숫자를 라벨에 함께 둔다
            st.metric(f"{icons[key]} {key} {scores[key]}/5", stars(scores[key]), width=135)


def result_as_text(result: str, scores: dict[str, int] | None) -> str:
    """다운로드용 텍스트."""
    text = result
    if scores:
        score_lines = "\n".join(f"- {k}: {stars(v)} ({v}/5)" for k, v in scores.items())
        text += f"\n\n### 운세 점수\n{score_lines}"
    return text


def render_result() -> None:
    """저장된 결과를 카드로 표시한다."""
    if st.session_state.no_face:
        st.warning(
            "사진에서 얼굴을 찾을 수 없습니다. "
            "정면을 바라보는 밝은 사진으로 다시 올려주세요."
        )
    elif st.session_state.refused:
        st.warning(
            "이 사진으로는 관상 풀이를 받지 못했습니다. "
            "얼굴이 잘 보이는 다른 사진으로 다시 시도해주세요."
        )
    elif st.session_state.result:
        with st.container(border=True):
            if st.session_state.scores:
                render_scores(st.session_state.scores)
                st.divider()
            st.markdown(st.session_state.result)
    else:
        return

    # 결과가 있으면 저장/다시 보기를 나란히, 얼굴 미검출이면 다시 보기만 꽉 차게
    if st.session_state.result:
        left, right = st.columns(2)
        left.download_button(
            "📥 결과 저장하기",
            data=result_as_text(st.session_state.result, st.session_state.scores),
            file_name="관상_결과.txt",
            mime="text/plain",
            width="stretch",
        )
    else:
        right = st.container()
    right.button("🔄 다른 사진으로 다시 보기", on_click=start_over, width="stretch")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")
    init_state()
    render_header()

    # 키가 없으면 여기서 앱이 중단됨
    get_api_key()

    style = render_sidebar()

    data = get_input_bytes()
    reset_result_if_new_image(hashlib.sha256(data).hexdigest() if data else None)

    image_b64 = None
    if data is None:
        st.session_state.processed = None  # 사진을 내리면 보관한 전처리 결과도 비운다
    else:
        try:
            preview, image_b64 = get_processed_image(data, st.session_state.image_id)
            render_preview(preview)
        except ImageError as e:
            st.error(str(e))

    # 모바일에서는 사이드바가 접혀 있어 스타일 선택이 안 보이므로 본문에 현재 스타일을 알려준다
    st.caption(f"🎴 분석 스타일: **{style}** · 왼쪽 ≫ 메뉴에서 변경")

    remaining = remaining_analyses()
    clicked = st.button(
        "🔍 관상 보기",
        type="primary",
        disabled=image_b64 is None or remaining == 0,
        width="stretch",
    )
    if remaining == 0:
        st.info(f"이번 세션의 분석 횟수({MAX_ANALYSES_PER_SESSION}회)를 모두 사용했습니다.")
    elif image_b64 is None:
        st.caption("사진을 올리면 관상 보기 버튼이 활성화됩니다.")

    if clicked and image_b64 and remaining > 0:
        used_before = st.session_state.analyses_used
        run_analysis(image_b64, style)
        # API 호출이 성공했으면 다시 그려서 버튼 상태, 남은 횟수, 결과를 세션 상태 기준으로 맞춘다.
        # 실패했으면 오류 메시지가 사라지지 않도록 다시 그리지 않는다.
        if st.session_state.analyses_used != used_before:
            st.rerun()
        return

    render_result()


if __name__ == "__main__":
    main()
