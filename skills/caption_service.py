import base64
import hashlib
import io
import os
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from PIL import Image
except ImportError:
    Image = None


CAPTION_MODEL_NAME = os.environ.get("CAPTION_MODEL", "gpt-5")
CAPTION_BASE_URL = os.environ.get("OPENAI_BASE_URL")
CAPTION_TIMEOUT = float(os.environ.get("CAPTION_TIMEOUT", "120"))
CAPTION_PROMPT = os.environ.get(
    "CAPTION_PROMPT",
    "Describe the image in detail. Focus on objects, text, spatial relations, numbers, labels, and visual evidence useful for answering questions.",
)

_client_cache = {}
_caption_cache = {}


def _get_client():
    if OpenAI is None:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    cache_key = (CAPTION_BASE_URL, api_key)
    if cache_key not in _client_cache:
        kwargs = {"api_key": api_key}
        if CAPTION_BASE_URL:
            kwargs["base_url"] = CAPTION_BASE_URL
        _client_cache[cache_key] = OpenAI(**kwargs)
    return _client_cache[cache_key]


def _image_path_to_jpeg_base64(image_path):
    if not image_path:
        return ""

    path = Path(image_path)
    if not path.exists():
        return ""

    if Image is None:
        with path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _image_url_to_base64(image_url):
    if image_url and image_url.startswith("data:image") and "," in image_url:
        return image_url.split(",", 1)[1]
    return ""


def _cache_key(image_base64):
    return hashlib.sha256(image_base64.encode("utf-8")).hexdigest()


def caption_image(image_path="", image_url="", timeout=CAPTION_TIMEOUT):
    image_base64 = _image_path_to_jpeg_base64(image_path) or _image_url_to_base64(image_url)
    if not image_base64:
        return ""

    cache_key = _cache_key(image_base64)
    if cache_key in _caption_cache:
        return _caption_cache[cache_key]

    client = _get_client()
    if client is None:
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    },
                },
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model=CAPTION_MODEL_NAME,
            messages=messages,
            timeout=timeout,
        )
        text = response.choices[0].message.content if response.choices else ""
    except Exception as e:
        print(f"Caption request failed: {e}")
        text = ""

    _caption_cache[cache_key] = text
    return text


def format_caption_results_for_images(image_paths, image_urls=None, empty_default=False):
    if not image_paths and not image_urls:
        return ""

    image_paths = image_paths or []
    image_urls = image_urls or []
    image_count = max(len(image_paths), len(image_urls))
    formatted = []

    for index in range(image_count):
        image_path = image_paths[index] if index < len(image_paths) else ""
        image_url = image_urls[index] if index < len(image_urls) else ""
        fallback = "" if empty_default else "Not found"
        text = caption_image(image_path=image_path, image_url=image_url) or fallback
        formatted.append(f"The caption result of image{index + 1} is: {text}")

    return "\n\n".join(formatted)
