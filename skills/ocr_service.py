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


OCR_BASE_URL = os.environ.get("GLM_OCR_BASE_URL", "http://localhost:2388/v1")
OCR_MODEL_NAME = os.environ.get("GLM_OCR_MODEL", "GLM-OCR")
OCR_API_KEY = os.environ.get("GLM_OCR_API_KEY", "EMPTY")
OCR_TIMEOUT = float(os.environ.get("GLM_OCR_TIMEOUT", "120"))

_client_cache = {}
_ocr_cache = {}


def _get_client(base_url=OCR_BASE_URL, api_key=OCR_API_KEY):
    if OpenAI is None:
        return None

    cache_key = (base_url, api_key)
    if cache_key not in _client_cache:
        _client_cache[cache_key] = OpenAI(base_url=base_url, api_key=api_key)
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
    if not image_url:
        return ""
    if image_url.startswith("data:image") and "," in image_url:
        return image_url.split(",", 1)[1]
    return ""


def _cache_key(image_base64):
    return hashlib.sha256(image_base64.encode("utf-8")).hexdigest()


def recognize_text(image_path="", image_url="", timeout=OCR_TIMEOUT):
    image_base64 = _image_path_to_jpeg_base64(image_path) or _image_url_to_base64(image_url)
    if not image_base64:
        return ""

    cache_key = _cache_key(image_base64)
    if cache_key in _ocr_cache:
        return _ocr_cache[cache_key]

    client = _get_client()
    if client is None:
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Text Recognition:"},
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
            model=OCR_MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
            timeout=timeout,
        )
        text = response.choices[0].message.content if response.choices else ""
    except Exception as e:
        print(f"GLM-OCR request failed: {e}")
        text = ""

    _ocr_cache[cache_key] = text
    return text


def format_ocr_results_for_images(image_paths, image_urls=None):
    if not image_paths and not image_urls:
        return ""

    image_paths = image_paths or []
    image_urls = image_urls or []
    image_count = max(len(image_paths), len(image_urls))
    formatted = []

    for index in range(image_count):
        image_path = image_paths[index] if index < len(image_paths) else ""
        image_url = image_urls[index] if index < len(image_urls) else ""
        text = recognize_text(image_path=image_path, image_url=image_url) or "Not found"
        formatted.append(f"The ocr result of image{index + 1} is: {text}")

    return "\n\n".join(formatted)
