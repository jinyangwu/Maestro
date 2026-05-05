import base64
import hashlib
import io
import json
import os
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from PIL import Image
except ImportError:
    Image = None


DETECTION_BASE_URL = os.environ.get("DETECTION_BASE_URL", "http://localhost:2389/v1")
DETECTION_MODEL_NAME = os.environ.get("DETECTION_MODEL", "PR1-Qwen2.5-VL-3B-Detection")
DETECTION_API_KEY = os.environ.get("DETECTION_API_KEY", "EMPTY")
DETECTION_TIMEOUT = float(os.environ.get("DETECTION_TIMEOUT", "120"))

DETECTION_PROMPT = (
    "Locate every item from the category list in the image and output "
    "the coordinates in JSON format. The category set includes "
    "person, bicycle, car, motorcycle, airplane, bus, train, truck, "
    "boat, traffic light, fire hydrant, stop sign, parking meter, bench, "
    "bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, "
    "backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, "
    "sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, "
    "tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, "
    "banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, "
    "donut, cake, chair, couch, potted plant, bed, dining table, toilet, "
    "tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, "
    "toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, "
    "hair drier, toothbrush."
)

_client_cache = {}
_detection_cache = {}
_box_image_cache = {}


def _get_client(base_url=DETECTION_BASE_URL, api_key=DETECTION_API_KEY):
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
    if image_url and image_url.startswith("data:image") and "," in image_url:
        return image_url.split(",", 1)[1]
    return ""


def _cache_key(image_base64):
    return hashlib.sha256(image_base64.encode("utf-8")).hexdigest()


def detect_objects(image_path="", image_url="", timeout=DETECTION_TIMEOUT):
    image_base64 = _image_path_to_jpeg_base64(image_path) or _image_url_to_base64(image_url)
    if not image_base64:
        return ""

    cache_key = _cache_key(image_base64)
    if cache_key in _detection_cache:
        return _detection_cache[cache_key]

    client = _get_client()
    if client is None:
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DETECTION_PROMPT},
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
            model=DETECTION_MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
            timeout=timeout,
        )
        text = response.choices[0].message.content if response.choices else ""
    except Exception as e:
        print(f"Detection request failed: {e}")
        text = ""

    _detection_cache[cache_key] = text
    return text


def parse_detection_output(raw_output):
    if not raw_output or (isinstance(raw_output, str) and raw_output.startswith("[ERROR]")):
        return []

    cleaned = str(raw_output).replace("```json", "").replace("```", "").strip()
    try:
        bboxes = json.loads(cleaned)
    except json.JSONDecodeError:
        bboxes = json.loads(cleaned.replace("'", '"'))

    if isinstance(bboxes, dict) and "objects" in bboxes:
        bboxes = bboxes["objects"]
    if not isinstance(bboxes, list):
        return []
    return bboxes


def draw_boxes_to_data_url(image_path, detection_output):
    if not image_path or cv2 is None:
        return ""

    cache_key = (image_path, detection_output)
    if cache_key in _box_image_cache:
        return _box_image_cache[cache_key]

    img = cv2.imread(image_path)
    if img is None:
        return ""

    try:
        bboxes = parse_detection_output(detection_output)
        for obj in bboxes:
            if not isinstance(obj, dict):
                continue
            bbox = obj.get("bbox_2d")
            label = obj.get("label", "unknown")
            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            height, width = img.shape[:2]
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(
                img,
                (x1, max(0, y1 - text_size[1] - 5)),
                (min(width - 1, x1 + text_size[0]), y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    except Exception as e:
        print(f"Drawing detection boxes failed: {e}")

    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        return ""
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"
    _box_image_cache[cache_key] = data_url
    return data_url


def format_detection_results_for_images(image_paths, image_urls=None):
    if not image_paths and not image_urls:
        return ""

    image_paths = image_paths or []
    image_urls = image_urls or []
    image_count = max(len(image_paths), len(image_urls))
    formatted = []

    for index in range(image_count):
        image_path = image_paths[index] if index < len(image_paths) else ""
        image_url = image_urls[index] if index < len(image_urls) else ""
        text = detect_objects(image_path=image_path, image_url=image_url) or "Not found"
        formatted.append(f"The detection result of image{index + 1} is: {text}")

    return "\n\n".join(formatted)


def box_images_for_paths(image_paths, image_urls=None):
    image_paths = image_paths or []
    image_urls = image_urls or []
    image_count = max(len(image_paths), len(image_urls))
    box_images = []

    for index in range(image_count):
        image_path = image_paths[index] if index < len(image_paths) else ""
        image_url = image_urls[index] if index < len(image_urls) else ""
        detection_output = detect_objects(image_path=image_path, image_url=image_url)
        box_images.append(draw_boxes_to_data_url(image_path, detection_output) if image_path else "")

    return box_images
