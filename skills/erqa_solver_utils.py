import base64
import io
import json
import os
import re
import time
from ast import literal_eval
from types import SimpleNamespace

from skills.caption_service import format_caption_results_for_images
from skills.detection_service import (
    box_images_for_paths,
    format_detection_results_for_images,
)
from skills.ocr_service import format_ocr_results_for_images

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEEPEYES_SYSTEM_PROMPT = """You are a helpful assistant.
<skills>
{"type":"function","function":{"name":"image_zoom_in","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}},"required":["bbox_2d"]}}}
</skills>
# How to call a skill
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "image_zoom_in", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}
</tool_call>"""


def init_usage_totals():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def accumulate_usage(usage_totals, outputs):
    usage = getattr(outputs, "usage", None)
    if usage is None:
        return

    usage_totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    usage_totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    usage_totals["total_tokens"] += getattr(usage, "total_tokens", 0) or 0


def merge_usage_totals(target, source):
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] += source.get(key, 0) or 0


def attach_usage_totals(outputs, usage_totals):
    if outputs is None or isinstance(outputs, str):
        return outputs

    usage = getattr(outputs, "usage", None)
    if usage is None:
        outputs.usage = SimpleNamespace(**usage_totals)
        return outputs

    usage.prompt_tokens = usage_totals["prompt_tokens"]
    usage.completion_tokens = usage_totals["completion_tokens"]
    usage.total_tokens = usage_totals["total_tokens"]
    return outputs


def extract_content(outputs):
    if hasattr(outputs, "choices") and outputs.choices:
        return outputs.choices[0].message.content
    return ""


def remove_think_content(text):
    if not isinstance(text, str) or text == "":
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def clean_output_content(outputs):
    if hasattr(outputs, "choices") and outputs.choices:
        outputs.choices[0].message.content = remove_think_content(
            outputs.choices[0].message.content
        )
    return outputs


def format_result_for_images(result_type, image_paths, dataset="", caption_empty_default=False):
    if not image_paths:
        return ""

    if result_type == "ocr":
        return format_ocr_results_for_images(image_paths)
    if result_type == "detection":
        return format_detection_results_for_images(image_paths)
    if result_type == "caption":
        return format_caption_results_for_images(
            image_paths,
            empty_default=caption_empty_default,
        )

    raise ValueError(f"Unsupported result type: {result_type}")


def get_box_images_for_paths(image_paths, dataset=""):
    return box_images_for_paths(image_paths)


def build_messages(system_prompt, images, prompt_text, per_image_extras=None, include_images=True):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if include_images and images:
        content = []
        per_image_extras = per_image_extras or [[] for _ in images]
        for index, image in enumerate(images):
            content.append({"type": "image_url", "image_url": {"url": f"{image}"}})
            for extra_image in per_image_extras[index]:
                if extra_image:
                    content.append({"type": "image_url", "image_url": {"url": f"{extra_image}"}})
        content.append({"type": "text", "text": prompt_text})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt_text})

    return messages


def build_preview_messages(question, workflow_steps):
    step_lines = [f"{idx + 1}. {step}" for idx, step in enumerate(workflow_steps)]
    preview_text = (
        f"Question:\n{question}\n\n"
        "Workflow summary:\n"
        + "\n".join(step_lines)
        + "\n"
    )
    return [{"role": "user", "content": preview_text}]


def create_deepeyes_client(base_url="http://localhost:2370/v1"):
    if OpenAI is None:
        return None

    try:
        return OpenAI(base_url=base_url, api_key="EMPTY")
    except Exception as e:
        print(f"初始化 DeepEyes-7B 客户端失败: {e}")
        return None


def get_deepeyes_decision(deepeyes_client, question, img_url, timeout, focus_hint=""):
    if deepeyes_client is None:
        return "", None

    prompt_text = (
        f"Question: {question}\n"
        "Find the single most helpful local region for answering the question. "
        "Call image_zoom_in only if a crop would materially improve accuracy. "
        "If you use the skill, crop tightly around the key evidence.\n"
    )
    if focus_hint:
        prompt_text += f"Extra focus hint: {focus_hint}\n"
    prompt_text += (
        "Format strictly as: <think>...</think> <tool_call>...</tool_call> (if needed) <answer>...</answer>"
    )

    messages = [
        {"role": "system", "content": DEEPEYES_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_url}},
            {"type": "text", "text": prompt_text},
        ]},
    ]

    try:
        response = deepeyes_client.chat.completions.create(
            model="DeepEyes-7B",
            messages=messages,
            temperature=0.0,
            stop=["<|im_end|>\n", "</answer>", "</tool_call>"],
            timeout=timeout,
        )
        return extract_content(response), response
    except Exception as e:
        print(f"DeepEyes-7B 调用失败(Erqa): {e}")
        return "", None


def crop_image_from_output(image_path, deepeyes_output, save_dir=None, crop_suffix="erqa_crop"):
    if Image is None:
        print("裁剪图像失败: Pillow is not installed")
        return None, None

    try:
        call_str = deepeyes_output.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        try:
            action = json.loads(call_str.replace("'", '"'))
        except Exception:
            action = literal_eval(call_str)

        bbox = action["arguments"]["bbox_2d"]
        with Image.open(image_path) as img:
            crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            saved_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                timestamp = int(time.time() * 1000)
                saved_path = os.path.join(save_dir, f"{image_name}_{timestamp}_{crop_suffix}.jpg")
                crop.save(saved_path, format="JPEG")

            buffer = io.BytesIO()
            crop.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}", saved_path
    except Exception as e:
        print(f"裁剪图像失败: {e}")
        return None, None


def collect_focus_crops(
    question,
    images,
    image_paths,
    timeout,
    focus_hint="",
    max_images=1,
    crop_suffix="erqa_crop",
    is_test=False,
):
    crop_urls = [""] * len(images)
    decisions = []
    usage_totals = init_usage_totals()
    saved_paths = []

    deepeyes_client = create_deepeyes_client()
    if deepeyes_client is None:
        return {
            "crop_urls": crop_urls,
            "saved_paths": saved_paths,
            "decisions": decisions,
            "usage_totals": usage_totals,
        }

    for idx, (img_url, img_path) in enumerate(zip(images[:max_images], image_paths[:max_images])):
        decision, response = get_deepeyes_decision(
            deepeyes_client=deepeyes_client,
            question=question,
            img_url=img_url,
            timeout=timeout,
            focus_hint=focus_hint,
        )
        accumulate_usage(usage_totals, response)
        decisions.append(decision)

        if "<tool_call>" not in decision or not img_path:
            continue

        crop_url, saved_path = crop_image_from_output(
            image_path=img_path,
            deepeyes_output=decision,
            crop_suffix=crop_suffix,
        )
        if crop_url:
            crop_urls[idx] = crop_url
        if saved_path:
            saved_paths.append(saved_path)

    return {
        "crop_urls": crop_urls,
        "saved_paths": saved_paths,
        "decisions": decisions,
        "usage_totals": usage_totals,
    }
