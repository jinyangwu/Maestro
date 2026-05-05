import ast
import os
import re
import sys


import tempfile
import subprocess


def _get_test_script(ground_truth):
    if isinstance(ground_truth, list):
        return str(ground_truth[0]).strip() if ground_truth else ""
    if isinstance(ground_truth, dict):
        for key in ("answer", "test", "tests", "ground_truth"):
            if key in ground_truth and ground_truth[key] is not None:
                return str(ground_truth[key]).strip()
        return ""
    return str(ground_truth).strip() if ground_truth is not None else ""
def _is_valid_python(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def extract_code(solution_str: str) -> str:
    source = solution_str or ""
    think_end_idx = source.rfind("</think>")
    if think_end_idx != -1:
        source = source[think_end_idx + len("</think>"):]

    start_idx = source.lower().rfind("```python")
    if start_idx != -1:
        content_start = start_idx + len("```python")
        end_idx = source.find("```", content_start)
        if end_idx != -1:
            code = source[content_start:end_idx].strip()
            if _is_valid_python(code):
                return code

    return solution_str


def _run_test_script(code: str, test_script: str, timeout: int = 10) -> tuple[bool, str]:
    if not code:
        return False, "No executable Python code found."
    if not test_script:
        return False, "No test script found in ground truth."

    script_content = f"{code.rstrip()}\n\n{test_script.strip()}\n"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(script_content)
            temp_path = handle.name

        completed = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
        )
        error_output = (completed.stderr or completed.stdout or "").strip()
        return completed.returncode == 0, error_output
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout}s."
    except Exception as exc:
        return False, str(exc)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def compute_score(
    solution_str,
    ground_truth,
    format_score=0.0,
    score=1.0,
    cost_coe=0.0,
    api_cost=0.0,
    state="train",
    reward_metric="f1",
    return_answer=False,
    timeout=10,
):
    del score, reward_metric
    extracted_code = extract_code(solution_str)
    test_script = _get_test_script(ground_truth)
    passed, _ = _run_test_script(extracted_code, test_script, timeout=timeout)
    # import pdb; pdb.set_trace()
    metric = 1.0 if passed else 0.0

    if state == "train":
        if format_score == -1.0:
            score_info = (metric, api_cost, format_score)
        elif metric == 0:
            score_info = (metric, api_cost, metric + format_score)
        else:
            score_info = (metric, api_cost, (metric + format_score) * (1.0 - cost_coe) + api_cost * cost_coe)
    else:
        metric_em = metric
        metric_f1 = metric
        if format_score == -1.0:
            score_info = (metric_em, metric_f1, api_cost, format_score)
        elif metric == 0:
            score_info = (metric_em, metric_f1, api_cost, metric + format_score)
        else:
            score_info = (metric_em, metric_f1, api_cost, (metric + format_score) * (1.0 - cost_coe) + api_cost * cost_coe)

    if return_answer:
        return score_info, extracted_code
    return score_info


def compute_score_humaneval_v(solution_str, ground_truth, format_score=0.0, **kwargs):
    return compute_score(solution_str, ground_truth, format_score=format_score, **kwargs)


    # assert passing_score == (1.0, 0.0, 1.0)

