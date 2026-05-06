import time
import os
import sys
import base64
from io import BytesIO

import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

try:
    from skills.factory import SkillFactory
    from skills.load_skills import load_skills
    SKILL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import skill modules: {e}")
    import traceback
    traceback.print_exc()
    SKILL_IMPORTS_AVAILABLE = False
    SkillFactory = None
    load_skills = None

# Try to import config loader
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    from skills.config_loader import load_config
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    # Fallback: create a simple load_config function
    CONFIG_LOADER_AVAILABLE = False
    def load_config(path):
        # Return empty config if loader not available
        return {"skills": []}

# Cache clients by port for different models
_cached_clients = {}



def image_path_to_base64(image_path):
    """
    将图片文件路径转换为base64格式的data URL

    Args:
        image_path (str): 图片文件路径

    Returns:
        str: base64格式的data URL，格式为 "data:image/jpeg;base64,{base64_string}"
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None

        # 读取图片并转换为base64
        with Image.open(image_path) as img:
            # 转换为RGB模式（如果不是的话）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 保存到内存缓冲区
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()

            # 转换为base64
            base64_string = base64.b64encode(image_bytes).decode('utf-8')

            # 返回data URL格式
            return f"data:image/jpeg;base64,{base64_string}"

    except Exception as e:
        print(f"Error converting image {image_path} to base64: {e}")
        return None


def get_client(
    base_url="",
    api_key="",
    max_retries=2,
    timeout=60,
    port=None
):
    """
    Get or create a cached client for a specific port.
    If port is provided, construct base_url from it.
    """
    # If port is provided, construct base_url
    if port is not None:
        base_url = f"http://127.0.0.1:{port}/v1"
    
    # Use base_url as cache key
    cache_key = base_url
    if cache_key not in _cached_clients:
        _cached_clients[cache_key] = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout
        )
    return _cached_clients[cache_key]


def get_llm_response_via_api(prompt,
                             LLM_MODEL="",
                             base_url="",
                             api_key="",
                             TAU=1.0,
                             TOP_P=1.0,
                             SEED=42,
                             MAX_TRIALS=500,
                             TIME_GAP=5,
                             port=None):
    '''
    res = get_llm_response_via_api(prompt='hello')  # Default: TAU Sampling (TAU=1.0)
    res = get_llm_response_via_api(prompt='hello', TAU=0)  # Greedy Decoding
    res = get_llm_response_via_api(prompt='hello', TAU=0.5, N=2, SEED=None)  # Return Multiple Responses w/ TAU Sampling
    '''
    # If port is provided, use it; otherwise use base_url
    if port is not None:
        client = get_client(api_key=api_key, port=port)
    else:
        client = get_client(base_url=base_url, api_key=api_key)
    completion = None
    if "deepseek" in LLM_MODEL:
        max_tokens = 14000
    elif "chart" in LLM_MODEL.lower:
        max_tokens = 1024
    else:
        max_tokens = 2048
    while MAX_TRIALS:
        MAX_TRIALS -= 1
        try:
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=TAU,
                top_p=TOP_P,
                seed=SEED,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(e)
            if "request timed out" in str(e).strip().lower():
                break
            print("Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception("Reach MAX_TRIALS={}".format(MAX_TRIALS))
    contents = completion.choices
    meta_info = completion.usage
    completion_tokens = meta_info.completion_tokens
    # prompt_tokens = meta_info.prompt_tokens
    # total_tokens = meta_info.total_tokens
    # print(completion_tokens, prompt_tokens, total_tokens)
    if len(contents) == 1:
        return contents[0].message.content, completion_tokens
    else:
        return [c.message.content for c in contents], completion_tokens


API_PRICE_1M_TOKENS = {
    "qwen/qwen2.5-7b-instruct": 0.3,
    "meta/llama-3.1-70b-instruct": 0.88,
    "meta/llama-3.1-8b-instruct": 0.18,
    "mistralai/mistral-7b-instruct-v0.3": 0.2,
    "mistralai/mixtral-8x22b-instruct-v0.1": 1.2,
    "google/gemma-2-27b-it": 0.8,
    "writer/palmyra-creative-122b": 1.8,
    "nvidia/llama3-chatqa-1.5-8b": 0.18,
}


AGENT_PROMPT = """
You are a helpful assistant. \
You are participating in a multi-agent reasoning process, where a base model delegates sub-questions to specialized models like you. \
\nYour task is to do your **absolute best** to either: \n
    + Answer the question directly, if possible, and provide a brief explanation; or \n
    + Offer helpful and relevant context, background knowledge, or insights related to the question, even if you cannot fully answer it. \

If you are completely unable to answer the question or provide any relevant or helpful information, you must: \n
    + Clearly state that you are unable to assist with this question, and \n
    + Explicitly instruct the base model to consult other LLMs for further assistance. \

**Important Constraints**: \n
    + Keep your response clear, concise, and informative (preferably under 512 tokens). Your response will help guide the base model’s reasoning and next steps. \n
    + Stay strictly on-topic. Do not include irrelevant or generic content. \

\n\nHere is the sub-question for you to assist with: {query}\n
"""

AGENT_PROMPT = '''
Answer the question:
{query}
Put the final answer int \\boxed{...} or Answer: ...
Let's think step by step.
'''


def request_task(data):
    # Accept 7 or 8 arguments (8th is skill_config_path, ignored for old format)
    if len(data) == 8:
        q_id, query_text, TAU, LLM_NAME, api_base, api_key, _, port = data
    else:
        q_id, query_text, TAU, LLM_NAME, api_base, api_key, port = data
    if LLM_NAME == "":
        # print("LLM Name Error")
        return q_id, "LLM Name Error", 0.0
    # print(f"LLM: {LLM_NAME}, Port: {port}")
    try:
        # print(f"[DEBUG] query_text: {query_text}")
        input_prompt = AGENT_PROMPT.format_map({"query": query_text})
        single_response, completion_tokens = get_llm_response_via_api(prompt=input_prompt,
                                                                      base_url=api_base,
                                                                      api_key=api_key,
                                                                      TAU=TAU,
                                                                      LLM_MODEL=LLM_NAME,
                                                                      port=port)
        # print(single_response, completion_tokens)
        # print(f"[DEBUG] single_response_skill: {single_response}")
    except Exception as e:
        print(e)
        single_response = "API Request Error"
        completion_tokens = 0.0

    # Calculate cost - use default pricing if model not in dict
    cost_per_token = API_PRICE_1M_TOKENS.get(LLM_NAME, 0.3) / 1_000_000
    return q_id, single_response, int(completion_tokens) * cost_per_token


def check_llm_name(target_llm):
    """
    Check and normalize model name for skill calls.
    Returns: (LLM_NAME, TAU, PORT)
    - LLM_NAME: Model name for API calls
    - TAU: Temperature parameter
    - PORT: Local vLLM deployment port

    Available models:
    - Chart-R1 (port 2364): Best for chart-related tasks
    - GLM-4.6V-Flash (port 2376): Best for mathematical problems and logical reasoning
    - qwen3-VL-8B-Instruct (port 2362): Best for counting tasks
    - Qwen3.5-9B (port 2381): General-purpose text model
    - Intern-S1-mini (port 2368): Best for scientific questions (biology, geography, physics, chemistry)
    - medgemma-4b-it (port 2369): Best for medical problems and CT image analysis
    """
    TAU = 0
    LLM_NAME = ""
    PORT = None
    target_llm_lower = target_llm.strip().lower()

    # Chart-R1: Best for chart-related tasks
    if "chart-r1" in target_llm_lower:
        LLM_NAME = "Chart-R1"
        PORT = 2364

    # GLM-4.1V-9B-Thinking: Best for mathematical problems and logical reasoning

    elif "glm-4.6v" in target_llm_lower:
        LLM_NAME = "GLM-4.6V-Flash"
        PORT = 2376

    # qwen3-vl-8b-instruct: Best for counting tasks
    elif "qwen3-vl" in target_llm_lower:
        LLM_NAME = "qwen3-VL-8B-Instruct"
        PORT = 2362

    # Qwen3.5-9B: General-purpose text model
    elif "qwen3.5-9b" in target_llm_lower or "qwen35-9b" in target_llm_lower:
        LLM_NAME = "Qwen3.5-9B"
        PORT = 2381

    # Intern-S1-mini: Best for scientific questions
    elif "intern-s1-mini" in target_llm_lower:
        LLM_NAME = "Intern-S1-mini"
        PORT = 2368

    # medgemma-4b-it: Best for medical problems
    elif "medgemma-1.5-4b" in target_llm_lower:
        LLM_NAME = "medgemma-1.5-4b-it"
        PORT = 2369
    elif "step3" in target_llm_lower:
        LLM_NAME = "Step3-VL-10B"
        PORT = 2382
    else:
        LLM_NAME = ""
        PORT = None

    return LLM_NAME, TAU, PORT


def check_skill_name(target_skill):
    """Check and normalize skill name with flexible matching."""
    target_skill_lower = target_skill.strip().lower()

    # Exact mapping for precise matches
    skill_mapping = {
        "geometric_problem_solver": "Geometric_Problem_Solver",
        "geometric-problem-solver": "Geometric_Problem_Solver",
        "geometricproblemsolver": "Geometric_Problem_Solver",
        "chart_problem_solver": "Chart_Problem_Solver",
        "chart-problem-solver": "Chart_Problem_Solver",
        "chartproblemsolver": "Chart_Problem_Solver",
        "science_problem_solver": "Science_Problem_Solver",
        "science-problem-solver": "Science_Problem_Solver",
        "scienceproblemsolver": "Science_Problem_Solver",
        "counting_problem_solver": "Counting_Problem_Solver",
        "counting-problem-solver": "Counting_Problem_Solver",
        "countingproblemsolver": "Counting_Problem_Solver",
        "perception_problem_solver": "Perception_Problem_Solver",
        "perception-problem-solver": "Perception_Problem_Solver",
        "perceptionproblemsolver": "Perception_Problem_Solver",
        "perception_problem_solver_eval": "Perception_Problem_Solver_Eval",
        "perception-problem-solver-eval": "Perception_Problem_Solver_Eval",
        "perceptionproblemsolvereval": "Perception_Problem_Solver_Eval",
        "code_problem_solver": "Code_Problem_Solver",
        "code-problem-solver": "Code_Problem_Solver",
        "codeproblemsolver": "Code_Problem_Solver",
        "erqa_trajectory_outcome_solver": "ERQA_Trajectory_Outcome_Solver",
        "erqa-trajectory-outcome-solver": "ERQA_Trajectory_Outcome_Solver",
        "erqatrajectoryoutcomesolver": "ERQA_Trajectory_Outcome_Solver",
        "erqa_action_adjustment_solver": "ERQA_Action_Adjustment_Solver",
        "erqa-action-adjustment-solver": "ERQA_Action_Adjustment_Solver",
        "erqaactionadjustmentsolver": "ERQA_Action_Adjustment_Solver",
        "erqa_spatial_mechanics_solver": "ERQA_Spatial_Mechanics_Solver",
        "erqa-spatial-mechanics-solver": "ERQA_Spatial_Mechanics_Solver",
        "erqaspatialmechanicssolver": "ERQA_Spatial_Mechanics_Solver",
        "erqa_pointing_localization_solver": "ERQA_Pointing_Localization_Solver",
        "erqa-pointing-localization-solver": "ERQA_Pointing_Localization_Solver",
        "erqapointinglocalizationsolver": "ERQA_Pointing_Localization_Solver",
        "erqa_multi_view_task_solver": "ERQA_Multi_View_Task_Solver",
        "erqa-multi-view-task-solver": "ERQA_Multi_View_Task_Solver",
        "erqamultiviewtasksolver": "ERQA_Multi_View_Task_Solver",
        "embodied_scene_qa_skill": "Embodied_Scene_QA_Skill",
        "embodied-scene-qa-skill": "Embodied_Scene_QA_Skill",
        "embodiedsceneqaskill": "Embodied_Scene_QA_Skill",
        "erqa_embodied_scene_qa_skill": "ERQA_Embodied_Scene_QA_Skill",
        "erqa-embodied-scene-qa-skill": "ERQA_Embodied_Scene_QA_Skill",
        "erqaembodiedsceneqaskill": "ERQA_Embodied_Scene_QA_Skill",
        "compositional_visual_reasoning_skill": "Compositional_Visual_Reasoning_Skill",
        "compositional-visual-reasoning-skill": "Compositional_Visual_Reasoning_Skill",
        "compositionalvisualreasoningskill": "Compositional_Visual_Reasoning_Skill",
        "clevr_problem_solver": "clevr_problem_solver",
        "clevr-problem-solver": "clevr_problem_solver",
        "clevrproblemsolver": "clevr_problem_solver",
        "clevr_master_router_skill": "CLEVR_Master_Router_Skill",
        "clevr-master-router-skill": "CLEVR_Master_Router_Skill",
        "clevrmasterrouterskill": "CLEVR_Master_Router_Skill",
        "ocrbench_text_recognition_skill": "OCRBench_Text_Recognition_Skill",
        "ocrbench-text-recognition-skill": "OCRBench_Text_Recognition_Skill",
        "ocrbenchtextrecognitionskill": "OCRBench_Text_Recognition_Skill",
        "ocrbench_key_information_extraction_skill": "OCRBench_Key_Information_Extraction_Skill",
        "ocrbench-key-information-extraction-skill": "OCRBench_Key_Information_Extraction_Skill",
        "ocrbenchkeyinformationextractionskill": "OCRBench_Key_Information_Extraction_Skill",
        "ocrbench_scene_text_qa_skill": "OCRBench_Scene_Text_QA_Skill",
        "ocrbench-scene-text-qa-skill": "OCRBench_Scene_Text_QA_Skill",
        "ocrbenchscenetextqaskill": "OCRBench_Scene_Text_QA_Skill",
        "ocrbench_document_chart_qa_skill": "OCRBench_Document_Chart_QA_Skill",
        "ocrbench-document-chart-qa-skill": "OCRBench_Document_Chart_QA_Skill",
        "ocrbenchdocumentchartqaskill": "OCRBench_Document_Chart_QA_Skill",
        "ocrbench_formula_recognition_skill": "OCRBench_Formula_Recognition_Skill",
        "ocrbench-formula-recognition-skill": "OCRBench_Formula_Recognition_Skill",
        "ocrbenchformularecognitionskill": "OCRBench_Formula_Recognition_Skill",
        "ocr_problem_solver": "ocr_problem_solver",
        "ocr-problem-solver": "ocr_problem_solver",
        "ocrproblemsolver": "ocr_problem_solver",
        "ocrbench_master_router_skill": "OCRBench_Master_Router_Skill",
        "ocrbench-master-router-skill": "OCRBench_Master_Router_Skill",
        "ocrbenchmasterrouterskill": "OCRBench_Master_Router_Skill",
        "diagram_reasoning_skill": "Diagram_Reasoning_Skill",
        "diagram-reasoning-skill": "Diagram_Reasoning_Skill",
        "diagramreasoningskill": "Diagram_Reasoning_Skill",
        "diagram_reasoning_router_skill": "Diagram_Reasoning_Skill",
        "diagram-reasoning-router-skill": "Diagram_Reasoning_Skill",
        "diagramreasoningrouterskill": "Diagram_Reasoning_Skill",
        "vlmsareblind_master_router_skill": "Diagram_Reasoning_Skill",
        "vlmsareblind-master-router-skill": "Diagram_Reasoning_Skill",
        "vlmsareblindmasterrouterskill": "Diagram_Reasoning_Skill",
    }

    # Check for exact match first
    if target_skill_lower in skill_mapping:
        return skill_mapping[target_skill_lower]


    if "geometric" in target_skill_lower and "solver" in target_skill_lower:
        return "Geometric_Problem_Solver"
    elif "chart" in target_skill_lower and "solver" in target_skill_lower:
        return "Chart_Problem_Solver"
    elif "science" in target_skill_lower and "solver" in target_skill_lower:
        return "Science_Problem_Solver"
    elif "count" in target_skill_lower and "solver" in target_skill_lower:
        return "Counting_Problem_Solver"
    elif "perception" in target_skill_lower and "solver" in target_skill_lower:
        if "eval" in target_skill_lower:
            return "Perception_Problem_Solver_Eval"
        return "Perception_Problem_Solver"
    elif "code" in target_skill_lower or "python" in target_skill_lower or "generator" in target_skill_lower:
        return "Code_Problem_Solver"
    elif "trajectory" in target_skill_lower:
        return "ERQA_Trajectory_Outcome_Solver"
    elif "action" in target_skill_lower:
        return "ERQA_Action_Adjustment_Solver"
    elif "spatial" in target_skill_lower:
        return "ERQA_Spatial_Mechanics_Solver"
    elif "point" in target_skill_lower:
        return "ERQA_Pointing_Localization_Solver"
    elif "embodied" in target_skill_lower and "skill" in target_skill_lower:
        return "Embodied_Scene_QA_Skill"
    elif "compositional" in target_skill_lower and "reasoning" in target_skill_lower and "skill" in target_skill_lower:
        return "Compositional_Visual_Reasoning_Skill"
    elif "clevr" in target_skill_lower and "problem" in target_skill_lower and "solver" in target_skill_lower:
        return "clevr_problem_solver"
    elif "clevr" in target_skill_lower and "router" in target_skill_lower:
        return "CLEVR_Master_Router_Skill"
    elif "ocr" in target_skill_lower and "problem" in target_skill_lower and "solver" in target_skill_lower:
        return "ocr_problem_solver"
    elif "ocrbench" in target_skill_lower and "router" in target_skill_lower:
        return "OCRBench_Master_Router_Skill"
    elif "diagram" in target_skill_lower and "reason" in target_skill_lower and "skill" in target_skill_lower:
        return "Diagram_Reasoning_Skill"
    elif "vlmsareblind" in target_skill_lower and "router" in target_skill_lower:
        return "Diagram_Reasoning_Skill"
    elif "ocrbench" in target_skill_lower and "formula" in target_skill_lower:
        return "OCRBench_Formula_Recognition_Skill"
    elif "ocrbench" in target_skill_lower and ("key" in target_skill_lower or "information" in target_skill_lower):
        return "OCRBench_Key_Information_Extraction_Skill"
    elif "ocrbench" in target_skill_lower and "scene" in target_skill_lower:
        return "OCRBench_Scene_Text_QA_Skill"
    elif "ocrbench" in target_skill_lower and ("document" in target_skill_lower or "chart" in target_skill_lower):
        return "OCRBench_Document_Chart_QA_Skill"
    elif "ocrbench" in target_skill_lower and "text" in target_skill_lower:
        return "OCRBench_Text_Recognition_Skill"
    elif "ocr" in target_skill_lower:
        return "ocr"
    elif ("multi" in target_skill_lower or "task" in target_skill_lower):
        return "ERQA_Multi_View_Task_Solver"

    # No match found
    return ""


def build_minimal_skill(skill_name, model_name, api_base, api_key):
    """Create a minimal skill config when the skill is missing from YAML config."""
    from skills.load_skills import Skill

    skill_config = {
        'name': skill_name,
        'model_name': model_name,
        'base_url': api_base,
        'api_key': api_key,
    }
    if skill_name == "calculator":
        skill_config.update({'number': 3, 'retry': 3})
    elif skill_name == "python-code":
        skill_config.update({'number': 12, 'retry': 12})
    elif skill_name == "web-search":
        skill_config.update({'k': 3})
    elif skill_name == "prm":
        skill_config.update({'number': 6, 'retry': 5})
    elif skill_name in {
        "none",
        "Geometric_Problem_Solver",
        "Chart_Problem_Solver",
        "Science_Problem_Solver",
        "Counting_Problem_Solver",
        "Perception_Problem_Solver",
        "Perception_Problem_Solver_Eval",
        "Code_Problem_Solver",
        "ERQA_Trajectory_Outcome_Solver",
        "ERQA_Action_Adjustment_Solver",
        "ERQA_Spatial_Mechanics_Solver",
        "ERQA_Pointing_Localization_Solver",
        "ERQA_Multi_View_Task_Solver",
        "Embodied_Scene_QA_Skill",
        "ERQA_Embodied_Scene_QA_Skill",
        "Compositional_Visual_Reasoning_Skill",
        "clevr_problem_solver",
        "CLEVR_Master_Router_Skill",
        "OCRBench_Text_Recognition_Skill",
        "OCRBench_Key_Information_Extraction_Skill",
        "OCRBench_Scene_Text_QA_Skill",
        "OCRBench_Document_Chart_QA_Skill",
        "OCRBench_Formula_Recognition_Skill",
        "ocr_problem_solver",
        "OCRBench_Master_Router_Skill",
        "Diagram_Reasoning_Skill",
    }:
        skill_config.update({'number': 3, 'retry': 3})

    return Skill(skill_name=skill_name, skill_config=skill_config)


def request_skill_task(data, dataset=""):
    """Execute a skill call task."""
    # Handle both old format (8 args) and new format (9 args with extra_fields)
    if len(data) == 9:
        q_id, model_name, skill_name, skill_input, api_base, api_key, skill_config_path, port, extra_fields = data
    else:
        q_id, model_name, skill_name, skill_input, api_base, api_key, skill_config_path, port = data
        extra_fields = None
    
    if not SKILL_IMPORTS_AVAILABLE:
        return q_id, "Skill imports not available", 0.0
    
    try:
        # Load skill configuration
        skills = []
        if skill_config_path and os.path.exists(skill_config_path):
            try:
                config = load_config(skill_config_path)
                skills = load_skills(config)
            except Exception as e:
                print(f"Warning: Could not load skill config from {skill_config_path}: {e}")
        
        # If no skills loaded, create a minimal skill config based on skill_name
        if not skills:
            skills = [build_minimal_skill(skill_name, model_name, api_base, api_key)]
        
        # Find the requested skill
        skill = None
        for t in skills:
            if t.skill_name == skill_name:
                skill = t
                break
        
        if skill is None:
            skill = build_minimal_skill(skill_name, model_name, api_base, api_key)
        
        # Create client using port if available, otherwise use api_base
        api_key = 'EMPTY'
        if port is not None:
            client = get_client(api_key=api_key, port=port)
            skill_client = get_client(api_key=api_key, port=port)
        else:
            client = get_client(base_url=api_base, api_key=api_key)
            skill_client = get_client(base_url=api_base, api_key=api_key)
        
        # Create skill factory and get skill instance
        timeout = 900

        # Extract images from extra_fields by converting image_paths to base64
        image_paths = []
        images = []
        have_image = False
        if extra_fields:
            if "image_paths" in extra_fields:
                image_paths = extra_fields["image_paths"] or []
                if image_paths:
                    # 将image_paths转换为base64格式
                    for img_path in image_paths:
                        if isinstance(img_path, str) and img_path.strip():
                            base64_data = image_path_to_base64(img_path.strip())
                            if base64_data:
                                images.append(base64_data)
                    have_image = len(images) > 0

        skill_factory = SkillFactory(
            client=client,
            skill_client=skill_client,
            skill_model_name=model_name,
            model=model_name,
            system_prompt="",
            question=skill_input,
            images=images,
            have_image=have_image,
            temperature=0.2,
            top_p=1.0,
            timeout=timeout,
            image_paths=image_paths,
            dataset=dataset
        )
        
        skill_instance = skill_factory.get_skill(skill)
        
        # Execute skill
        # import pdb; pdb.set_trace()
        retry = 2
        while retry > 0:
            outputs = skill_instance.generate()

            if hasattr(outputs, 'choices') and len(outputs.choices) > 0:
                break
            else:
                print(f"Skill execution error({skill.skill_name}). Retry: {retry}")
                retry -= 1  
                time.sleep(6)
        if retry == 0:
            return q_id, "Skill execution error", 0.0
        
        # Extract response and tokens
        if hasattr(outputs, 'choices') and len(outputs.choices) > 0:
            response_original = outputs.choices[0].message.content
            # print(f"[DEBUG] response_skill: {response_original}")
            if len(response_original) > 1000:
                response = response_original[-1000:]
                # print("[DEBUG] original response length: ", len(response_original))
                # print("[DEBUG] response_skill length: ", len(response))
                # print(f"[DEBUG] response_skill_short: {response}")
            else:
                response = response_original
            completion_tokens = outputs.usage.completion_tokens if hasattr(outputs, 'usage') and outputs.usage else 0
        else:
            response = str(outputs) if outputs else "Skill execution failed"
            completion_tokens = 0
        # print(f"[DEBUG] response: {response}, completion_tokens: {completion_tokens}")
        # Calculate cost (simplified, using default pricing)
        cost = completion_tokens * 0.0003  # Default cost per token
        
        return q_id, response, completion_tokens
        
    except Exception as e:
        print(f"Skill execution error: {e}")
        import traceback
        traceback.print_exc()
        return q_id, f"Skill execution error: {str(e)}", 0.0


def access_routing_pool(queries, api_base, api_key, skill_config_path=None):
    """
    Access routing pool with skill support.
    Format: Model-Name@@Skill-Name:Your-Input
    """
    task_args = []
    deepseek_count = 0
    for q_id, single_query in enumerate(queries):
        # Parse format: Model-Name@@Skill-Name:Your-Input
        if "@@" in single_query and ":" in single_query:
            if "deepseek" in single_query:
                deepseek_count += 1
            import random
            if random.randint(1, 24) == 1:
                print(f"[DEBUG] single_query: {single_query}")
            parts = single_query.split(":", 1)
            if len(parts) == 2:
                model_skill_part = parts[0].strip()
                skill_input = parts[1].strip()
                
                if "@@" in model_skill_part:
                    model_name_part, skill_name_part = model_skill_part.split("@@", 1)
                    model_name = model_name_part.strip()
                    skill_name = skill_name_part.strip()
                    
                    # Normalize model and skill names
                    LLM_NAME, TAU, PORT = check_llm_name(target_llm=model_name)
                    normalized_skill_name = check_skill_name(target_skill=skill_name)
                    
                    if not LLM_NAME:
                        task_args.append((q_id, "Model Name Error", "", "", api_base, api_key, skill_config_path, None))
                    elif not normalized_skill_name:
                        task_args.append((q_id, LLM_NAME, "Skill Name Error", "", api_base, api_key, skill_config_path, None))
                    else:
                        task_args.append((q_id, LLM_NAME, normalized_skill_name, skill_input, api_base, api_key, skill_config_path, PORT))
                else:
                    # Fallback to old format: LLM-Name:Query
                    target_llm = single_query.split(":")[0].strip().lower()
                    query_text = single_query.split(":")[1]
                    LLM_NAME, TAU, PORT = check_llm_name(target_llm=target_llm)
                    # For backward compatibility, treat as direct LLM call
                    task_args.append((q_id, query_text, TAU, LLM_NAME, api_base, api_key, None, PORT))
            else:
                task_args.append((q_id, "Format Error", "", "", api_base, api_key, skill_config_path, None))
        else:
            # Fallback to old format: LLM-Name:Query
            if ":" in single_query:
                target_llm = single_query.split(":")[0].strip().lower()
                query_text = single_query.split(":")[1]
                LLM_NAME, TAU, PORT = check_llm_name(target_llm=target_llm)
                task_args.append((q_id, query_text, TAU, LLM_NAME, api_base, api_key, None, PORT))
            else:
                task_args.append((q_id, "Format Error", "", "", api_base, api_key, skill_config_path, None))

    def _execute_task(task_arg):
        """Wrapper function to execute a single task (skill or LLM call)."""
        if len(task_arg) in [8, 9]:  # Support both 8 args (old) and 9 args (with extra_fields)
            # Check if it's skill format
            # Skill format: task_arg[1] is model_name, task_arg[2] is skill_name
            # Old format: task_arg[1] is query_text (string), task_arg[2] is TAU (number)
            model_name_candidate = task_arg[1] if len(task_arg) > 1 else ""
            skill_name_candidate = task_arg[2] if len(task_arg) > 2 else ""

            # Check if 1st arg (after q_id) is a model name and 2nd arg is a valid skill name
            is_skill_format = (
                model_name_candidate and
                skill_name_candidate and
                check_skill_name(skill_name_candidate)
            )

            if is_skill_format:
                # New skill format
                return request_skill_task(task_arg)
            else:
                # Old LLM format (remove extra_fields if present for backward compatibility)
                llm_args = task_arg[:7] if len(task_arg) >= 7 else task_arg
                return request_task(llm_args)
        else:
            # Fallback: try to determine by length
            if len(task_arg) >= 7:
                # Assume old format
                return request_task(task_arg[:7] if len(task_arg) > 7 else task_arg)
            else:
                # Error case
                return (task_arg[0] if task_arg else 0, "Invalid task format", 0.0)
    
    ret = []
    if deepseek_count > 2:
        thread_num = 24
    else:
        thread_num = 24
    # Use ThreadPoolExecutor to parallelize task execution
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        ret = list(executor.map(_execute_task, task_args))

    ret.sort(key=lambda x: x[0], reverse=False)
    resp = []
    completion_tokens_list = []
    for _, response, completion_tokens in ret:
        resp.append(response)
        completion_tokens_list.append(completion_tokens)

    return {"result": resp, "completion_tokens_list": completion_tokens_list}
