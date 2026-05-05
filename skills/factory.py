from enum import Enum

# Import skills directly from current directory (skills/)
from skills.none import NoSkill
from skills.Geometric_Problem_Solver import Geometric_Problem_Solver
from skills.Chart_Problem_Solver import Chart_Problem_Solver
from skills.Science_Problem_Solver import Science_Problem_Solver
from skills.Counting_Problem_Solver import Counting_Problem_Solver
from skills.Perception_Problem_Solver import Perception_Problem_Solver
from skills.Code_Problem_Solver import Code_Problem_Solver
from skills.zoom import Zoom
from skills.ERQA_Trajectory_Outcome_Solver import ERQA_Trajectory_Outcome_Solver
from skills.ERQA_Action_Adjustment_Solver import ERQA_Action_Adjustment_Solver
from skills.ERQA_Spatial_Mechanics_Solver import ERQA_Spatial_Mechanics_Solver
from skills.ERQA_Pointing_Localization_Solver import ERQA_Pointing_Localization_Solver
from skills.ERQA_Multi_View_Task_Solver import ERQA_Multi_View_Task_Solver
from skills.ERQA_Embodied_Scene_QA_Skill import ERQA_Embodied_Scene_QA_Skill
from skills.OCRBench_Text_Recognition_Skill import OCRBench_Text_Recognition_Skill
from skills.OCRBench_Key_Information_Extraction_Skill import OCRBench_Key_Information_Extraction_Skill
from skills.OCRBench_Scene_Text_QA_Skill import OCRBench_Scene_Text_QA_Skill
from skills.OCRBench_Document_Chart_QA_Skill import OCRBench_Document_Chart_QA_Skill
from skills.OCRBench_Formula_Recognition_Skill import OCRBench_Formula_Recognition_Skill
from skills.OCRBench_Master_Router_Skill import OCRBench_Master_Router_Skill
from skills.VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill import (
    VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill,
)
from skills.VLMsAreBlind_Geometric_Shape_Counting_Skill import (
    VLMsAreBlind_Geometric_Shape_Counting_Skill,
)
from skills.VLMsAreBlind_Grid_Structure_Parsing_Skill import (
    VLMsAreBlind_Grid_Structure_Parsing_Skill,
)
from skills.VLMsAreBlind_Highlighted_Character_Recognition_Skill import (
    VLMsAreBlind_Highlighted_Character_Recognition_Skill,
)
from skills.VLMsAreBlind_Intersection_Route_Counting_Skill import (
    VLMsAreBlind_Intersection_Route_Counting_Skill,
)
from skills.VLMsAreBlind_Master_Router_Skill import (
    VLMsAreBlind_Master_Router_Skill,
)
    
class SkillFactory:
    def __init__(self, client, skill_client, skill_model_name, model, system_prompt, question, images, have_image, temperature, top_p, timeout, image_paths=[], dataset=""):
        self.client = client
        self.model = model
        self.skill_client = skill_client
        self.skill_model_name = skill_model_name
        # self.messages = messages
        self.system_prompt = system_prompt
        self.question = question
        self.images = images
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.have_image = have_image
        self.image_paths = image_paths
        self.dataset = dataset
    
    def get_skill(self, skill):
        # import pdb; pdb.set_trace()
        if skill.skill_name == "none":
            return NoSkill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image)
        elif skill.skill_name in {"Geometric_Problem_Solver", "geometric_problem_solver", "geometric-problem-solver"}:
            return Geometric_Problem_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"Chart_Problem_Solver", "chart_problem_solver", "chart-problem-solver"}:
            return Chart_Problem_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"Science_Problem_Solver", "science_problem_solver", "science-problem-solver"}:
            return Science_Problem_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"Counting_Problem_Solver", "counting_problem_solver", "counting-problem-solver"}:
            return Counting_Problem_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"Perception_Problem_Solver", "perception_problem_solver", "perception-problem-solver"}:
            return Perception_Problem_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"Code_Problem_Solver", "code_problem_solver", "code-problem-solver"}:
            return Code_Problem_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"ERQA_Trajectory_Outcome_Solver", "erqa_trajectory_outcome_solver", "erqa-trajectory-outcome-solver"}:
            return ERQA_Trajectory_Outcome_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"ERQA_Action_Adjustment_Solver", "erqa_action_adjustment_solver", "erqa-action-adjustment-solver"}:
            return ERQA_Action_Adjustment_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"ERQA_Spatial_Mechanics_Solver", "erqa_spatial_mechanics_solver", "erqa-spatial-mechanics-solver"}:
            return ERQA_Spatial_Mechanics_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"ERQA_Pointing_Localization_Solver", "erqa_pointing_localization_solver", "erqa-pointing-localization-solver"}:
            return ERQA_Pointing_Localization_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"ERQA_Multi_View_Task_Solver", "erqa_multi_view_task_solver", "erqa-multi-view-task-solver"}:
            return ERQA_Multi_View_Task_Solver(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {
            "Embodied_Scene_QA_Skill",
            "embodied_scene_qa_skill",
            "embodied-scene-qa-skill",
            "ERQA_Embodied_Scene_QA_Skill",
            "erqa_embodied_scene_qa_skill",
            "erqa-embodied-scene-qa-skill",
        }:
            return ERQA_Embodied_Scene_QA_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"OCRBench_Text_Recognition_Skill", "ocrbench_text_recognition_skill", "ocrbench-text-recognition-skill"}:
            return OCRBench_Text_Recognition_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"OCRBench_Key_Information_Extraction_Skill", "ocrbench_key_information_extraction_skill", "ocrbench-key-information-extraction-skill"}:
            return OCRBench_Key_Information_Extraction_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"OCRBench_Scene_Text_QA_Skill", "ocrbench_scene_text_qa_skill", "ocrbench-scene-text-qa-skill"}:
            return OCRBench_Scene_Text_QA_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"OCRBench_Document_Chart_QA_Skill", "ocrbench_document_chart_qa_skill", "ocrbench-document-chart-qa-skill"}:
            return OCRBench_Document_Chart_QA_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {"OCRBench_Formula_Recognition_Skill", "ocrbench_formula_recognition_skill", "ocrbench-formula-recognition-skill"}:
            return OCRBench_Formula_Recognition_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {
            "ocr_problem_solver",
            "ocr-problem-solver",
            "OCR_Problem_Solver",
            "OCRBench_Master_Router_Skill",
            "ocrbench_master_router_skill",
            "ocrbench-master-router-skill",
        }:
            return OCRBench_Master_Router_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {
            "Circle_Contact_and_Overlap_Judge_Skill",
            "circle_contact_and_overlap_judge_skill",
            "circle-contact-and-overlap-judge-skill",
            "VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill",
            "vlmsareblind_circle_contact_overlap_judge_skill",
            "vlmsareblind-circle-contact-overlap-judge-skill",
        }:
            return VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {
            "Intersection_and_Route_Counting_Skill",
            "intersection_and_route_counting_skill",
            "intersection-and-route-counting-skill",
            "VLMsAreBlind_Intersection_Route_Counting_Skill",
            "vlmsareblind_intersection_route_counting_skill",
            "vlmsareblind-intersection-route-counting-skill",
        }:
            return VLMsAreBlind_Intersection_Route_Counting_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {
            "Grid_Structure_Parsing_Skill",
            "grid_structure_parsing_skill",
            "grid-structure-parsing-skill",
            "VLMsAreBlind_Grid_Structure_Parsing_Skill",
            "vlmsareblind_grid_structure_parsing_skill",
            "vlmsareblind-grid-structure-parsing-skill",
        }:
            return VLMsAreBlind_Grid_Structure_Parsing_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {
            "Highlighted_Character_Recognition_Skill",
            "highlighted_character_recognition_skill",
            "highlighted-character-recognition-skill",
            "VLMsAreBlind_Highlighted_Character_Recognition_Skill",
            "vlmsareblind_highlighted_character_recognition_skill",
            "vlmsareblind-highlighted-character-recognition-skill",
        }:
            return VLMsAreBlind_Highlighted_Character_Recognition_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {
            "Geometric_Shape_Counting_Skill",
            "geometric_shape_counting_skill",
            "geometric-shape-counting-skill",
            "VLMsAreBlind_Geometric_Shape_Counting_Skill",
            "vlmsareblind_geometric_shape_counting_skill",
            "vlmsareblind-geometric-shape-counting-skill",
        }:
            return VLMsAreBlind_Geometric_Shape_Counting_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name in {
            "Diagram_Reasoning_Skill",
            "diagram_reasoning_skill",
            "diagram-reasoning-skill",
            "Diagram_Reasoning_Router_Skill",
            "diagram_reasoning_router_skill",
            "diagram-reasoning-router-skill",
            "VLMsAreBlind_Master_Router_Skill",
            "vlmsareblind_master_router_skill",
            "vlmsareblind-master-router-skill",
            "vlmsareblind_problem_solver",
            "vlmsareblind-problem-solver",
        }:
            return VLMsAreBlind_Master_Router_Skill(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif skill.skill_name == "zoom":
            return Zoom(skill = skill, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        else:
            raise ValueError(f"Invalid skill: {skill.skill_name}")
        
