from dataclasses import dataclass
from typing import List

from loguru import logger
from openai import OpenAI

# Try to import config_loader, with fallback
try:
    from skills.config_loader import load_config
except ImportError:
    try:
        import sys
        import os
        # Try to find config_loader in parent directories
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from skills.config_loader import load_config
    except ImportError:
        # Fallback: create a simple load_config function
        def load_config(path):
            import yaml
            try:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception:
                return {"skills": []}
import hishel, httpx
import json, hashlib
from typing import Optional
# from hishel._utils import normalized_url

@dataclass
class Skill:
    skill_name: str
    skill_config: dict
    


def load_skills(config: dict) -> List[Skill]:
    skills = []
    for skill_config in config.get("skills", []):
        skills.append(Skill(skill_name=skill_config['name'], skill_config=skill_config))
    skill_names = [t.skill_name for t in skills]
    logger.info(
        f"Load {len(skills)} skills: {skill_names}"
    )
           
    return skills
# test
