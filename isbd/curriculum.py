"""
Dynamic Curriculum Learning Scheduler for ISBD AI
=================================================
Manages ordered progression from coarse boundary segmentation to fine
micro-detail and alpha-matting synthesis.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CURRICULUM_STATE_FILE = DATA / "curriculum_state.json"

STAGES = {
    1: {
        "name": "solid_contrast",
        "title": "Level 1: Coarse Boundary & Solid Contrast",
        "description": "Background extraction, clipping paths, and high-contrast silhouettes",
        "target_loss": 0.038,
        "weights": {"l1": 1.0, "ssim": 0.5, "edge": 0.3, "color": 0.2},
        "min_steps": 60
    },
    2: {
        "name": "texture_skin",
        "title": "Level 2: Micro-Texture & Skin Retouching",
        "description": "Skin pores preservation, subtle blemish blending, and tonal gradation",
        "target_loss": 0.024,
        "weights": {"l1": 0.8, "ssim": 1.0, "edge": 0.7, "color": 0.4},
        "min_steps": 120
    },
    3: {
        "name": "fine_alpha_details",
        "title": "Level 3: Fine Alpha Matting & Jewelry Facets",
        "description": "Hair strands, translucent lace, specular reflections, and sub-pixel edge feathering",
        "target_loss": 0.015,
        "weights": {"l1": 0.6, "ssim": 1.1, "edge": 1.4, "color": 0.5},
        "min_steps": 200
    }
}


class CurriculumScheduler:
    def __init__(self, state_file: Path = CURRICULUM_STATE_FILE):
        self.state_file = state_file
        self.current_level = 1
        self.total_steps = 0
        self.level_steps = 0
        self.loss_ema = 0.045
        self.history = []
        self.load()

    def load(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.current_level = int(data.get("current_level", 1))
                self.total_steps = int(data.get("total_steps", 0))
                self.level_steps = int(data.get("level_steps", 0))
                self.loss_ema = float(data.get("loss_ema", 0.045))
                self.history = data.get("history", [])
            except Exception as e:
                print(f"[Curriculum] Failed to load state: {e}")

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_level": self.current_level,
            "stage_info": STAGES.get(self.current_level, STAGES[1]),
            "total_steps": self.total_steps,
            "level_steps": self.level_steps,
            "loss_ema": round(self.loss_ema, 5),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "history": self.history[-30:]
        }
        try:
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Curriculum] Failed to write state: {e}")

    def update(self, batch_loss: float) -> Dict[str, Any]:
        """Updates curriculum moving average loss and checks progression."""
        batch_loss = float(batch_loss)
        self.total_steps += 1
        self.level_steps += 1
        self.loss_ema = 0.92 * self.loss_ema + 0.08 * batch_loss

        stage = STAGES.get(self.current_level, STAGES[3])
        advanced = False
        if self.current_level < 3:
            if self.level_steps >= stage["min_steps"] and self.loss_ema <= stage["target_loss"]:
                self.history.append({
                    "from_level": self.current_level,
                    "to_level": self.current_level + 1,
                    "at_step": self.total_steps,
                    "loss_ema": round(self.loss_ema, 5),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                self.current_level += 1
                self.level_steps = 0
                advanced = True

        self.save()
        return {
            "current_level": self.current_level,
            "stage_name": STAGES[self.current_level]["name"],
            "advanced": advanced,
            "loss_ema": self.loss_ema,
            "weights": STAGES[self.current_level]["weights"]
        }

    def get_stage_weights(self) -> Dict[str, float]:
        return STAGES.get(self.current_level, STAGES[1])["weights"]

    def get_info(self) -> Dict[str, Any]:
        stage = STAGES.get(self.current_level, STAGES[1])
        return {
            "current_level": self.current_level,
            "level_name": stage["name"],
            "title": stage["title"],
            "description": stage["description"],
            "target_loss": stage["target_loss"],
            "loss_ema": round(self.loss_ema, 5),
            "level_steps": self.level_steps,
            "total_steps": self.total_steps,
            "weights": stage["weights"]
        }


def get_curriculum_scheduler() -> CurriculumScheduler:
    return CurriculumScheduler()
