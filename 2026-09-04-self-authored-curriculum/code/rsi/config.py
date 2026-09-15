"""Paths and fixed defaults. Everything the loop needs to find lives here, nothing model-editable."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("RSI_RUNS_DIR", PROJECT_DIR.parent / "runs"))
MODELS_DIR = Path(os.environ.get("RSI_MODELS_DIR", PROJECT_DIR.parent / "models"))
BASE_MODEL = MODELS_DIR / "Qwen3.5-9B"
PILOT_MODEL = MODELS_DIR / "Qwen3.5-4B"
MERGED_PATH = MODELS_DIR / "trainee-merged"          # fixed path: vLLM compile cache is keyed on it
MERGED_ALT_PATH = MODELS_DIR / "trainee-merged-alt"  # second fixed path for ablation / control adapters
VENVS = {k: PROJECT_DIR / "venvs" / k for k in ("serve", "train", "loop")}
HARBOR = VENVS["loop"] / "bin" / "harbor"
BASE_IMAGE = "rsi-base:latest"
DATASETS_DIR = Path(os.environ.get("RSI_DATASETS_DIR", PROJECT_DIR.parent / "datasets"))

VLLM_PORT = 8000
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}/v1"
SERVED_MODEL_NAME = "trainee"

@dataclass
class HarnessDefaults:
    """The L0 contract's numeric limits (docs/03 §5). Frozen for rollouts and evaluation alike."""
    max_turns: int = 30
    max_tokens_per_turn: int = 3072
    window_tokens: int = 16384          # W: training window; compaction triggers before it overflows
    summary_cap_tokens: int = 1024
    retained_turns: int = 2             # k turns kept after compaction
    max_compactions: int = 4
    command_timeout_sec: int = 120
    observation_cap_chars: int = 4000
    temperature: float = 1.0
    top_p: float = 1.0
    enable_thinking: bool = True
    brevity_sentence: bool = False

@dataclass
class RunPaths:
    root: Path
    def __post_init__(self):
        self.root = Path(self.root)
    @property
    def db(self) -> Path: return self.root / "rsi.sqlite"
    @property
    def tasks(self) -> Path: return self.root / "tasks"          # the task pool (Harbor layout, one dir per task)
    @property
    def jobs(self) -> Path: return self.root / "jobs"            # harbor job outputs
    @property
    def adapters(self) -> Path: return self.root / "adapters"
    @property
    def improver(self) -> Path: return self.root / "improver"    # git-versioned H_k
    @property
    def dashboard(self) -> Path: return self.root / "dashboard"
    @property
    def logs(self) -> Path: return self.root / "logs"
    def mkdirs(self):
        for p in (self.root, self.tasks, self.jobs, self.adapters, self.improver, self.dashboard, self.logs):
            p.mkdir(parents=True, exist_ok=True)

def run_paths(name: str) -> RunPaths:
    return RunPaths(RUNS_DIR / name)
