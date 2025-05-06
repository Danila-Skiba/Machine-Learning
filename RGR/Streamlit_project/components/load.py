from typing import Dict, Any, Tuple
from pathlib import Path
import toml

def get_project_root() -> str:
    return str(Path(__file__).parent.parent)

def load_config(filename: str) -> Dict[Any, Any]:
    config = toml.load(Path(get_project_root()) / f"config/{filename}")
    return dict(config)
