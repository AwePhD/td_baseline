from typing import Dict
from pathlib import Path
from yaml import safe_load


def build_path(path: str) -> Path:
    return Path(path).expanduser()


def get_config(config_file: Path) -> Dict:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = safe_load(f)

    return config
