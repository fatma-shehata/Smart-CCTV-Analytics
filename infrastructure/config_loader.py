"""
config_loader.py
Reads config.yaml and returns it as a Python dictionary.
"""

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Loads the YAML config file. Raises a clear error if the file is
    missing or has invalid syntax, instead of failing later with a
    confusing error deep in the pipeline."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file '{path}' not found. Make sure you're running "
            f"the script from inside the project root folder."
        )
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in config.yaml: {e}")