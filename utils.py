import yaml
import subprocess


def load_config(config_path):
    """
    Loads a YAML config file and returns its contents as a dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit():
    """Get the current git commit hash."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        commit = "unknown"
    return commit
