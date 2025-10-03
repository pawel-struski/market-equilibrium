import yaml
import subprocess
import logging


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


def extract_price(number: str) -> float:
    """
    Parses the price response from the LLM. Expects a number in the str format.
    """
    clean_number = number.strip()
    try:
        clean_number_float = float(clean_number)
        return clean_number_float
    except ValueError:
        logging.info(f"Could not parse '{clean_number}' as float.")
        return None 
    

def extract_response(text: str) -> bool:
    """
    Extracts a boolean indicator of whether a deal is accepted from a textual response.
    """
    if "yes" in text.lower():
        return True
    else:
        return False

