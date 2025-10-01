import yaml

def load_config(config_path):
    """
    Loads a YAML config file and returns its contents as a dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
