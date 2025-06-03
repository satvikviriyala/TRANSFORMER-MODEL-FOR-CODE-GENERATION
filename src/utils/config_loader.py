import yaml
import os
from dotenv import load_dotenv

def load_config(config_path="configs/config.yaml"):
    """Loads YAML config and substitutes environment variables."""
    load_dotenv() # Load .env file variables into environment

    with open(config_path, 'r') as f:
        config_str = f.read()

    # Basic environment variable substitution ${VAR_NAME}
    config_str_expanded = os.path.expandvars(config_str)

    try:
        config = yaml.safe_load(config_str_expanded)
        return config
    except yaml.YAMLError as e:
        print(f"Error loading YAML config: {e}")
        raise

# Example usage:
# if __name__ == "__main__":
#     config = load_config()
#     print(config)