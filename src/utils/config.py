import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration manager for the project."""
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = self.model_dump()
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def update(self, **kwargs: Any) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration from file or environment."""
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "configs/default_config.yaml")
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return Config.from_yaml(str(config_path)) 