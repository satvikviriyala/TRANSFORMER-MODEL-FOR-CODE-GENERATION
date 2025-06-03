import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import Config, get_config


def test_config_creation():
    """Test basic configuration creation."""
    config = Config()
    assert isinstance(config, Config)


def test_config_from_yaml():
    """Test loading configuration from YAML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        test_config = {
            "model": {
                "architecture": "transformer",
                "hidden_size": 768
            }
        }
        yaml.dump(test_config, f)
        f.flush()
        
        config = Config.from_yaml(f.name)
        assert config.model.architecture == "transformer"
        assert config.model.hidden_size == 768


def test_config_save_yaml():
    """Test saving configuration to YAML."""
    config = Config()
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        config.save_yaml(f.name)
        assert Path(f.name).exists()
        
        # Verify saved content
        with open(f.name, "r") as saved_file:
            saved_config = yaml.safe_load(saved_file)
            assert isinstance(saved_config, dict)


def test_config_update():
    """Test updating configuration values."""
    config = Config()
    config.update(model={"architecture": "transformer"})
    assert config.model.architecture == "transformer"


def test_get_config():
    """Test getting configuration from environment or default."""
    # Test with default path
    config = get_config()
    assert isinstance(config, Config)
    
    # Test with custom path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        test_config = {"model": {"architecture": "test"}}
        yaml.dump(test_config, f)
        f.flush()
        
        config = get_config(f.name)
        assert config.model.architecture == "test"
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        get_config("non_existent.yaml") 