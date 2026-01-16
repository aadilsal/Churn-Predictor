"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class Config:
    """Application configuration manager."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_env()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value
        """
        return os.getenv(key, default)

    @property
    def data_raw_path(self) -> Path:
        """Get raw data directory path."""
        return Path(self.get("data.raw_path", "data/raw"))

    @property
    def data_processed_path(self) -> Path:
        """Get processed data directory path."""
        return Path(self.get("data.processed_path", "data/processed"))

    @property
    def data_features_path(self) -> Path:
        """Get features data directory path."""
        return Path(self.get("data.features_path", "data/features"))

    @property
    def random_seed(self) -> int:
        """Get random seed for reproducibility."""
        return self.get("model.random_seed", 42)

    @property
    def mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI."""
        return self.get_env("MLFLOW_TRACKING_URI", "http://localhost:5000")


# Global configuration instance
config = Config()
