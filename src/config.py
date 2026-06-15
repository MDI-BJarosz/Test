"""Configuration loader for Survey Helper Agent."""
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any


class Config:
    """Configuration class for Survey Helper Agent."""

    def __init__(self, config_path: str = None):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file. If None, defaults to configs/config.yaml
        """
        if config_path is None:
            # Default to configs/config.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "configs" / "config.yaml"

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please copy configs/config.example.yaml to configs/config.yaml "
                f"and fill in your API key."
            )

        # Load YAML config
        with open(self.config_path, 'r') as f:
            self._config: Dict[str, Any] = yaml.safe_load(f)

        # Validate required fields
        self._validate()

    def _validate(self):
        """Validate required configuration fields."""
        # Check for DATASET_TYPE field
        if 'DATASET_TYPE' not in self._config:
            raise ValueError(
                f"Missing required field 'DATASET_TYPE' in {self.config_path}\n"
                f"Please specify either 'tdsc' or 'nhts'"
            )

        dataset_type = self._config['DATASET_TYPE'].lower()
        if dataset_type not in ['tdsc', 'nhts']:
            raise ValueError(
                f"Invalid DATASET_TYPE '{dataset_type}'. Must be 'tdsc' or 'nhts'"
            )

        # Validate dataset-specific configuration exists
        if dataset_type.upper() not in self._config:
            raise ValueError(
                f"Missing dataset configuration section '{dataset_type.upper()}' in {self.config_path}"
            )

        # Validate AWS Bedrock configuration
        # Either use AWS_PROFILE or direct credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
        has_profile = 'AWS_PROFILE' in self._config and self._config['AWS_PROFILE'] and self._config['AWS_PROFILE'] != 'your_value_here'
        has_credentials = ('AWS_ACCESS_KEY_ID' in self._config and self._config['AWS_ACCESS_KEY_ID'] and
                          self._config['AWS_ACCESS_KEY_ID'] != 'your_value_here' and
                          'AWS_SECRET_ACCESS_KEY' in self._config and self._config['AWS_SECRET_ACCESS_KEY'] and
                          self._config['AWS_SECRET_ACCESS_KEY'] != 'your_value_here')

        if not has_profile and not has_credentials:
            raise ValueError(
                "AWS credentials not configured. Please set either:\n"
                "  1. AWS_PROFILE (recommended): Name of AWS profile from ~/.aws/credentials\n"
                "  2. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY: Direct credentials"
            )

        # AWS_REGION and BEDROCK_MODEL_ID are always required
        for field in ['AWS_REGION', 'BEDROCK_MODEL_ID']:
            if field not in self._config or not self._config[field] or self._config[field] == 'your_value_here':
                raise ValueError(f"{field} not set in config.yaml")

    @property
    def api_key(self) -> str:
        """Get Google Gemini API key (kept for backward compatibility)."""
        return self._config.get('API_KEY', '')

    @property
    def aws_profile(self) -> str:
        """Get AWS Profile name (optional, from ~/.aws/credentials)."""
        return self._config.get('AWS_PROFILE', '')

    @property
    def aws_access_key_id(self) -> str:
        """Get AWS Access Key ID for Bedrock (optional if using profile)."""
        return self._config.get('AWS_ACCESS_KEY_ID', '')

    @property
    def aws_secret_access_key(self) -> str:
        """Get AWS Secret Access Key for Bedrock (optional if using profile)."""
        return self._config.get('AWS_SECRET_ACCESS_KEY', '')

    @property
    def aws_region(self) -> str:
        """Get AWS Region for Bedrock."""
        return self._config['AWS_REGION']

    @property
    def bedrock_model_id(self) -> str:
        """Get Bedrock Model ID (e.g., anthropic.claude-3-5-sonnet-20241022-v2:0)."""
        return self._config['BEDROCK_MODEL_ID']

    @property
    def dataset_type(self) -> str:
        """Get the active dataset type ('tdsc' or 'nhts')."""
        return self._config['DATASET_TYPE'].lower()

    def get_dataset_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration for the active dataset.

        Returns:
            Dictionary containing dataset-specific file paths, mode_options, purpose_options, etc.
        """
        dataset_type = self.dataset_type.upper()
        return self._config[dataset_type]

    @property
    def mode_options(self) -> List[str]:
        """Get valid transportation mode options for the active dataset."""
        dataset_config = self.get_dataset_config()
        return dataset_config.get('mode_options', [])

    @property
    def purpose_options(self) -> List[str]:
        """Get valid trip purpose options for the active dataset."""
        dataset_config = self.get_dataset_config()
        return dataset_config.get('purpose_options', [])

    # Legacy properties for backward compatibility (deprecated)
    @property
    def households_file(self) -> str:
        """Get path to households CSV file (TDSC only, deprecated)."""
        if self.dataset_type == 'tdsc':
            return self.get_dataset_config().get('HOUSEHOLDS_FILE', '')
        raise ValueError("households_file property only available for TDSC dataset")

    @property
    def trips_file(self) -> str:
        """Get path to trips CSV file (TDSC only, deprecated)."""
        if self.dataset_type == 'tdsc':
            return self.get_dataset_config().get('TRIPS_FILE', '')
        raise ValueError("trips_file property only available for TDSC dataset")

    def get_mode_options_str(self) -> str:
        """Get mode options formatted as a comma-separated string."""
        return ', '.join(self.mode_options)

    def get_purpose_options_str(self) -> str:
        """Get purpose options formatted as a comma-separated string."""
        return ', '.join(self.purpose_options)


# Global config instance - loaded once when module is imported
config = Config()
