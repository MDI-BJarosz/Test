"""Base adapter interface for different survey datasets."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class BaseDatasetAdapter(ABC):
    """
    Abstract base class defining the interface for dataset adapters.

    Each dataset adapter translates dataset-specific file structures and column names
    into a common interface used by the prediction agent.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter with dataset-specific configuration.

        Args:
            config: Dictionary containing dataset-specific file paths and settings
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate that all required configuration fields are present.

        Raises:
            ValueError: If required fields are missing from config
        """
        pass

    @abstractmethod
    def get_person_attributes(self, person_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve all attributes for a specific person.

        Args:
            person_identifier: Person identifier. Format depends on dataset:
                              - Simple string for datasets with globally unique person IDs (TDSC)
                              - Composite "household_id:person_id" for datasets requiring household context (NHTS)

        Returns:
            Dictionary containing person attributes, or None if person not found.
            Should include demographics, employment, household info, etc.
        """
        pass

    @abstractmethod
    def get_trip_history(self, person_identifier: str, trip_identifier: str) -> List[Dict[str, Any]]:
        """
        Retrieve all trips for a person that occurred before the specified trip.

        Args:
            person_identifier: Person identifier (format depends on dataset)
            trip_identifier: Trip identifier (format depends on dataset)
                            - Simple string for TDSC
                            - Composite "household_id:person_id:trip_id" for NHTS

        Returns:
            List of dictionaries containing trip attributes, ordered chronologically.
            Returns empty list if person or trip not found.
        """
        pass

    @abstractmethod
    def get_current_trip_attributes(self, trip_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve attributes for the current trip, including derived features.

        Args:
            trip_identifier: Trip identifier (format depends on dataset)
                            - Simple string for TDSC
                            - Composite "household_id:person_id:trip_id" for NHTS

        Returns:
            Dictionary containing trip attributes plus derived features like:
            - speed_mps: Average speed in meters per second
            - speed_interpretation: Human-readable speed category
            - distance/duration categories
            Or None if trip not found.
        """
        pass

    @abstractmethod
    def get_mode_options(self) -> List[str]:
        """
        Get list of valid transportation mode options for this dataset.

        Returns:
            List of mode strings (e.g., ['walk', 'bike', 'drove_alone', ...])
        """
        pass

    @abstractmethod
    def get_purpose_options(self) -> List[str]:
        """
        Get list of valid trip purpose options for this dataset.

        Returns:
            List of purpose strings (e.g., ['work', 'home', 'shopping', ...])
        """
        pass

    @abstractmethod
    def get_prediction_guidance(self) -> str:
        """
        Get dataset-specific guidance for the prediction agent.

        Returns:
            String containing dataset-specific tips for mode and purpose prediction,
            including which features to prioritize and how to interpret them.
        """
        pass
