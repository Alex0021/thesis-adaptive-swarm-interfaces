import logging
import threading
from typing import Any

import requests

from workload_inference.experiments.data_structures import ExperimentStatus

API_TIMEOUT = 0.1  # seconds

# Suppress anoying http request deub messages
logging.getLogger("urllib3").setLevel(logging.WARNING)


class ExperimentAPIError(Exception):
    """Custom exception for errors related to the ExperimentAPI."""

    pass


class ExperimentAPI:
    def __init__(self, endpoint: str = "http://localhost", port: int = 8080):
        self.endpoint = endpoint
        self.port = port

    def get_experiment_state(self) -> ExperimentStatus:
        """Fetches the current state of the experiment from the API.

        Returns:
            ExperimentStatus: The current status (including the states) of the experiment.
        Raises:
            ExperimentAPIError: If there was an error making the HTTP request.
        """
        try:
            response = requests.get(
                f"{self.endpoint}:{self.port}/api/state", timeout=API_TIMEOUT
            )
            response.raise_for_status()
            return ExperimentStatus.from_dict(response.json())
        except Exception as e:
            raise ExperimentAPIError(f"Error fetching experiment state: {e}") from e

    def trigger_next_state(self) -> None:
        """Sends a request to the API to move to the next state in the experiment if possible.

        Raises:
            ExperimentAPIError: If there was an error making the HTTP request.
        """
        try:
            response = requests.get(
                f"{self.endpoint}:{self.port}/api/operatorclicked", timeout=API_TIMEOUT
            )
            response.raise_for_status()
        except Exception as e:
            raise ExperimentAPIError(f"Error triggering next state: {e}") from e

    def send_to(self, endpoint: str, data: Any) -> None:
        """Sends data to a specified API endpoint on a background thread.

        Launches a daemon thread to avoid blocking the caller.

        Args:
            endpoint: The API endpoint path starting from /api/ (e.g., "cwl/level").
            data: The data to send as JSON.
        """
        thread = threading.Thread(
            target=self._send_to_blocking,
            args=(endpoint, data),
            daemon=True,
        )
        thread.start()

    def _send_to_blocking(self, endpoint: str, data: Any) -> None:
        """Internal method that performs the actual HTTP POST request."""
        try:
            response = requests.post(
                f"{self.endpoint}:{self.port}/api/{endpoint}",
                json=data,
                timeout=API_TIMEOUT,
            )
            response.raise_for_status()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("Error sending data to %s: %s", endpoint, e)
