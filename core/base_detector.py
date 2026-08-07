"""
base_detector.py
Defines the common interface that any detection model (Detection or
Segmentation) must follow. The rest of the project only talks to this
interface — it never needs to know which concrete model is running.
"""

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Abstract base class (interface) for all detector implementations."""

    @abstractmethod
    def load(self) -> None:
        """Loads the model weights into memory. Called once at startup."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, frame):
        """
        Runs detection (and tracking) on a single frame.
        Must return a raw result object that tracker.py knows how to
        convert into TrackedObject instances.
        """
        raise NotImplementedError