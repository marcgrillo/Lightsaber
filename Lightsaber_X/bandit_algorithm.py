from abc import ABC, abstractmethod
import numpy as np

class BanditAlgorithm(ABC):
    """
    Abstract base class for Multi-Armed Bandit algorithms.
    """

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.t = 0 # Time step

    @abstractmethod
    def select_arm(self):
        """
        Selects an arm to play for the current time step.
        Returns:
            int: The index of the selected arm (0 to num_arms - 1).
        """
        pass

    @abstractmethod
    def update(self, arm, reward):
        """
        Updates the algorithm's internal state with the observed reward.
        Args:
            arm (int): The index of the arm that was played.
            reward (float): The observed reward.
        """
        pass

    @abstractmethod
    def reset(self):
        """Resets the algorithm state."""
        pass