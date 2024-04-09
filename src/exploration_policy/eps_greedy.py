import random
import numpy as np

from .exploration_policy import ExplorationPolicy


class EpsilonGreedy(ExplorationPolicy):
    """
    Epsilon-greedy exploration policy.
    """

    def __init__(self, epsilon: float = 0.1, decay: float = 1) -> None:
        """
        Args:
            epsilon (float, optional): Probability of selecting a random action. Defaults to 0.1.
        """
        super().__init__()
        self.epsilon = epsilon
        self.decay = decay

    def __call__(self, state: int, actions: np.ndarray, q_values: np.ndarray):
        """
        Selects an action based on the epsilon-greedy policy.

        Args:
            state (int): Current state.
            actions (np.ndarray): List of possible actions from the current state.
            q_values (np.ndarray): Q-values
        """
        if random.random() < self.epsilon:
            action =  random.choice(actions)
        else:
            action =  actions[np.argmax(q_values[state][actions])]

        self.epsilon *= self.decay
        return action
