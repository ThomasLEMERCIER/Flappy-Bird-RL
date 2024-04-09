"""
Module for MonteCarlo algorithms.
"""

import numpy as np
from collections import defaultdict

from ..exploration_policy import ExplorationPolicy
from ..utils.data_structures import Transition, MonteCarloParams


class OffPolicyMonteCarloAgent:
    """
    SARSA agent.
    """

    def __init__(
        self,
        params: MonteCarloParams,
    ) -> None:
        """
        Args:
            params (MonteCarloParams): Monte Carlo parameters.
        """
        self.params = params

        self.q_values = defaultdict(lambda: np.ones(self.params.num_actions))
        self.c = defaultdict(lambda: np.zeros(self.params.num_actions))

    def update(self, episode: list) -> None:
        """
        Updates the Q-values.

        Args:
            episode (list): Episode.
        """
        g = 0
        w = 1
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            g = self.params.gamma * g + reward
            self.c[state][action] += w
            self.q_values[state][action] += w / self.c[state][action] * (g - self.q_values[state][action])
            if action != np.argmax(self.q_values[state]):
                break
            w /= self.behavior_policy(state, action)

    def behavior_policy(self, state: int, action: int) -> float:
        """
        Returns the probability of selecting an action given a state.

        Args:
            state (int): Current state.
            action (int): Action.

        Returns:
            float: Probability of selecting the action.
        """
        if np.argmax(self.q_values[state]) == action:
            return 1 - self.params.epsilon + self.params.epsilon / self.params.num_actions
        return self.params.epsilon / self.params.num_actions

    def act(self, state: int) -> int:
        """
        Selects an action based on the exploration policy.

        Args:
            state (int): Current state.
        """
        if np.random.rand() < self.params.epsilon:
            return np.random.choice(self.params.num_actions)
        return np.argmax(self.q_values[state])

    def get_best_action(self, state: int) -> np.int64:
        """
        Returns the best action for a given state.

        Args:
            state (int): Current state.

        Returns:
            np.int64: Best action.
        """
        return np.argmax(self.q_values[state])

    def get_policy(self) -> np.ndarray:
        """
        Returns the policy.
        """
        return np.argmax(self.q_values, axis=1)

    def get_value_function(self) -> np.ndarray:
        """
        Returns the value function.
        """
        return np.max(self.q_values, axis=1)
