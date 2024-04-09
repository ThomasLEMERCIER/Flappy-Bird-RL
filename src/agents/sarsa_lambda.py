"""
Module for SARSA Lambda algorithm.
"""

import numpy as np
from collections import defaultdict

from ..exploration_policy import ExplorationPolicy
from ..utils.data_structures import Transition, SarsaLambdaParams


class SarsaLambdaAgent:
    """
    SARSA agent.
    """

    def __init__(
        self,
        params: SarsaLambdaParams,
        exploration_policy: ExplorationPolicy,
    ) -> None:
        """
        Args:
            params (SarsaParams): SARSA parameters.
            exploration_policy (ExplorationPolicy): Exploration policy.
        """
        self.params = params
        self.exploration_policy = exploration_policy

        self.q_values = defaultdict(lambda: np.ones(self.params.num_actions))
        self.eligibility_traces = defaultdict(lambda: np.zeros(self.params.num_actions))

    def update(self, transition: Transition, next_action: int) -> None:
        """
        Updates the Q-values.

        Args:
            transition (Transition): Transition.
        """
        delta = (
            transition.reward
            + self.params.gamma * self.q_values[transition.next_state][next_action] * (1 - transition.done)
            - self.q_values[transition.state][transition.action]
        )

        self.eligibility_traces[transition.state][transition.action] += 1

        for state in self.q_values.keys():
            for action in range(self.params.num_actions):
                self.q_values[state][action] += self.params.alpha * delta * self.eligibility_traces[state][action]
                self.eligibility_traces[state][action] *= self.params.gamma * self.params.lambd

    def act(self, state: int) -> int:
        """
        Selects an action based on the exploration policy.

        Args:
            state (int): Current state.
        """
        return self.exploration_policy(
            state, np.arange(self.params.num_actions), self.q_values
        )

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
