"""
Data structures used in the project.
"""

from dataclasses import dataclass

import numpy as np
import torch

@dataclass(frozen=True)
class Transition:
    """
    Transition.
    """

    state: int
    action: int
    reward: float
    next_state: int
    done: bool

@dataclass(frozen=True)
class QLearningParams:
    """
    Parameters for the Q-Learning algorithm.
    """

    alpha: float
    num_actions: int
    gamma: float = 0.9


@dataclass(frozen=True)
class SarsaParams:
    """
    Parameters for the SARSA algorithm.
    """

    alpha: float
    num_actions: int
    gamma: float = 0.9

@dataclass(frozen=True)
class SarsaLambdaParams:
    """
    Parameters for the SARSA(λ) algorithm.
    """

    alpha: float
    num_actions: int
    gamma: float = 0.9
    lambd: float = 0.9

@dataclass(frozen=True)
class MonteCarloParams:
    """
    Parameters for the Monte Carlo algorithm.
    """

    num_actions: int
    gamma: float = 0.9
    epsilon: float = 0.1