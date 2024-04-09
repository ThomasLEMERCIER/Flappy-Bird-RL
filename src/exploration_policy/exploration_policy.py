"""
Module implementing exploration policies for Reinforcement Learning algorithms.
"""

import random
import numpy as np


class ExplorationPolicy:
    """
    Base class for exploration policies.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, state, actions, q_values):
        raise NotImplementedError("ExplorationPolicy.__call__")
