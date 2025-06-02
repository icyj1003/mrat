from typing import Any, List

import numpy as np


class MarkovTransitionModel:
    """
    A Markov Transition Model for state transitions.
    """

    def __init__(self, states: List[Any], random_state=None):
        self.states = states
        self.num_states = len(states)
        self.transition_matrix = np.full(
            (self.num_states, self.num_states), 1 / self.num_states
        )
        self.random = random_state or np.random  # Use provided random state or default
        self.value = self.random.choice(self.states)

    def step(self) -> Any:
        current = self.states.index(self.value)
        next_state = self.random.choice(self.states, p=self.transition_matrix[current])
        self.value = next_state
        return next_state
