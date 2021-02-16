"""A bunch of models that can result in chimera states
"""
from typing import Callable
import numpy as np


def abrams_model(n: int, coupling: float, phase: float):
    """From Abrams et. al, 2004 (doi: 10.1103/PhysRevLett.101.084103)

    Two populations of oscillators. Intra-population coupling is 1,
    inter-population coupling is `coupling`.

    Arguments:
        n (int): The size of each population.
        coupling (float): The interpopulation coupling strength. Should be
            between 0 and 1.
        phase (float): The phase shift. Should be between 0 and 2 pi.

    Returns:
        Callable: A function (that can be put into an ODE solver) of the form
        `f(t, state)` using the given parameters.
    """
    k = np.ones([n, n])
    k[:n, :n] = coupling
    k[n:, n:] = coupling

    def out(t: float, state: np.ndarray):
        """The abrams model

        Args:
            t (float): The time (irrelevant to this model, but required for the
                scipy ODE solver).
            state (np.ndarray): The current state of the oscillators.

        Returns:
            np.ndarray: The time derivative of the state of the oscillators,
                as given by the Abrams model.
        """
        state_x, state_y = np.meshgrid(state, state)
        return -np.sum(k * np.sin(state_y - state_x - phase), axis=1) / n

    return out


def kuramoto_model():
    ...

abram = abrams_model(32, 0.6, 0)

abram()