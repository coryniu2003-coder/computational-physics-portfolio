"""
Utilities for simulating point particles interacting via the Morse potential.

The class defined here purposefully keeps the API close to the coursework
submission but tidies up the implementation and naming so it reads well in a
portfolio context.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Particle:
    """Simple point particle with mass, position and velocity."""

    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

    def update_position_euler(self, dt: float) -> None:
        """Advance the position using a symplectic Euler step."""
        self.position = self.position + dt * self.velocity

    def update_velocity_euler(self, dt: float, force: np.ndarray) -> None:
        """Advance the velocity using a symplectic Euler step."""
        self.velocity = self.velocity + dt * force / self.mass

    def update_position_verlet(self, dt: float, force: np.ndarray) -> None:
        """Advance the position using the velocity Verlet scheme."""
        acceleration = force / self.mass
        self.position = self.position + dt * self.velocity + 0.5 * (dt**2) * acceleration

    def update_velocity_verlet(
        self, dt: float, force_old: np.ndarray, force_new: np.ndarray
    ) -> None:
        """Advance the velocity using the velocity Verlet scheme."""
        self.velocity = self.velocity + 0.5 * dt * (force_old + force_new) / self.mass

    def kinetic_energy(self) -> float:
        """Return the kinetic energy."""
        return 0.5 * self.mass * float(np.dot(self.velocity, self.velocity))
