"""
3D particle helper used by the N-body solar system simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Particle3D:
    label: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

    def __str__(self) -> str:
        x, y, z = self.position
        return f"{self.label} {x} {y} {z}"

    def momentum(self) -> np.ndarray:
        return self.mass * self.velocity

    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * float(np.dot(self.velocity, self.velocity))

    def update_position_1st(self, dt: float) -> None:
        self.position = self.position + dt * self.velocity

    def update_position_2nd(self, dt: float, force: np.ndarray) -> None:
        acceleration = force / self.mass
        self.position = self.position + dt * self.velocity + 0.5 * (dt**2) * acceleration

    def update_velocity(self, dt: float, force: np.ndarray) -> None:
        self.velocity = self.velocity + dt * force / self.mass

    @staticmethod
    def read_line(line: str) -> "Particle3D":
        tokens = line.split()
        label = tokens[0]
        mass = float(tokens[1])
        position = np.array([float(tokens[i]) for i in range(2, 5)])
        velocity = np.array([float(tokens[i]) for i in range(5, 8)])
        return Particle3D(label=label, mass=mass, position=position, velocity=velocity)

    @staticmethod
    def total_kinetic_energy(particles: Iterable["Particle3D"]) -> float:
        return sum(p.kinetic_energy() for p in particles)

    @staticmethod
    def com_velocity(particles: Iterable["Particle3D"]) -> np.ndarray:
        total_mass = sum(p.mass for p in particles)
        if total_mass == 0.0:
            return np.zeros(3)
        total_momentum = sum((p.mass * p.velocity) for p in particles)
        return total_momentum / total_mass
