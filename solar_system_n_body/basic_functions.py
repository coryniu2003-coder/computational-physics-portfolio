"""
Basic N-body helpers for the solar system simulation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from particle3d import Particle3D


def compute_separations(particles: Iterable[Particle3D]) -> np.ndarray:
    particles = list(particles)
    n = len(particles)
    separations = np.zeros((n, n, 3))

    for i in range(n):
        for j in range(i + 1, n):
            separation = particles[i].position - particles[j].position
            separations[i, j] = separation
            separations[j, i] = -separation

    return separations


def compute_forces_potential(
    particles: Iterable[Particle3D],
    separations: np.ndarray,
    gravitational_constant: float = 8.887692593e-10,
) -> tuple[np.ndarray, float]:
    particles = list(particles)
    n = len(particles)
    forces = np.zeros((n, 3))
    potential = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            displacement = separations[i, j]
            distance = np.linalg.norm(displacement)
            if distance == 0.0:
                continue

            force_mag = (
                gravitational_constant * particles[i].mass * particles[j].mass / distance**2
            )
            force_vec = force_mag * displacement / distance

            forces[i] -= force_vec
            forces[j] += force_vec
            potential -= gravitational_constant * particles[i].mass * particles[j].mass / distance

    return forces, potential
