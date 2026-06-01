"""
Numerical integration of a diatomic molecule bound by the Morse potential.

The script provides a small command line interface that mirrors the workflow of
the original coursework: load parameters from a `.dat` file, choose either the
symplectic Euler or the velocity Verlet scheme, run the simulation and export
the trajectory and energies. It is intentionally self-contained so it can be
shared as a portfolio example.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from particle import Particle


@dataclass
class MorseParameters:
    de: float
    re: float
    alpha: float
    dt: float
    steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("euler", "verlet"), default="verlet")
    parser.add_argument("--config", type=Path, default=Path("data/oxygen.dat"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--output", type=Path, default=Path("morse_simulation.csv"))
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def load_initial_conditions(path: Path) -> Tuple[Particle, Particle, MorseParameters]:
    params: MorseParameters | None = None
    particles: list[Particle] = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if tokens[0].lower() in {"p1", "p2"}:
                mass = float(tokens[1])
                position = np.array([float(x) for x in tokens[2:5]])
                velocity = np.array([float(x) for x in tokens[5:8]])
                particles.append(Particle(mass=mass, position=position, velocity=velocity))
            else:
                de, re, alpha, dt, steps = tokens[:5]
                params = MorseParameters(float(de), float(re), float(alpha), float(dt), int(float(steps)))

    if params is None or len(particles) != 2:
        raise ValueError(f"Could not parse a valid configuration from {path}")
    return particles[0], particles[1], params


def morse_force(p1: Particle, p2: Particle, params: MorseParameters) -> np.ndarray:
    displacement = p2.position - p1.position
    distance = np.linalg.norm(displacement)
    if distance == 0.0:
        return np.zeros(3)
    exp_term = np.exp(-params.alpha * (distance - params.re))
    magnitude = 2.0 * params.alpha * params.de * (1.0 - exp_term) * exp_term
    return magnitude * displacement / distance


def morse_potential(distance: float, params: MorseParameters) -> float:
    exp_term = np.exp(-params.alpha * (distance - params.re))
    return params.de * ((1.0 - exp_term) ** 2 - 1.0)


@dataclass
class SimulationResult:
    times: np.ndarray
    separations: np.ndarray
    energy: np.ndarray


def run_simulation(p1: Particle, p2: Particle, params: MorseParameters, mode: str) -> SimulationResult:
    times = np.zeros(params.steps)
    separations = np.zeros(params.steps)
    energy = np.zeros(params.steps)
    force12 = morse_force(p1, p2, params)

    for i in range(params.steps):
        times[i] = i * params.dt
        separations[i] = float(np.linalg.norm(p1.position - p2.position))
        energy[i] = p1.kinetic_energy() + p2.kinetic_energy() + morse_potential(separations[i], params)

        if mode == "euler":
            p1.update_position_euler(params.dt)
            p2.update_position_euler(params.dt)
            force12 = morse_force(p1, p2, params)
            p1.update_velocity_euler(params.dt, force12)
            p2.update_velocity_euler(params.dt, -force12)
        else:
            force21 = -force12
            p1.update_position_verlet(params.dt, force12)
            p2.update_position_verlet(params.dt, force21)
            new_force12 = morse_force(p1, p2, params)
            p1.update_velocity_verlet(params.dt, force12, new_force12)
            p2.update_velocity_verlet(params.dt, force21, -new_force12)
            force12 = new_force12

    return SimulationResult(times, separations, energy)


def write_csv(path: Path, result: SimulationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((result.times, result.separations, result.energy))
    np.savetxt(path, data, delimiter=",", header="time,interatomic_distance,total_energy", comments="")


def maybe_plot(result: SimulationResult) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(result.times, result.separations)
    axes[0].set_ylabel("Distance / Angstrom")
    axes[1].plot(result.times, result.energy, color="tab:orange")
    axes[1].set_ylabel("Energy / eV")
    axes[1].set_xlabel("Time / ps")
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    p1, p2, params = load_initial_conditions(args.config)
    if args.steps is not None:
        params.steps = args.steps
    if args.dt is not None:
        params.dt = args.dt
    result = run_simulation(p1, p2, params, args.mode)
    write_csv(args.output, result)
    if args.plot:
        maybe_plot(result)


if __name__ == "__main__":
    main()
