"""
Velocity Verlet integrator for a simplified solar system.

The code started from coursework scaffolding but has been rewritten so the
structure, interfaces and documentation are suitable for a public portfolio.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from basic_functions import compute_forces_potential, compute_separations
from particle3d import Particle3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mini_system.txt"),
        help="Initial conditions file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3650,
        help="Number of integration steps to perform.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Timestep in days.",
    )
    parser.add_argument(
        "--xyz",
        type=Path,
        help="Optional XYZ trajectory output file.",
    )
    parser.add_argument(
        "--energy-csv",
        type=Path,
        help="Optional CSV file storing time and total energy.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show quick-look plots (requires matplotlib).",
    )
    return parser.parse_args()


def load_particles(path: Path) -> list[Particle3D]:
    particles: list[Particle3D] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            particles.append(Particle3D.read_line(line))
    return particles


def write_xyz_step(path: Path, particles: Iterable[Particle3D], step: int) -> None:
    with path.open("a") as handle:
        particles = list(particles)
        handle.write(f"{len(particles)}\n")
        handle.write(f"Step {step}\n")
        for particle in particles:
            handle.write(str(particle) + "\n")


@dataclass
class SimulationResult:
    times: np.ndarray
    energy: np.ndarray
    positions: np.ndarray  # shape (steps, n, 3)
    min_planet_distance: dict[str, float]
    max_planet_distance: dict[str, float]
    moon_distance_min: float | None
    moon_distance_max: float | None


def run_simulation(
    particles: list[Particle3D],
    steps: int,
    dt: float,
    xyz_path: Path | None = None,
) -> SimulationResult:
    n = len(particles)
    times = np.zeros(steps)
    energy = np.zeros(steps)
    positions = np.zeros((steps, n, 3))

    if xyz_path:
        xyz_path.parent.mkdir(parents=True, exist_ok=True)
        xyz_path.write_text("")

    # Remove centre-of-mass drift before integration.
    com_v = Particle3D.com_velocity(particles)
    for particle in particles:
        particle.velocity -= com_v

    separations = compute_separations(particles)
    forces, potential = compute_forces_potential(particles, separations)

    sun_index = next((i for i, p in enumerate(particles) if p.label.lower() == "sun"), None)
    earth_index = next(
        (i for i, p in enumerate(particles) if p.label.lower() == "earth"), None
    )
    moon_index = next(
        (i for i, p in enumerate(particles) if p.label.lower() == "moon"), None
    )

    min_planet_distance: dict[str, float] = {}
    max_planet_distance: dict[str, float] = {}
    if sun_index is not None:
        for idx, particle in enumerate(particles):
            if idx == sun_index:
                continue
            min_planet_distance[particle.label] = np.inf
            max_planet_distance[particle.label] = 0.0
    moon_distance_min = np.inf if earth_index is not None and moon_index is not None else None
    moon_distance_max = 0.0 if earth_index is not None and moon_index is not None else None

    for step in range(steps):
        times[step] = step * dt
        positions[step] = np.array([p.position for p in particles])
        energy[step] = Particle3D.total_kinetic_energy(particles) + potential

        if xyz_path:
            write_xyz_step(xyz_path, particles, step)

        if sun_index is not None:
            sun_position = particles[sun_index].position
            for idx, particle in enumerate(particles):
                if idx == sun_index:
                    continue
                distance = np.linalg.norm(particle.position - sun_position)
                min_planet_distance[particle.label] = min(
                    min_planet_distance[particle.label], distance
                )
                max_planet_distance[particle.label] = max(
                    max_planet_distance[particle.label], distance
                )

        if earth_index is not None and moon_index is not None:
            distance = np.linalg.norm(
                particles[moon_index].position - particles[earth_index].position
            )
            moon_distance_min = min(moon_distance_min, distance)
            moon_distance_max = max(moon_distance_max, distance)

        for i, particle in enumerate(particles):
            particle.update_position_2nd(dt, forces[i])

        separations = compute_separations(particles)
        new_forces, potential = compute_forces_potential(particles, separations)

        for i, particle in enumerate(particles):
            avg_force = 0.5 * (forces[i] + new_forces[i])
            particle.update_velocity(dt, avg_force)

        forces = new_forces

    if moon_distance_min is not None and not np.isfinite(moon_distance_min):
        moon_distance_min = None
    if moon_distance_max is not None and moon_distance_max == 0.0:
        moon_distance_max = None

    return SimulationResult(
        times=times,
        energy=energy,
        positions=positions,
        min_planet_distance=min_planet_distance,
        max_planet_distance=max_planet_distance,
        moon_distance_min=moon_distance_min,
        moon_distance_max=moon_distance_max,
    )


def write_energy_csv(path: Path, result: SimulationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((result.times, result.energy))
    np.savetxt(path, data, delimiter=",", header="time_days,total_energy", comments="")


def maybe_plot(result: SimulationResult, particles: list[Particle3D]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    earth_idx = next((i for i, p in enumerate(particles) if p.label.lower() == "earth"), None)
    sun_idx = next((i for i, p in enumerate(particles) if p.label.lower() == "sun"), None)

    if earth_idx is not None and sun_idx is not None:
        axes[0].plot(
            result.positions[:, earth_idx, 0] - result.positions[:, sun_idx, 0],
            result.positions[:, earth_idx, 1] - result.positions[:, sun_idx, 1],
            lw=1.2,
        )
        axes[0].set_title("Earth trajectory in the barycentric frame")
        axes[0].set_ylabel("y / AU")

    axes[1].plot(result.times, result.energy, color="tab:orange", lw=1.2)
    axes[1].set_title("Total energy (kinetic + potential)")
    axes[1].set_xlabel("Time / days")
    axes[1].set_ylabel("Energy / $M_\oplus$ AU$^2$ day$^{-2}$")

    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    particles = load_particles(args.input)
    result = run_simulation(particles, steps=args.steps, dt=args.dt, xyz_path=args.xyz)

    if args.energy_csv:
        write_energy_csv(args.energy_csv, result)

    if args.plot:
        maybe_plot(result, particles)

    print("Simulation complete.")
    if result.min_planet_distance:
        print("Closest approach to the Sun (AU):")
        for label, distance in result.min_planet_distance.items():
            print(f"  {label:<10} {distance: .3f}")
    if result.max_planet_distance:
        print("Farthest distance from the Sun (AU):")
        for label, distance in result.max_planet_distance.items():
            print(f"  {label:<10} {distance: .3f}")
    if result.moon_distance_min is not None and result.moon_distance_max is not None:
        print(
            f"Moon-Earth distance ranged from {result.moon_distance_min:.3f} to "
            f"{result.moon_distance_max:.3f} AU"
        )


if __name__ == "__main__":
    main()
