"""Velocity-Verlet N-body simulation used in the computational physics portfolio.

The script is intentionally lightweight: it reads a simple particle file,
integrates the system with a velocity-Verlet scheme, and can write energy and
trajectory outputs for inspection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from basic_functions import total_energy, update_all_forces, velocity_verlet_step
from particle3d import Particle3D


G = 6.67430e-11


def load_particles(path: Path) -> list[Particle3D]:
    """Load particles from rows: label mass x y z vx vy vz."""
    particles: list[Particle3D] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"{path}:{line_number}: expected 8 fields, got {len(fields)}")
        label = fields[0]
        mass = float(fields[1])
        position = np.array([float(v) for v in fields[2:5]], dtype=float)
        velocity = np.array([float(v) for v in fields[5:8]], dtype=float)
        particles.append(Particle3D(label, mass, position, velocity))
    if not particles:
        raise ValueError(f"No particles loaded from {path}")
    return particles


def write_xyz_frame(handle, particles: Iterable[Particle3D], step: int) -> None:
    particles = list(particles)
    handle.write(f"{len(particles)}\n")
    handle.write(f"step={step}\n")
    for particle in particles:
        x, y, z = particle.position
        handle.write(f"{particle.label} {x:.10e} {y:.10e} {z:.10e}\n")


def run_simulation(
    particles: list[Particle3D],
    dt: float,
    steps: int,
    sample_every: int,
    energy_path: Path | None,
    trajectory_path: Path | None,
) -> tuple[list[float], list[float]]:
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")

    update_all_forces(particles, G)
    times: list[float] = []
    energies: list[float] = []

    energy_file = energy_path.open("w", newline="") if energy_path else None
    trajectory_file = trajectory_path.open("w") if trajectory_path else None
    try:
        writer = None
        if energy_file:
            writer = csv.writer(energy_file)
            writer.writerow(["step", "time_s", "total_energy_j"])

        for step in range(steps + 1):
            if step % sample_every == 0:
                time = step * dt
                energy = total_energy(particles, G)
                times.append(time)
                energies.append(energy)
                if writer:
                    writer.writerow([step, f"{time:.10e}", f"{energy:.10e}"])
                if trajectory_file:
                    write_xyz_frame(trajectory_file, particles, step)

            if step < steps:
                velocity_verlet_step(particles, dt, G)
    finally:
        if energy_file:
            energy_file.close()
        if trajectory_file:
            trajectory_file.close()

    return times, energies


def plot_energy(times: list[float], energies: list[float], output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(times, energies, linewidth=1.5)
    plt.xlabel("Time / s")
    plt.ylabel("Total energy / J")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small N-body simulation")
    parser.add_argument("input", type=Path, nargs="?", default=Path("data/mini_system.txt"))
    parser.add_argument("--dt", type=float, default=1000.0, help="Time step in seconds")
    parser.add_argument("--steps", type=int, default=1000, help="Number of integration steps")
    parser.add_argument("--sample-every", type=int, default=10, help="Output stride in steps")
    parser.add_argument("--energy-csv", type=Path, default=None, help="Optional energy CSV output")
    parser.add_argument("--trajectory", type=Path, default=None, help="Optional XYZ trajectory output")
    parser.add_argument("--plot", type=Path, default=None, help="Optional energy plot output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    particles = load_particles(args.input)
    times, energies = run_simulation(
        particles=particles,
        dt=args.dt,
        steps=args.steps,
        sample_every=args.sample_every,
        energy_path=args.energy_csv,
        trajectory_path=args.trajectory,
    )

    if args.plot:
        plot_energy(times, energies, args.plot)

    drift = energies[-1] - energies[0]
    print(f"Loaded {len(particles)} particles from {args.input}")
    print(f"Initial energy: {energies[0]:.6e} J")
    print(f"Final energy:   {energies[-1]:.6e} J")
    print(f"Energy drift:   {drift:.6e} J")


if __name__ == "__main__":
    main()
