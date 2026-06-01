# Solar System N-body Simulation

A velocity-Verlet integrator for a compact solar system model. The code cleans
up the original coursework scaffolding with a friendlier command line
interface, modular helpers and basic diagnostics for sensible defaults.

## Usage

```bash
python simulation.py --input data/solar_system.txt --steps 9125 --dt 2 --plot
```

| option | description |
| ------ | ----------- |
| `--input` | Initial conditions file (XYZ-like format with label, mass, position and velocity). |
| `--steps` / `--dt` | Number of velocity-Verlet steps and timestep size in days. |
| `--xyz` | Optional XYZ trajectory export for visualisation in tools like Ovito. |
| `--energy-csv` | Write a CSV of time vs total energy for post-processing. |
| `--plot` | Display quick-look plots (Earth trajectory and total energy). |

After the run finishes a short summary of orbital distances is printed to help
spot-calculate whether the integration behaved as expected.
