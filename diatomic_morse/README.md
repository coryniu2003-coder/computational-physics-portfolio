# Diatomic Morse Oscillator

This mini-project integrates the relative motion of a diatomic molecule in a
Morse potential. It is based on the core coursework exercise but tidied up for
public consumption with a small command line interface, clean module structure
and documentation.

## Run the simulation

```bash
python morse_simulation.py --config data/oxygen.dat --mode verlet --plot
```

| option | description |
| ------ | ----------- |
| `--config` | Path to a data file describing the potential and initial conditions (`oxygen.dat` and `nitrogen.dat` are provided). |
| `--mode` | Either `euler` or `verlet`; Verlet is the energy-conserving default. |
| `--steps` / `--dt` | Optional overrides for the integration length and timestep. |
| `--output` | CSV file where the time, bond length and total energy are recorded. |
| `--plot` | Show quick-look plots of the bond length and total energy. |

The data file format mirrors the original coursework specification so that any
existing inputs can be reused without modification.
