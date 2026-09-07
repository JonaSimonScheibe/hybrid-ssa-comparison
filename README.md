# Hybrid stochastic simulation algorithms for reaction kinetics

Python code accompanying the project work *Numerical comparison of hybrid stochastical
simulation algorithms for reaction kinetics* (University of Potsdam, group "Mathematical
Modelling and Systems Biology").

The report compares two hybrid schemes — the propensity-threshold method of Alfonsi et al.
and the blending framework of Duncan et al., in its Duncan-SSA and Duncan-Thinning variants —
against the exact stochastic simulation algorithm (SSA) and the chemical Langevin equation
(CLE), on a biochemical Lotka–Volterra model and the bacteriophage T7 replication model.

> **Status: the repository structure is in place, the code is not uploaded yet.**

## Structure

| Folder | Contents |
| --- | --- |
| `src/` | simulation library: SSA, CLE and the three hybrid algorithms |
| `notebooks/` | one notebook per model and experiment block |
| `scripts/` | run scripts for the HPC cluster |
| `figures/` | figures used in the report |
| `results/` | simulation output — **not** under version control (see `.gitignore`) |
| `docs/` | pseudocode and notes |

Trajectory ensembles reach several gigabytes and are therefore excluded from the repository;
`results/` stays empty here and is filled by running the code.

## Usage

```bash
pip install -r requirements.txt
```

Then run the notebooks in `notebooks/`.

## License

MIT — see [LICENSE](LICENSE).
