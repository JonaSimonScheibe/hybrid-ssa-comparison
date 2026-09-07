# `notebooks/` — interactive analyses

The notebooks carry the Lotka–Volterra work and the evaluation of the stored bacteriophage
T7 ensembles. Each of them starts with

```python
modul_path = "/home/jona/Schreibtisch/Studium/Semester4/Projektarbeit"
sys.path.append(modul_path)
import stochastinetics
```

Change that path to this repository's `src/` directory before running them. The stored
outputs are from the original runs, so the notebooks can also simply be read.

## `Lotka_Volterra.ipynb`

The largest of the three and the source of most Lotka–Volterra results. Its sections, in
order:

- model and parameter definition, deterministic solution
- visualisation of the stochastic reaction-event output
- CLE simulation, and the comparison of the reflective against the absorbing boundary
  condition at zero
- step size from CLE accuracy: copy-number distributions for a range of step sizes
- SSA reference run (about 28 minutes, according to the note in the cell)
- Duncan-SSA, then Duncan-Thinning — including the search for the upper bounds of the
  generalised propensities that the thinning variant needs
- comparison of Duncan-SSA against Duncan-Thinning
- Alfonsi, and the reproduction of the expectation-value figure with SSA, CLE and both hybrid
  methods
- expectation values as a function of the control parameter, for each algorithm
- average computation time
- weak order of convergence

## `Bacteriophage_T7_aktuell.ipynb`

Evaluation of the T7 model: model definition, deterministic solution, expectation values, and
the probability distributions over time. Reads the ensembles produced by the scripts rather
than simulating them itself.

## `Confirming_Alfonsi.ipynb`

The validation of the first implementations against two analytically solvable systems, before
any of the actual test models were touched:

1. a birth–death process, compared against the analytic mean and its Poisson stationary
   distribution,
2. a reversible dimerisation, whose conserved quantity makes the state space finite so that
   the master equation can be solved exactly by eigendecomposition.

Each system appears twice: once with the analytic solution alone, once as the comparison
against the simulation. In the stored state the comparison cells run `stochastinetics.hybrid`, that is Alfonsi; the calls to
`hybrid_duncan_ssa` and `SSA` sit next to them, commented out. Note that the plot legends say
"Gillespie" regardless of which of the three is active.
