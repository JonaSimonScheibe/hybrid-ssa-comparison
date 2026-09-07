# `src/stochastinetics.py` — library reference

Everything in this project is built on one module. It provides the exact SSA, a pure CLE
integrator, the two hybrid algorithms compared in the report, and the helpers used to turn
ensembles of trajectories into expectation values, distributions and CSV files.

```python
import sys; sys.path.append("path/to/repo/src")
import stochastinetics
import numpy as np

# Lotka-Volterra:  A -> 2A,  A + B -> 2B,  B -> 0
S    = np.array([[1, -1, 0], [0, 1, -1]])          # stoichiometry
S_ed = np.array([[1,  1, 0], [0, 1,  1]], dtype=int)  # educts
c    = np.array([2.0, 0.002, 2.0])                 # rate constants
x0   = np.array([50, 60])

t, x = stochastinetics.SSA(S, S_ed, c, 0, x0, 5)
```

## Model representation

| Symbol | Meaning |
| --- | --- |
| `S` | stoichiometric matrix, rows = species, columns = reactions; entry = net change |
| `S_ed` | educt matrix, same shape, entries `abs(stoichiometry)` of the **reactants** |
| `c` | vector of volume-dependent rate constants, one per reaction |
| `x0` | initial molecule numbers, one entry per species |
| `t0`, `tf` | start and end time of the simulation |

`S_ed` is not redundant. In a reaction such as `A + B -> 2A` the net change of `A` is zero,
so `A` would not be recognised as a reactant from `S` alone and its propensity would be
wrong. `S_ed` records who actually takes part.

All trajectory functions return `[t, x]`: a 1-D array of event times and an array of states
whose rows correspond to those times. Times are irregular for the event-driven methods
(`SSA`, `hybrid`) and follow the step size for the CLE-driven ones.

## Propensities and blending weights

```python
propensity(reaction, state, c, S_ed, is_continous=False)
propensity_vec(reaction, state, c, S_ed, is_continous=False)
```

Propensity of a single reaction (by column index) or of a list of reactions. Zeroth, first
and second order including the two-identical-reactants case `x(x-1)/2` are handled; with
`is_continous=True` the continuous form `x**2/2` is used instead, which is what the CLE
needs. Unknown reaction orders return `0.0`.

```python
blendings(state, S, boundaries)
generalised_blendings(state, c, S, S_ed, boundaries, double_prime)
```

`blendings` evaluates the per-reaction weight `beta_j` of the Duncan scheme at the current
state. `boundaries` is a flat vector of length `2 * n_species`, holding `I1, I2` for each
species in species order. `generalised_blendings` returns the split propensities:
`beta_j * alpha_j` (the discrete share) or, with `double_prime=True`,
`(1 - beta_j) * alpha_j` (the continuous share).

## The CLE step

```python
cle(x0, c, delta_t, S, S_ed, boundaries=None)          # one step, returns the new state
cle_trajectory(S, S_ed, c, t0, x0, tf, Delta_t)        # full trajectory, returns [t, x]
```

One step of the chemical Langevin equation, integrated with the weak trapezoidal method of
Anderson et al. as used by Duncan et al. Passing `boundaries` switches to the blended
propensities, which is how the hybrid algorithms reuse this function; passing `None` gives a
pure CLE simulation.

## Trajectory algorithms

```python
SSA(S, S_ed, c, t0, x0, tf)
hybrid(S, S_ed, c, t0, x0, tf, part)
hybrid_duncan_ssa(S, S_ed, c, t0, x0, tf, delta_t, Delta_t, boundaries)
hybrid_duncan_thinning(S, S_ed, c, t0, x0, tf, delta_t, Delta_t, boundaries, lambda_bounds)
hybrid_duncan_next_reaction(S, S_ed, c, t0, x0, tf, delta_t, Delta_t, boundaries)
```

- **`SSA`** — Gillespie's exact algorithm. The reference every other method is measured
  against.
- **`hybrid`** — the algorithm of Alfonsi et al. (2005). `part` is the propensity threshold:
  reactions below it are treated stochastically, the rest deterministically. The
  deterministic subsystem is integrated with `scipy.integrate.solve_ivp` using the stiff BDF
  method, and the next stochastic event is found by its event detection on the accumulated
  stochastic propensity.
- **`hybrid_duncan_ssa`** — the blending scheme of Duncan et al. (2016) with an SSA step in
  the discrete part. `delta_t` is the step in the pure CLE regime, `Delta_t` the upper bound
  for the continuous step in the transition regime, and `boundaries` the blending windows as
  described above.
- **`hybrid_duncan_thinning`** — the same scheme with thinning: proposals are drawn from the
  constant upper bounds `lambda_bounds` (one per reaction, each at least the supremum of the
  corresponding blended propensity) and accepted with the appropriate probability.
- **`hybrid_duncan_next_reaction`** — a further variant using the next-reaction method in the
  discrete part. It is **not** part of the comparison in the report.

All five return `[t, x]`.

## Evaluating an ensemble

```python
expectation(result, n, t_range)                  # E[x](t) from an in-memory ensemble
expectation2(path, n_lines_per_run, t_range)     # the same, streamed from a CSV file
probability(result, t_range, state, n)           # P(X(t) = state) over t_range
get_distribution(result, n, ti, type="SSA")      # (states, probabilities) at one time ti
```

`result` is the list of `[t, x]` pairs collected over the runs, `n` their number.

`expectation2` exists because a full ensemble does not fit in memory on a cluster node with
a 4 GB limit; it reads the CSV run by run instead. `n_lines_per_run` is one row for the time
points plus one row per species.

`get_distribution` interpolates linearly between recorded steps, which is right for the
Alfonsi output. **Pass `type="Hybrid"` explicitly when evaluating SSA results** — the
argument name is the wrong way round, and the zero-order-hold branch is the one the SSA
needs.

## Storing and reading results

```python
save_sim_results(result, path)          # write an ensemble to CSV
read_in_sim_results(path, n)            # read it back as a list
read_in_sim_results_gen(path, n)        # the same, as a generator
```

The CSV layout is one row of time points per run, followed by one row per species. `n` is
the number of species plus one. Use the generator form for large files: it yields one run at
a time instead of building the whole list, which is what makes the cluster evaluations fit
into memory.
