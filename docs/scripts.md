# `scripts/` — batch runs

These are the runs that were too expensive for a notebook: the bacteriophage T7 ensembles and
the parameter studies. All of them were executed on the university HPC cluster.

## Before running any of them

Three things are hard-coded and have to be adjusted first.

1. **Module path.** Every script contains

   ```python
   modul_path = "/work/scheibe5/bacteriophage_t7/"
   ```

   which is where `stochastinetics.py` sat on the cluster. Point it at this repository's
   `src/` directory instead.
2. **Output paths.** Figures and CSV files are written to absolute paths under
   `/work/scheibe5/...`. Redirect them, ideally into `results/`, which is kept out of version
   control for exactly this purpose.
3. **`multiprocess`.** The scripts import `multiprocess`, not the standard library's
   `multiprocessing` — it serialises with `dill` and therefore copes with the closures and
   `functools.partial` objects used here. Install it (`pip install multiprocess`)


## The scripts

### `trajectories_bacteriophage_t7_model.py`

Generates the bacteriophage T7 trajectory ensemble with the Alfonsi algorithm, alongside the
deterministic solution from `solve_ivp` for comparison. The ensemble is written to CSV with
`save_sim_results`, so the expensive part is done once and every later evaluation reads the
file. Model constants are the ones of the report (`c1 = 0.025` … `c6 = 1.99`, initial state
`tem = 1`, `gen = struc = 0`), with `Delta_t = 1e-3`.

### `average_computation_time.py`

Mean wall-clock time per trajectory as a function of the simulated duration, for the T7
model.

### `expectation_values_para.py`

Parameter study on the **T7 model**: expectation values over
`parameter_values = [20, 40, 80, 160, 320, 640]`, computed for Alfonsi and for Duncan-SSA at
each value, with `n = 1000` trajectories over `t_range = np.arange(0, 200, 2)`. The parameter
enters the two algorithms differently — as the propensity threshold for Alfonsi, and as the
blending window `[0.5 p, p]` per species for Duncan. Writes
`expectation_comparison_alfonsi.png` and `expectation_comparison_duncan.png`.

### `stochastisches_regime.py`

Counts how many stochastic reaction events each algorithm produces per trajectory, on the
**Lotka–Volterra model**, for thresholds `[20, 40, 80, 160, 320, 640, 1280]` and `M = 1000`
runs per value.

> **Note:** this script unpacks three values from the algorithms
> (`_, _, n_alfonsi = stochastinetics.hybrid(...)`), while the current
> `src/stochastinetics.py` returns only `[t, x]`. It was written against a variant that also
> returned the event count, and needs that counter reinstated before it will run.

### `step_size_propensity.py`

Determines the step size up to which the propensities may be treated as constant.

1. `load_states_from_csv` reads a filtered SSA ensemble and builds a sampling base in which
   each visited state carries its residence time until the next event as a weight;
   `min_count` drops states below the blending threshold.
2. `sample_states_time_weighted` draws `M = 1000` states from it.
3. From each state, `n = 1000` short evolutions of length `delta_t` are run and the relative
   propensity change is recorded per reaction.
4. `tost_one_sample` performs the two one-sided tests against the equivalence margin
   `EPS = 0.05` at `ALPHA_LEVEL = 0.05`, and equivalence is concluded when
   `max(p_lower, p_upper) < alpha`. The one-sided p-values are computed from
   `scipy.stats.t` with `n - 1` degrees of freedom.

The step sizes to scan are set in `t_range`; the logarithmic sweep
`np.logspace(-3, -1, 5)` is commented out in favour of a single value, so uncomment it to
reproduce the whole table.

### `weak_order_of_convergence.py`

Estimates the weak order of convergence, `|E[X_n] - E[X(t)]| <= C * Delta_t**gamma`, by
comparing ensembles at different step sizes. It streams the stored ensembles through
`read_in_sim_results_gen` rather than loading them.
