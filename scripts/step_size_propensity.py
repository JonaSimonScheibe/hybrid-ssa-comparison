import scipy
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
modul_path = "/work/scheibe5/bacteriophage_t7/"
if modul_path not in sys.path:
    sys.path.append(modul_path)
import stochastinetics
from tqdm import tqdm
import time
import multiprocess
from functools import partial


def load_states_from_csv(csv_path, min_count=None):
    """
    Liest die gefilterte, im Long-Format gespeicherte CSV ein und baut eine
    zeitgewichtete Stichprobenbasis: pro Zustand die Verweildauer bis zum
    naechsten Ereignis als Sampling-Gewicht. Optional Filterung auf
    min_count (z.B. I1), um nur Zustaende aus dem fuer den Hybridalgorithmus
    relevanten Regime zu behalten.
    """
    df = pd.read_csv(csv_path)
    species_cols = [c for c in df.columns if c.startswith("species_")]

    all_states = []
    all_weights = []
    for traj_id, g in df.groupby("traj_id", sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        X = g[species_cols].to_numpy()

        dwell = np.diff(t)
        states = X[:-1]  # letzte Zeile ist nur Endzeit-Marker

        if min_count is not None:
            mask = np.all(states >= min_count, axis=1)
            states = states[mask]
            dwell = dwell[mask]

        if len(states) > 0:
            all_states.append(states)
            all_weights.append(dwell)

    all_states = np.concatenate(all_states, axis=0)
    all_weights = np.concatenate(all_weights, axis=0)
    return all_states, all_weights


def sample_states_time_weighted(all_states, all_weights, M, rng=None):
    rng = rng or np.random
    p = all_weights / all_weights.sum()
    idx = rng.choice(len(all_states), size=M, replace=True, p=p)
    return all_states[idx]


def tost_one_sample(data, low, upp, alpha=0.05):
    data = np.asarray(data)
    n = data.size
    mean = data.mean()
    se = data.std(ddof=1) / np.sqrt(n)
    t_lower = (mean - low) / se
    p_lower = 1 - scipy.stats.t.cdf(t_lower, df=n - 1)
    t_upper = (mean - upp) / se
    p_upper = scipy.stats.t.cdf(t_upper, df=n - 1)
    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < alpha
    return p_tost, p_lower, p_upper, mean, equivalent


def CLE_diff_SSA(i, delta_t, n, S, S_ed, c, reaction, states):
    x0 = states[i]
    alpha0 = stochastinetics.propensity(reaction, x0, c, S_ed)

    rel_dev = []
    for _ in range(n):
        x_t_ssa = stochastinetics.SSA(S, S_ed, c, 0, x0, delta_t)[1][-1]
        alpha_ssa = stochastinetics.propensity(reaction, x_t_ssa, c, S_ed)
        if alpha0 == 0:
            continue
        rel_dev.append((alpha_ssa - alpha0) / alpha0)

    return rel_dev


def init_worker():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())


if __name__ == "__main__":
    M = 1000
    n = 1000
    k1 = 2.0
    k2 = 0.002
    k3 = 2.0
    I1 = 25.0
    I2 = 35.0
    boundaries = np.array([I1, I2, I1, I2])

    S = np.array([[1, -1, 0], [0, 1, -1]])
    S_ed = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    c = np.array([k1, k2, k3])

    CSV_PATH = "/work/scheibe5/bacteriophage_t7/result_lotka_volterra_ssa_filtered.csv"  
    print("Lade csv datei:\n")
    all_states, all_weights = load_states_from_csv(CSV_PATH, min_count=I1)
    print(f"Verfuegbare Zustaende nach Filterung: {len(all_states)}", flush=True)
    states_sample = sample_states_time_weighted(all_states, all_weights, M)

    EPS = 0.05
    ALPHA_LEVEL = 0.05

    start = time.time()
    num_cpus = len(os.sched_getaffinity(0))

    print(f"Starte Simulation auf {num_cpus} CPUs... (Gesamtanzahl: {n})", flush=True)

    #t_range = np.logspace(-3, -1, 5)
    t_range = [10**(-1)]
    for delta_t in t_range:
        for reaction in [1,2]:
            print(f"\n--- Delta_t: {delta_t} | Reaction: {reaction} ---", flush=True)
            worker_fn = partial(CLE_diff_SSA, delta_t=delta_t, n=n, S=S, S_ed=S_ed, c=c, reaction=reaction, states=states_sample)
            D = []
            with multiprocess.Pool(processes=num_cpus, initializer=init_worker) as pool:
                for idx, d in enumerate(pool.imap_unordered(worker_fn, range(M), chunksize=1), 1):
                    D.extend(d)
                    if idx % 10 == 0 or idx == M:
                        print(f"[PROGRESS] {idx}/{M} Simulationen beendet ({idx/M*100:.1f}%)", flush=True)

            p_tost, p_lower, p_upper, mean_rel_dev, equivalent = tost_one_sample(D, -EPS, EPS, alpha=ALPHA_LEVEL)
            print(f"Mittlere relative Abweichung: {mean_rel_dev:.4f}")
            print(f"TOST p-value: {p_tost:.4g} (p_lower={p_lower:.4g}, p_upper={p_upper:.4g})")
            print(f"Äquivalent innerhalb ±{EPS*100:.0f}% (alpha={ALPHA_LEVEL}): {equivalent}")
            D = np.array(D)
            print("Median:", np.median(D))
            print("90./99. Perzentil:", np.percentile(D, [90, 99]))
            print("Anteil exakt 0:", np.mean(D == 0))
            print("Max:", D.max(), "Min:", D.min())
    end = time.time()
    print(f"\nGesamtdauer: {end - start:.2f} Sekunden.")