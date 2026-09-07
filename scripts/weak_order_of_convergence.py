import scipy
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
import time
import multiprocess
modul_path = "/work/scheibe5/bacteriophage_t7/"
if modul_path not in sys.path:
    sys.path.append(modul_path)
import stochastinetics


# Weak order of convergence: |E[X_n] - E[X(t)]| <= C* Delta_t^gamma
def fill_time_dict_gen(data, path,n_state, n_simulation, time_points):
    for current_result in stochastinetics.read_in_sim_results_gen(path,n_state):
        for j,ti in enumerate(time_points):
            idx = np.searchsorted(current_result[0], ti)
            idx = np.clip(idx, 1, len(current_result[1]) - 1)
            if abs(current_result[0][idx - 1] - ti) < abs(current_result[0][idx] - ti):
                idx -= 1
            data[ti].append(current_result[1][idx].copy())
    for _,ti in enumerate(time_points):
        data[ti] = np.array(data[ti])
    return data

def jump_count_alfonsi_duncan(i,S,S_ed,c,t0,x0,tf,threshold,delta_t,Delta_t):
    boundaries = np.array([threshold*0.5,threshold,threshold*0.5,threshold])
    _,X_alfonsi,N_alfonsi = stochastinetics.hybrid(S,S_ed,c,t0,x0,tf,threshold)
    _,X_duncan,N_duncan = stochastinetics.hybrid_duncan_ssa(S,S_ed,c,t0,x0,tf,delta_t,Delta_t,boundaries)
    return X_alfonsi,N_alfonsi,X_duncan,N_duncan

def init_worker():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())


if __name__ == "__main__":
    M = 10**5
    t0 = 0
    tf = 5
    k1 = 2.0
    k2 = 0.002
    k3 = 2.0
    A0 = 50
    B0 = 60
    Delta_t = 10**(-3)
    delta_t = 10**(-2)

    S = np.array([[1,-1,0],[0,1,-1]])
    S_ed = np.array([[1,1,0],[0,1,1]],dtype=int)
    c = np.array([k1,k2,k3]) 
    x0 = np.array([A0,B0])

    path_ssa= "/work/scheibe5/result_lotka_volterra_ssa.csv"
    time_points = [5]
    data_ssa = {5:[]}
    data_ssa = fill_time_dict_gen(data_ssa,path_ssa,3,1000,time_points)
    data_ssa = data_ssa[5]
    data_ssa = np.mean(data_ssa, axis=0)

    alfonsi_errors, alfonsi_dts = [], []
    duncan_errors, duncan_dts = [], []

    start = time.time()
    num_cpus = len(os.sched_getaffinity(0))

    print(f"Starte Simulation auf {num_cpus} CPUs... (Gesamtanzahl: {M})", flush=True)

    alfonsi_lambdas = [640,320,160,80,40,20]
    for threshold in alfonsi_lambdas:
        print(f"\n--- Propensity Threshold: {threshold}", flush=True)
        worker_fn2 = partial(jump_count_alfonsi_duncan, delta_t=delta_t,Delta_t = Delta_t, S=S, S_ed=S_ed, c=c, x0=x0,t0 =t0,tf=tf,threshold=threshold)
        total_N_alfonsi, total_N_duncan = 0, 0
        mean_alfonsi, mean_duncan = np.zeros((M,2)), np.zeros((M,2))
        with multiprocess.Pool(processes=num_cpus, initializer=init_worker) as pool:
            for idx,res in enumerate(pool.imap_unordered(worker_fn2, range(M), chunksize=1)):
                X_alfonsi, N_alfonsi, X_duncan, N_duncan = res

                total_N_alfonsi += N_alfonsi
                total_N_duncan += N_duncan

                mean_alfonsi[idx,:] = X_alfonsi[-1,:]
                mean_duncan[idx,:] = X_duncan[-1,:]

                completed = idx+1
                if idx % 10 == 0 or completed == M:
                    print(f"[PROGRESS] {completed}/{M} Simulationen beendet ({completed/M*100:.1f}%)", flush=True)

        mean_duncan = np.mean(mean_duncan, axis=0)
        mean_alfonsi = np.mean(mean_alfonsi,axis=0)

        alfonsi_errors.append(abs(mean_alfonsi-data_ssa))
        alfonsi_dts.append((tf*M)/total_N_alfonsi)
        duncan_errors.append(abs(mean_duncan-data_ssa))
        duncan_dts.append((tf*M)/total_N_duncan)

    print(f"alfonsi_errors: {alfonsi_errors}, alfonsi_dts: {alfonsi_dts}")
    print(f"duncan_errors: {duncan_errors}, duncan_dts: {duncan_dts}")
    plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'})

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5), sharey=True)

    duncan_err = np.array(duncan_errors)
    alfonsi_err = np.array(alfonsi_errors)

    color_A, color_B = '#1f77b4', '#ff7f0e'
    marker_A, marker_B = 'o', 's'

    axes[0].plot(duncan_dts, duncan_err[:, 0], label="Molecule A", color=color_A, marker=marker_A, linestyle='--', linewidth=1.5, markersize=6)
    axes[0].plot(duncan_dts, duncan_err[:, 1], label="Molecule B", color=color_B, marker=marker_B, linestyle='--', linewidth=1.5, markersize=6)
    axes[0].set_title("Duncan-SSA Hybrid", pad=12)

    axes[1].plot(alfonsi_dts, alfonsi_err[:, 0], label="Molecule A", color=color_A, marker=marker_A, linestyle='--', linewidth=1.5, markersize=6)
    axes[1].plot(alfonsi_dts, alfonsi_err[:, 1], label="Molecule B", color=color_B, marker=marker_B, linestyle='--', linewidth=1.5, markersize=6)
    axes[1].set_title("Alfonsi Hybrid", pad=12)

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"Average time step $\Delta t$")
        ax.grid(True, which="both", ls="--", lw=0.5)

    axes[0].set_ylabel(r"Weak Error $\left| \mathbb{E}[X(T)] - \mathbb{E}[X_L] \right|$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.02), 
        ncol=2, 
        frameon=False)

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)

    plt.savefig("/work/scheibe5/convergence_plot.png", dpi=300, bbox_inches='tight')
    plt.close()