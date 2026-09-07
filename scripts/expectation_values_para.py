import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless Mode für HPC Cluster
import matplotlib.pyplot as plt
import seaborn as sns
import time
import multiprocess
from functools import partial

# Modul-Pfad einbinden
modul_path = "/work/scheibe5/bacteriophage_t7/"
if modul_path not in sys.path:
    sys.path.append(modul_path)
import stochastinetics

# --- BACTERIOPHAGE T7 SYSTEM PARAMETER ---
k1 = 0.025
k2 = 0.25
k3 = 1.0
k4 = 7.5 * 10**(-6)
k5 = 1000
k6 = 1.99
tem0 = 1
gen0 = 0
struc0 = 0
Delta_t = 10**(-3)
delta_t = 10**(-2)

# Stoichiometry Matrix [gen, tem, struc]
S = np.array([[-1, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0], [0, 0, 0, -1, 1, -1]])
S_ed = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1]], dtype=int)
c = np.array([k1, k2, k3, k4, k5, k6])
x0 = np.array([gen0, tem0, struc0])

t0 = 0
n = 1000
t_range = np.arange(0, 200,2)
t_max = np.max(t_range) + 0.01
parameter_values = [20, 40, 80, 160, 320, 640]

def init_worker():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())

def run_single_alfonsi(i, para, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=t_max):
    t_alf, z_alf = stochastinetics.hybrid(S, S_ed, c, t0, x0, tf, part=para)
    return t_alf, z_alf

def run_single_duncan(i, para, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=t_max, delta_t=delta_t, Delta_t=Delta_t):
    # Dynamische Grenzen basierend auf dem System (3 Spezies: gen, tem, struc)
    boundaries = np.array([para * 0.5, para, para * 0.5, para, para * 0.5, para])
    t_dunc, z_dunc = stochastinetics.hybrid_duncan_ssa(S, S_ed, c, t0, x0, tf, delta_t, Delta_t, boundaries)
    return t_dunc, z_dunc

if __name__ == "__main__":
    num_cpus = len(os.sched_getaffinity(0))
    print(f"Starte HPC-Evaluation auf {num_cpus} CPUs... (n={n} Trajektorien)", flush=True)
    
    exp_comparison_alfonsi = np.zeros((len(parameter_values), len(t_range)))
    exp_comparison_duncan = np.zeros((len(parameter_values), len(t_range)))

    for k, para in enumerate(parameter_values):
        print(f"\n==========================================", flush=True)
        print(f" Starte Parameter / Threshold: {para}", flush=True)
        print(f"==========================================", flush=True)
        
        # --- 1. ALFONSI SIMULATIONEN ---
        print(f"--> Berechne Alfonsi Runs...", flush=True)
        sum_alfonsi = np.zeros(len(t_range))
        worker_alfonsi = partial(run_single_alfonsi, para=para, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=t_max)
        
        count_alf = 0
        with multiprocess.Pool(processes=num_cpus, initializer=init_worker) as pool:
            for t_alf, z_alf in pool.imap_unordered(worker_alfonsi, range(n), chunksize=1):
                count_alf += 1
                z_alf_vals = z_alf[:, 0] if z_alf.ndim > 1 else z_alf
                
                for j, tj in enumerate(t_range):
                    idx = max(0, np.searchsorted(t_alf, tj, side='right') - 1)
                    sum_alfonsi[j] += z_alf_vals[idx]
                
                if count_alf % 10 == 0 or count_alf == n:
                    print(f"   [Alfonsi] {count_alf}/{n} Simulationen beendet ({count_alf/n*100:.1f}%)", flush=True)
                    
        exp_comparison_alfonsi[k, :] = sum_alfonsi / n

        # --- 2. DUNCAN SIMULATIONEN ---
        print(f"--> Berechne Duncan Runs...", flush=True)
        sum_duncan = np.zeros(len(t_range))
        worker_duncan = partial(run_single_duncan, para=para, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=t_max)
        
        count_dunc = 0
        with multiprocess.Pool(processes=num_cpus, initializer=init_worker) as pool:
            for t_dunc, z_dunc in pool.imap_unordered(worker_duncan, range(n), chunksize=1):
                count_dunc += 1
                z_dunc_vals = z_dunc[:, 0] if z_dunc.ndim > 1 else z_dunc
                
                for j, tj in enumerate(t_range):
                    idx = max(0, np.searchsorted(t_dunc, tj, side='right') - 1)
                    sum_duncan[j] += z_dunc_vals[idx]
                
                if count_dunc % 100 == 0 or count_dunc == n:
                    print(f"   [Duncan]  {count_dunc}/{n} Simulationen beendet ({count_dunc/n*100:.1f}%)", flush=True)
                    
        exp_comparison_duncan[k, :] = sum_duncan / n

    # --- PLOTTING UND ERGEBNISSICHERUNG ---
    print("\nErstelle und speichere Visualisierungen...", flush=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    colors = sns.color_palette("viridis", n_colors=len(parameter_values))

    # Plot 1: Rein Alfonsi
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    for k, para in enumerate(parameter_values):
        ax1.plot(t_range, exp_comparison_alfonsi[k, :], marker='o', color=colors[k], 
                 linewidth=2, label=f"Threshold = {para}")
    ax1.set_xlabel("Time $t$", fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel("Expectation Value $\mathbb{E}[X(t)]$", fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title("Alfonsi: Expectation Value Dynamics over Time", fontsize=12, fontweight='bold', pad=12)
    ax1.legend(title="Parameter", frameon=True, facecolor='white', loc='best')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig("/work/scheibe5/bacteriophage_t7/expectation_comparison_alfonsi.png", dpi=300)
    plt.close(fig1)

    # Plot 2: Rein Duncan
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
    for k, para in enumerate(parameter_values):
        ax2.plot(t_range, exp_comparison_duncan[k, :], marker='s', linestyle='--', color=colors[k], 
                 linewidth=2, label=f"Threshold = {para}")
    ax2.set_xlabel("Time $t$", fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel("Expectation Value $\mathbb{E}[X(t)]$", fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title("Duncan-SSA: Expectation Value Dynamics over Time", fontsize=12, fontweight='bold', pad=12)
    ax2.legend(title="Parameter", frameon=True, facecolor='white', loc='best')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig("/work/scheibe5/bacteriophage_t7/expectation_comparison_duncan.png", dpi=300)
    plt.close(fig2)

    print("Job erfolgreich beendet. Bilddateien 'expectation_alfonsi.png' und 'expectation_duncan.png' gespeichert.", flush=True)