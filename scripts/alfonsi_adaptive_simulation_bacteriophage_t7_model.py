import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.integrate import solve_ivp
modul_path = "/home/scheibe5"
if modul_path not in sys.path:
    sys.path.append(modul_path)
import stochastinetics
from tqdm import tqdm
import time
import multiprocess

k1 = 0.025
k2 = 0.25
k3 = 1.0
k4 = 7.5*10**(-6)
k5 = 1000
k6 = 1.99
tem0 = 1
gen0 = 0
struc0 = 0
Delta_t = 10**(-3)
delta_t = 10**(-2)
I1 = 25.0
I2 = 35.0
boundaries = np.array([I1,I2,I1,I2,I1,I2])

S = np.array([[-1,0,1,-1,0,0],[1,-1,0,0,0,0],[0,0,0,-1,1,-1]]) # order [gen,tem,struc]
S_ed = np.array([[1,0,0,1,0,0],[0,1,1,0,1,0],[0,0,0,1,0,1]],dtype=int)
c = np.array([k1,k2,k3,k4,k5,k6]) # assuming this are already the volume dependent constants...
x0 = np.array([gen0,tem0,struc0])
t0 = 0
tf = 200
threshold = 10
n = 1000
n1=10000

"""Deterministic simulation"""
def rate_of_change(t,x,S,c):
    gen_dot = -c[0]*x[0] + c[2]*x[1] - c[3]*x[0]*x[2]
    tem_dot = c[0]*x[0]-c[1]*x[1]
    struc_dot = -c[3]*x[0]*x[2] + c[4]*x[1] -c[5]*x[2]
    return gen_dot,tem_dot,struc_dot

def init_worker():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())

def run_single_alfonsi(i, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=tf):
    t, z = stochastinetics.hybrid(S, S_ed, c, t0, x0, tf,threshold)
    return t, z


if __name__ == "__main__":
    t_eval = np.linspace(t0, tf, n1)
    bacteriophage = solve_ivp(rate_of_change, (t0, tf), x0, t_eval=t_eval, args=(S, c))
    tem = bacteriophage.y[1]
    
    result_alfonsi = []
    start = time.time()
    num_cpus = len(os.sched_getaffinity(0))

    print(f"Starte Simulation auf {num_cpus} CPUs... (Gesamtanzahl: {n})", flush=True)
    with multiprocess.Pool(processes=num_cpus, initializer=init_worker) as pool:
        for i,(t, z) in enumerate(pool.imap_unordered(run_single_alfonsi, range(n), chunksize=1), 1):
            result_alfonsi.append([t, z])
            if i % 10 == 0 or i == n:
                print(f"[PROGRESS] {i}/{n} Simulationen beendet ({i/n*100:.1f}%)", flush=True)
    end = time.time()
    print(f"Time elapsed: {end - start:.2f} seconds.")

    path = os.path.join("/work/scheibe5/bacteriophage_t7/alfonsi_adaptive",f"result_alfonsi_adaptive_10_bacteriophage.csv")
    stochastinetics.save_sim_results(result_alfonsi,path)
    t_range = np.arange(0, 200, 2)
    exp_alfonsi = stochastinetics.expectation(result_alfonsi, n, t_range)[:,1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bacteriophage.t, tem, label="deterministic")
    ax.plot(t_range,exp_alfonsi,label="alfonsi adaptive partitioning")
    ax.legend()
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Number of tem molecules")
    ax.grid(True, linestyle=':', color='gray', alpha=0.6)
    path2 = os.path.join("/work/scheibe5/bacteriophage_t7/alfonsi_adaptive","bacteriophage_model_alfonsi_adaptive_10_and_deterministic.png")
    plt.savefig(path2)
    plt.close(fig)
