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


def time_tracker_alfonsi_duncan(i,S,S_ed,c,t0,x0,tf,threshold,delta_t,Delta_t):
    boundaries = np.array([threshold,2*threshold,threshold,2*threshold])
    _,_,n_alfonsi = stochastinetics.hybrid(S,S_ed,c,t0,x0,tf,threshold)
    _,_,n_duncan = stochastinetics.hybrid_duncan_ssa(S,S_ed,c,t0,x0,tf,delta_t,Delta_t,boundaries)
    return n_alfonsi,n_duncan

def init_worker():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())


if __name__ == "__main__":
    M = 10**3
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
    alfonsi_lambdas = [20,40,80,160,320,640,1280]
    mean_alfonsi, mean_duncan = np.zeros((M,len(alfonsi_lambdas))), np.zeros((M,len(alfonsi_lambdas)))

    start = time.time()
    num_cpus = len(os.sched_getaffinity(0))

    print(f"Starte Simulation auf {num_cpus} CPUs... (Gesamtanzahl: {M})", flush=True)

    for j,threshold in enumerate(alfonsi_lambdas):
        print(f"\n--- Propensity Threshold: {threshold}", flush=True)
        worker_fn2 = partial(time_tracker_alfonsi_duncan, delta_t=delta_t,Delta_t = Delta_t, S=S, S_ed=S_ed, c=c, x0=x0,t0 =t0,tf=tf,threshold=threshold)
        with multiprocess.Pool(processes=num_cpus, initializer=init_worker) as pool:
            for idx,res in enumerate(pool.imap_unordered(worker_fn2, range(M), chunksize=1)):
                n_alfonsi, n_duncan = res
                mean_alfonsi[idx,j] = n_alfonsi
                mean_duncan[idx,j] = n_duncan

                completed = idx+1
                if idx % 10 == 0 or completed == M:
                    print(f"[PROGRESS] {completed}/{M} Simulationen beendet ({completed/M*100:.1f}%)", flush=True)

    mean_duncan = np.mean(mean_duncan, axis=0)
    mean_alfonsi = np.mean(mean_alfonsi,axis=0)

    print(f"alfonsi_Zeitanteile (absolut-wert): {mean_alfonsi}")
    print(f"duncan_Zeitanteile (absolut-wert): {mean_duncan}")