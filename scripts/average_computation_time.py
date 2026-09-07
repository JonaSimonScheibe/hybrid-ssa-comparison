# Average Computation Time over Simulation Duration
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.integrate import solve_ivp
modul_path = "/work/scheibe5/bacteriophage_t7/"
if modul_path not in sys.path:
    sys.path.append(modul_path)
import stochastinetics
from tqdm import tqdm
import time
import multiprocess
from functools import partial

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
I1 = 15.0
I2 = 25.0
boundaries = np.array([I1,I2,I1,I2,I1,I2])

S = np.array([[-1,0,1,-1,0,0],[1,-1,0,0,0,0],[0,0,0,-1,1,-1]]) # order [gen,tem,struc]
S_ed = np.array([[1,0,0,1,0,0],[0,1,1,0,1,0],[0,0,0,1,0,1]],dtype=int)
c = np.array([k1,k2,k3,k4,k5,k6])
x0 = np.array([gen0,tem0,struc0])

t0 = 0
n = 1000
#t_range = [1,10,50,100,150,200]
t_range = [150] 
comp_times_ssa = []
comp_times_cle = []
comp_times_duncan_02_12 = []
comp_times_duncan_10_90 = []
comp_times_duncan_80_90 = []
comp_times_duncan_5_15 = []
comp_times_duncan_10_20 = []
comp_times_duncan_15_25 = []
comp_times_alfonsi = []
threshold = 500


def init_worker():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())

def run_single_alfonsi(i, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=1,threshold = threshold):
    start = time.time()
    t, z = stochastinetics.hybrid(S, S_ed, c, t0, x0, tf,threshold)
    end = time.time()
    return end-start

def run_single_cle(i, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=1,Delta_t = Delta_t):
    start = time.time()
    t, z = stochastinetics.cle_trajectory(S, S_ed, c, t0, x0, tf,Delta_t)
    end = time.time()
    return end-start


def run_single_hybrid_duncan(i, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=1,delta_t = delta_t,Delta_t = Delta_t,boundaries=boundaries):
    start = time.time()
    t, z = stochastinetics.hybrid_duncan_ssa(S, S_ed, c, t0, x0, tf,delta_t,Delta_t,boundaries)
    end = time.time()
    return end-start

def run_single_ssa(i, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=1):
    start = time.time()
    t, z = stochastinetics.SSA(S, S_ed, c, t0, x0, tf)
    end = time.time()
    return end-start



if __name__ == "__main__":
    num_cpus = len(os.sched_getaffinity(0))
    print(f"Starte Simulation auf {num_cpus} CPUs... (Gesamtanzahl: {n})", flush=True)
    time_measures = np.zeros((n,len(t_range)))
    for j,sim_time in enumerate(t_range):
        print(f"\n--- Simulation Duration: {sim_time}", flush=True)
        worker_fn = partial(run_single_ssa, S=S, S_ed=S_ed, c=c, t0=t0, x0=x0, tf=sim_time)
        current_run_times = np.zeros(n)
        with multiprocess.Pool(processes=num_cpus, initializer=init_worker) as pool:
            for i,timing in enumerate(pool.imap_unordered(worker_fn, range(n), chunksize=1)):
                current_run_times[i] = timing
                if i % 10 == 0 or i == n:
                    print(f"[PROGRESS] {i}/{n} Simulationen beendet ({i/n*100:.1f}%)", flush=True)
                if i %100 ==0:
                    print(f"Current mean: {np.mean(current_run_times[:i+1])}")
        mean_time = np.mean(current_run_times)
        comp_times_ssa.append(mean_time)
        print(f"Durchschnittliche Rechenzeit für tf={sim_time}s (ssa): {mean_time:.6f}s", flush=True)

    # #    --- Hybrid-Duncan ---

    #     #    --- 5,15 ---
    #     boundaries = np.array([5.0,15.0,5.0,15.0])
    #     start = time.time()
    #     print(f"Current simulation time: {sim_time}s, Current algorithm: Hybrid-Duncan")
    #     for i in tqdm(range(n)):
    #         t,z = stochastinetics.hybrid_duncan_ssa(S,S_ed,c,t0,x0,sim_time,delta_t,Delta_t,boundaries)
    #     end = time.time()
    #     comp_times_duncan_5_15.append((end-start)/n)

    #         # --- 10,20 ---
    #     boundaries = np.array([10.0,20.0,10.0,20.0])
    #     start = time.time()
    #     print(f"Current simulation time: {sim_time}s, Current algorithm: Hybrid-Duncan")
    #     for i in tqdm(range(n)):
    #         t,z = stochastinetics.hybrid_duncan_ssa(S,S_ed,c,t0,x0,sim_time,delta_t,Delta_t,boundaries)
    #     end = time.time()
    #     comp_times_duncan_10_20.append((end-start)/n)

    #         # --- 15,25 --- 
    #     boundaries = np.array([15.0,25.0,15.0,25.0])
    #     start = time.time()
    #     print(f"Current simulation time: {sim_time}s, Current algorithm: Hybrid-Duncan")
    #     for i in tqdm(range(n)):
    #         t,z = stochastinetics.hybrid_duncan_ssa(S,S_ed,c,t0,x0,sim_time,delta_t,Delta_t,boundaries)
    #     end = time.time()
    #     comp_times_duncan_15_25.append((end-start)/n)

    #         # --- 80,90 --- 
    #     boundaries = np.array([80.0,90.0,80.0,90.0])
    #     start = time.time()
    #     print(f"Current simulation time: {sim_time}s, Current algorithm: Hybrid-Duncan")
    #     for i in tqdm(range(n)):
    #         t,z = stochastinetics.hybrid_duncan_ssa(S,S_ed,c,t0,x0,sim_time,delta_t,Delta_t,boundaries)
    #     end = time.time()
    #     comp_times_duncan_80_90.append((end-start)/n)

    #         # --- 10,90 --- 
    #     boundaries = np.array([10.0,90.0,10.0,90.0])
    #     start = time.time()
    #     print(f"Current simulation time: {sim_time}s, Current algorithm: Hybrid-Duncan")
    #     for i in tqdm(range(n)):
    #         t,z = stochastinetics.hybrid_duncan_ssa(S,S_ed,c,t0,x0,sim_time,delta_t,Delta_t,boundaries)
    #     end = time.time()
    #     comp_times_duncan_10_90.append((end-start)/n)

    #         # --- 02,12 --- 
    #     boundaries = np.array([2.0,12.0,2.0,12.0])
    #     start = time.time()
    #     print(f"Current simulation time: {sim_time}s, Current algorithm: Hybrid-Duncan")
    #     for i in tqdm(range(n)):
    #         t,z = stochastinetics.hybrid_duncan_ssa(S,S_ed,c,t0,x0,sim_time,delta_t,Delta_t,boundaries)
    #     end = time.time()
    #     comp_times_duncan_02_12.append((end-start)/n)

    #     # --- Alfonsi ---
    #     start = time.time()
    #     print(f"Current simulation time: {sim_time}s, Current algorithm: Alfonsi")
    #     for i in tqdm(range(n)):
    #         t,z = stochastinetics.hybrid(S,S_ed,c,t0,x0,sim_time,threshold)
    #     end = time.time()
    #     comp_times_gillespie.append((end-start)/n)

    #     # --- SSA ---
    #     start = time.time()
    #     print(f"Current simulation time: {sim_time}s, Current algorithm: SSA")
    #     for i in tqdm(range(n)):
    #         t,z = stochastinetics.SSA(S,S_ed,c,t0,x0,sim_time)
    #     end = time.time()
    #     comp_times_ssa.append((end-start)/n)

    print(f"\nSSA-computation times: {comp_times_ssa}")
    # fig,ax = plt.subplots(figsize=(10,6))
    # #ax.plot(t_range,comp_times_ssa,color="blue", ls="--",label="SSA")
    # # ax.plot(t_range,comp_times_duncan_5_15,color="red", ls="--",label="Duncan (5,15)")
    # #ax.plot(t_range,comp_times_duncan_15_25,color="turquoise", ls="--",label="Duncan (10,20)")
    # # ax.plot(t_range,comp_times_duncan_15_25,color="purple", ls="--",label="Duncan (15,25)")
    # ax.plot(t_range,comp_times_alfonsi,color="orange", ls="--",label="Alfonsi")
    # #ax.plot(t_range,comp_times_cle,color="green", ls="--",label="CLE")
    # ax.set_yscale('log')
    # #ax.set_ylim(10**(-5), 10**1)
    # ax.set_xlabel("Simulation duration (s)")
    # ax.set_ylabel("Average computation time (s)")
    # ax.legend()
    # ax.set_title("Average Computation Time over Simulation Duration")
    # ax.grid(True, linestyle=':', color='gray', alpha=0.6)
    # path2 = os.path.join("/work/scheibe5/bacteriophage_t7/","bacteriophage_model_computation_time_alfonsi.png")
    # plt.savefig(path2)
    # plt.close(fig)