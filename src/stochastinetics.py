from math import comb as binom
import numpy as np
from scipy.integrate import solve_ivp
from numpy.random import choice
import csv

# Help-functions

def propensity(reaction,state,c,S_ed, is_continous=False):
        """
        Calculates the reaction propensity for general reaction and system State,
        input is:
        - the index of the reaction (column number of S)
        - state vector containing all the molecules numbers
        - vector of volume dependent reaction constants c
        - an educt matrix S_ed (rows: molecules; columns: reactions), where the
          educts are marked as abs(stochiometry), this is nessecary since reactions
          like A+B --> 2*A would lead to A not being detected as an educt when only
          using the stochiometric matrix!!
        """
        state = np.maximum(state,0.0)
        prop = c[reaction]
        educts = np.where(S_ed[:,reaction]> 0)[0]
        overall_educt_order = np.sum(S_ed[:,reaction])

        if overall_educt_order == 0:
            return prop
        
        elif overall_educt_order == 1:
            return prop*state[educts[0]]
        
        elif overall_educt_order == 2 and len(educts) ==1:
           if is_continous:
               return prop * (state[educts[0]]**2)/ 2.0 
           else:
            return max(0.0, prop * state[educts[0]] * (state[educts[0]] - 1) / 2)
        
        elif overall_educt_order ==2 and len(educts) ==2:
            return prop*state[educts[0]]*state[educts[1]]
        
        else:
            for educt_idx in educts:
                stochio = S_ed[educt_idx, reaction]
                if is_continous:
                    # Kontinuierliche Approximation (z.B. A^3 / 3!)
                    prop *= (state[educt_idx]**stochio) / np.math.factorial(int(stochio))
                else:
                    if state[educt_idx] < stochio:
                        return 0.0
                    prop *= binom(round(state[educt_idx]), int(stochio))
            return prop

def propensity_vec(reaction,state,c,S_ed,is_continous=False):
        if type(reaction) is int:
            return propensity(reaction,state,c,S_ed, is_continous)
        elif type(reaction) is np.ndarray:
            return np.array([propensity(i,state,c,S_ed,is_continous) for i in reaction])
        else:
            raise Exception(f"Dafuq? {type(reaction)} ist kein int oder np.ndarray?")
        
def blendings(state,S,boundaries):
        """
        Calculates the blending vlaues for each reaction at a given state and boundarie set.
            - 'Boundaries' is a 2*m vector (m being the number of molecule species) containing 
            the blending region boundaries for each molecule. 
        """
        blendings = np.zeros(S.shape[1])
        for reaction in range(S.shape[1]):
            betas = []
            for j in np.where(S[:, reaction] != 0)[0]:
                x = state[j]
                b = boundaries[2*j:2*j+2]
                if x >= b[1]:
                    betas.append(0)
                elif b[0] < x < b[1]:
                    betas.append((b[1] - x) / (b[1] - b[0]))
                else:
                    betas.append(1)
            blendings[reaction] = 1.0 - np.prod(1 - np.array(betas))
        return blendings

def generalised_blendings(state, c, S, S_ed, boundaries, double_prime):
        """
        Calculates the generalised blending values lambda' and lambda'' defined in the
           Duncan et. al paper for a given state. 
           - 'double_prime' is a boolean, if True: lambda double prime = (1-beta_j)*lambdaj
              is calculated instead
        """
        blending = blendings(state,S,boundaries)
        if double_prime:
            propensities = np.array([propensity(i, state, c, S_ed) for i in range(S.shape[1])])
            return (1-blending) * propensities
        else:
            # Duncan et al. recommendation: evaluate discrete propensities on integer values
            discrete_state = np.round(state) 
            propensities = np.array([propensity(i, discrete_state, c, S_ed) for i in range(S.shape[1])])
            return blending * propensities

def cle(x0,c,delta_t,S,S_ed,boundaries=None):
        """
        Gives back the next state X_(n+1) after delta_t when simulating the CLE using the 
        weak trapezoidal method presented in Duncan et. al.
        If 'boundaries' are provided, it uses blended propensities for the hybrid method.
        If 'boundaries' is None, it uses standard propensities for a pure CLE simulation.
        """
        xi = np.random.normal(0,1,(2,S.shape[1]))

        if boundaries is not None:
            lambda_double = generalised_blendings(x0,c,S,S_ed,boundaries,True)
        else: 
            lambda_double = np.array([propensity(i, x0, c, S_ed) for i in range(S.shape[1])])

        x_star = x0 + 0.5*delta_t* (S @ lambda_double) + (delta_t/2)**0.5*(S @ (np.sqrt(np.maximum(lambda_double, 0))*xi[0]))
        
        if boundaries is not None:
            lambda_double2 = generalised_blendings(x_star,c,S,S_ed,boundaries,True)
        else: 
            lambda_double2 = np.array([propensity(i, x_star, c, S_ed) for i in range(S.shape[1])])

        h = 2*lambda_double2 - lambda_double
        x_n = x_star + 0.5*delta_t*(S @ h) + (delta_t/2)**0.5 * (S @ (xi[1]*np.sqrt(np.maximum(h,0))))
        return x_n

def cle_trajectory(S, S_ed, c, t0, x0, tf, Delta_t):
    """
    Trajectory calculation by using the pure Chemical Langevin Equation (CLE) method.
    """
    x = [x0]
    t = [t0]
    while t[-1] < tf:
        x_new = np.maximum(0.0, cle(x[-1], c, Delta_t, S, S_ed, boundaries=None))
        # reflective boundary condition for CLE
        #x_new = np.absolute(cle(x[-1],c,Delta_t,S,S_ed,boundaries=None))
        x.append(x_new)
        t.append(t[-1] + Delta_t)
    return [np.array(t),np.array(x)]

def probability(result,t_range,state,n):
    """ 
    Probability of being in 'state' for every timepoint in t_range, approximated by its relative
    frequency compared to the n simulation runs
    """
    x = [result[i][1] for i in range(n)]
    t = [result[i][0] for i in range(n)]
    count = np.zeros(len(t_range))
    for j,ti in enumerate(t_range):
        for i in range(len(x)):
            if np.array_equal(x[i][np.where(t[i] <= ti)[0][-1]], state):
                count[j] +=1
    return count/n

def expectation(result,n,t_range):
    """
    Expectation value for the molecule number over time. Result is the list-output
    from SSA/Hybrid-algorithms, n the number of simulation iterations 
    """
    is_multispecies = result[0][1].ndim > 1
    expectation = np.zeros((len(t_range),result[0][1].shape[1])) if is_multispecies else np.zeros(len(t_range))
    for j,tj in enumerate(t_range):
        if is_multispecies: 
            current_vals = np.zeros((n,result[0][1].shape[1]))
        else:
            current_vals = np.zeros(n)
        for i in range(n):
            t_current = result[i][0]
            index = np.where(t_current <= tj)[0][-1]
            current_vals[i] = result[i][1][index]
        current_vals = np.rint(current_vals)
        expectation[j] = np.mean(current_vals,axis=0)
    return expectation

def expectation2(path, n_lines_per_run, t_range):
    """
    Berechnet den Erwartungswert der Molekülanzahl über t_range direkt aus der CSV-Datei,
    ohne die gesamten Simulationsergebnisse auf einmal im RAM zu halten.
    
    Parameters:
    -----------
    path : str
        Pfad zur CSV-Datei.
    n_lines_per_run : int
        Anzahl der Zeilen pro Simulations-Run (1 Zeitzeile + Anzahl der Spezies).
    t_range : array-like
        Die auszuwertenden Zeitpunkte.
    """
    t_range = np.asarray(t_range)
    len_t = len(t_range)
    n_species = n_lines_per_run - 1

    if n_species > 1:
        running_sum = np.zeros((len_t, n_species), dtype=np.float64)
    else:
        running_sum = np.zeros(len_t, dtype=np.float64)
        
    num_successful_runs = 0
    
    with open(path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        current_run_lines = []
        
        for i, row in enumerate(reader):
            if not row:
                continue
            
            try:
                numeric_data = np.array([float(x) for x in row])
            except ValueError:
                print(f"Warning: Couldn't convert row {i+1} into numeric data. Skipping line.")
                continue
                
            current_run_lines.append(numeric_data)
            if len(current_run_lines) == n_lines_per_run:
                time_array = current_run_lines[0]
                
                if n_species > 1:
                    state_matrix = np.column_stack(current_run_lines[1:])
                else:
                    state_matrix = current_run_lines[1]
                indices = np.searchsorted(time_array, t_range, side='right') - 1
                indices = np.maximum(indices, 0)

                current_vals = np.rint(state_matrix[indices])
                running_sum += current_vals
                num_successful_runs += 1
                if num_successful_runs % 10 == 0:
                    print(f"\rVerarbeitete Runs: {num_successful_runs}", end="", flush=True)
                current_run_lines = []

                
    if num_successful_runs == 0:
        raise ValueError("Es wurden keine gültigen Simulationsruns in der Datei gefunden.")
        
    print(f"{num_successful_runs} Simulation-Runs erfolgreich gestreamt und ausgewertet.")
    return running_sum / num_successful_runs

def get_distribution(result,n, ti, type="SSA"):
    """
    Calculates the probability distribution of states at a specific timepoint ti
    using linear interpolation (for the Alfonsi algorithim only) between recorded simulation steps to avoid 
    zero-order hold (staircase) bias.

    For this function to be more general, type = "Hybrid" has to be explicitly given
    when evaluating the Gillespie simulation algorithm result!
    """

    is_multispecies = result[0][1].ndim > 1
    current_vals = np.zeros((n,result[0][1].shape[1])) if result[0][1].ndim > 1 else np.zeros(n)
    
    for j in range(n):
        t_current = result[j][0]  
        index = np.where(t_current <= ti)[0][-1]

        if type == "SSA":
            current_vals[j] = result[j][1][index]
        elif type == "Hybrid":
            if index +1 < len(t_current):
                t0, t1 = t_current[index], t_current[index+1]
                x0, x1 = result[j][1][index], result[j][1][index+1]
                weight = (ti - t0) / (t1 - t0)
                x_interp = x0 + (x1 - x0) * weight
                current_vals[j] = x_interp
            else:
                current_vals[j] = result[j][1][index]
    current_vals = np.rint(current_vals)
    axis = 0 if is_multispecies else None
    unique_states, counts = np.unique(current_vals, axis=axis, return_counts=True)
    return unique_states, counts / n

def save_sim_results(result,path):
    """Save the result-list of a typical reaction kinetics simulation.
    --> assume that each entry of result is a list itself for one simulation run
    that stores the array of timepoints and system-states or seperated arrays for each
    molecule number. Output is a .csv file where for each simulation run the time points
    are saved as a row followed by a row for each molecule.
    """
    import csv
    n = len(result)
    k = len(result[0])
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if k ==2 and type(result[0][1]) is np.ndarray:
            k2 = result[0][1].shape[1]
            for i in range(n):
                writer.writerow(result[i][0])
                for j in range(k2):
                    writer.writerow(result[i][1][:,j])
        else:
            for i in range(n):
                for j in range(k):   
                    writer.writerow(result[i][j])


def read_in_sim_results(path, n):
    """Read in the result-list saved by save_sim_results. Output
    list has every specimen seperated. n needs to be the length of
    our system state +1 (for the timepoints).
    """
    import csv
    reconstructed_result = []
    with open(path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        current_run_data = []
        for i, row in enumerate(reader):
            if not row:
                continue
            try:
                numeric_data = np.array([float(x) for x in row])
            except ValueError:
                print(f"Warning: Couldn't convert row {i+1} into numeric data. Skipping this row: {row}.")
                continue
            current_run_data.append(numeric_data)
            if len(current_run_data) == n:
                time_array = current_run_data[0]
                state_matrix = np.column_stack(current_run_data[1:])
                reconstructed_result.append([time_array,state_matrix])
                current_run_data = [] 
        print(f"{len(reconstructed_result)} simulation runs have been reconstructed succesfully.")
    return reconstructed_result

def read_in_sim_results_gen(path,n):
    """Same function as read_in_sim_results but as a generator --> HPC problems doesnt allow for nodes
    with RAM > 4G right now...
    """
    import csv
    with open(path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        current_run_data = []
        counter = 0
        for i, row in enumerate(reader):
            if not row:
                continue
            try:
                numeric_data = np.array([float(x) for x in row])
            except ValueError:
                print(f"Warning: Couldn't convert row {i+1} into numeric data. Skipping.")
                continue
                
            current_run_data.append(numeric_data)

            if len(current_run_data) == n:
                time_array = current_run_data[0]
                state_matrix = np.column_stack(current_run_data[1:])
                yield [time_array, state_matrix]
                current_run_data = []
                counter += 1
                print(f"{counter} simulation runs have been streamed successfully.")


# Stochastical Simulation algorithm
def SSA(S,S_ed,c,t0,x0,tf):
    """Stochastic Simulation algorithm reaction kinetics.
    SSA(S,c,t0,X0,tf,PART) computes simulated reaction times t_S (a column vector)
    and the corresponding system state x_S (rows: species,columns: timepoints 
    --> each column is one state vector at the corresponding time) 
   for a stochastic reaction kinetic model specified by 
   - a stoichiometric matrix S (rows: species; columns: reactions)
   - an educt matrix S_ed (rows: molecules; columns: reactions), where the
     educts are marked as abs(stochiometry)
   - a volume dependent reaction constant vector c
   - a starting time of simulation t0
   - an initial number of molecules x0 at t0 (as a row vector)
   - a final time of simulation tf """        
    ts = [t0]
    xs = [np.array(x0)]
    while ts[-1] < tf:
        propensities = [propensity(reaction,xs[-1],c,S_ed,is_continous=False) for reaction in range(S.shape[1])]
        total = sum(propensities)
        if total > 0: 
            tau = np.random.exponential(1/total)
            probs = [p/total for p in propensities]
            probs[-1] = 1.0 - sum(probs[:-1])
            next_reaction = np.random.choice(list(range(S.shape[1])), p=probs)
            if tau + ts[-1] > tf:
                break
            ts.append(ts[-1] + tau)
            xs.append(xs[-1]+S[:,next_reaction])
        else:
            # for the possible zero state (propensity == 0) skip to
            # the end for the simulation
            ts.append(tf)
            xs.append(xs[-1])
            break
    return [np.array(ts),np.array(xs)]

# Hybrid-algorithm Alfonsi
def hybrid(S,S_ed,c,t0,x0,tf,part):
    """ HYBRID Hybrid stochastic/deterministic algorithm by Alfonsi et al (2005).
   HYBRID(S,PROP,T0,X0,TF,PART) computes simulated reaction times t_S (a column vector)
    and the corresponding system state xs (rows: species,columns: timepoints 
    --> each column is one state vector at the corresponding time)
   for a mixed stochastic/deterministic reaction kinetic model specified by 
   - a stoichiometric matrix S (rows: species; columns: reactions)
   - an educt matrix S_ed (rows: molecules; columns: reactions), where the
     educts are marked as abs(stochiometry)
   - a volume dependent reaction constant vector c
   - a starting time of simulation to 
   - an initial number of molecules x0 at t0 (as a row vector)
   - a final time of simulation tf
   - a fixed partitioning part of reactions into stochastic or
     deterministic (a logical vector the same size a the number of
     reactions, with 'true' meaning stochastic and 'false' deterministic OR
     a float value threshold that specifies an adaptive scheme in which reactions are 
     modelled stochastically whenever their propensities are smaller than threshold"""

            
    def ode(t,x,S,S_ed,c,part):
        """Deterministic reaction kinects between stochastical events. Additionally 
           the integral of the total propensity function until the current time point is
           stored in the last entry of the output vector.
            """
        S2 = S[:,~part]
        reactions_D = np.where(~part)[0]
        reactions_S = np.where(part)[0]
        alpha_D = propensity_vec(reactions_D,x[:-1],c,S_ed,is_continous=True)
        alpha_D = alpha_D[:, np.newaxis]
        alpha_S = propensity_vec(reactions_S,x[:-1],c,S_ed, is_continous=True)
        return np.append(np.transpose(S2 @ alpha_D)[0],np.sum(alpha_S)) # wichtig damit g(t) auf alpha0(x(t)) zugreifen kann!!


    def g(t,y,S,S_ed,c,part):
        """The "acummulated probability for the first reaction event" 
           defines the ending condition for the deterministic time frames
           in the classical hybrid algorithm. Threshold is the random number drawn
           from the exponential distribution.
           """
        return  y[-1]-xi # xi muss global genutzt werden damit die Argumente von ode und g für den solver übereinstimmen...
            
    g.terminal = True
    g.direction = 1
    ts = [t0]
    xs = [x0]
    n0 = 0
    while ts[-1] < tf:
        xi = np.random.exponential(1)
        if np.ndim(part) >= 1:
            partition = part
        elif np.ndim(part) == 0:
            """adaptive partitioning or threshold"""
            probs = np.array([propensity(reaction,xs[-1],c,S_ed) for reaction in range(S.shape[1])])
            partition = probs < part

        result = solve_ivp(ode,[0,tf-ts[-1]],np.append(xs[-1], 0),args=(S, S_ed, c, partition), events=g, method='BDF')
        if len(result.t) > 1:
            ts.extend(ts[-1] + result.t[1:])
            xs.extend(list(result.y[:-1, 1:].T))
        x_new = np.maximum(xs[-1], 0.0)
        if result.t_events[0].size > 0:
            """Did the solver reach the end before the next reaction event?"""
            propensities = [propensity(reaction,x_new,c,S_ed) for reaction in np.where(partition)[0]]
            total = sum(propensities)
        else:
            break
            
        if total > 0: 
            """Are we in a steady Null state xs[-1] = [0,0,0,0....]?"""
            probs = [p/total for p in propensities]
            probs[-1] = max(0.0, 1.0 - sum(probs[:-1]))
            next_reaction = np.random.choice(list(np.where(partition)[0]), p=probs)
            ts.append(ts[-1])
            xs.append(x_new + S[:,next_reaction])
            n0 +=1
        else:
            ts.append(tf)
            xs.append(xs[-1])
            break
    return [np.array(ts),np.array(xs),n0]
    

def hybrid_duncan_ssa(S,S_ed,c,t0,x0,tf,delta_t,Delta_t,boundaries):
    """ HYBRID Hybrid stochastic/deterministic algorithm by Duncan et al (2016).
   HYBRID_duncan_ssa(S,S_ed,c,t0,x0,tf,delta_t,Delta_t,boundaries) computes simulated reaction times t_S
    and the corresponding system state xs (rows: species,columns: timepoints) --> each column 
    is one state vector at the corresponding time.
    The algorithm combines the purely stochastical SSA-approach with the diffusive CLE approximation
    for a reaction kinetics system specified by:
     
   - a stoichiometric matrix S (rows: species; columns: reactions)
   - an educt matrix S_ed (rows: molecules; columns: reactions), where the
     educts are marked as abs(stochiometry)
   - a volume dependent reaction constant vector c
   - a starting time of simulation to 
   - an initial number of molecules x0 at t0 (as a row vector)
   - a final time of simulation tf
   - BIGGER time-step for the pure CLE regime delta_t
   - SMALLER time-step/upper bound for the CLE-simulation in the hybrid regime
   - 2*m - blending region boundaries for all m-molecule species
   """    
    x = [x0]
    t = [t0]
    n0= 0
    while t[-1] < tf:
        betas = blendings(x[-1],S,boundaries)
        if max(betas) == 0:
            x_new = np.maximum(0.0, cle(x[-1],c,delta_t,S,S_ed,boundaries))
            x_new[x_new < 0.5] = 0.0 
            x.append(x_new)
            t.append(t[-1] + delta_t)
        elif min(betas) == 1:
            lambda_prime = generalised_blendings(x[-1],c,S,S_ed,boundaries,False) 
            lambda_0 = np.sum(lambda_prime)
            if lambda_0 == 0:
                t.append(tf)
                x.append(x[-1])
                break
            tau = -np.log(np.random.uniform(0,1))/lambda_0
            dist = lambda_prime/lambda_0
            dist[-1] = max(0.0,1.0- np.sum(dist[:-1]))
            next_reaction = np.random.choice(np.arange(S.shape[1]),p=np.maximum(0,dist))
            x_new = np.maximum(0.0, x[-1] + S[:,next_reaction])
            x_new[x_new < 0.5] = 0.0
            x.append(x_new)
            t.append(t[-1] + tau)
            n0 +=1
        else:
            lambda_prime = generalised_blendings(x[-1],c,S,S_ed,boundaries,False)
            lambda_0 = np.sum(lambda_prime)
            if lambda_0 == 0:
                # all stochastic reactions have propensity = 0 --> next tau is infinite!
                tau = float('inf')
            else:
                tau = -np.log(np.random.uniform(0,1))/lambda_0
                dist = lambda_prime/lambda_0
                dist[-1] = max(0.0,1.0- np.sum(dist[:-1]))
                next_reaction = np.random.choice(np.arange(S.shape[1]),p=np.maximum(0,dist))

            if tau < Delta_t:
                x2 = np.maximum(0.0,cle(x[-1],c,tau,S,S_ed,boundaries))
                x_new = np.maximum(0.0, x2 + S[:,next_reaction].flatten()) 
                x_new[x_new < 0.5] = 0.0
                x.append(x_new)
                t.append(t[-1] + tau)
                n0 +=1
            else:
                x_new = np.maximum(0.0, cle(x[-1],c,Delta_t,S,S_ed,boundaries))
                x_new[x_new < 0.5] = 0.0
                x.append(x_new)
                t.append(t[-1] + Delta_t)
    return [np.array(t),np.array(x),n0]


def hybrid_duncan_next_reaction(S,S_ed,c,t0,x0,tf,delta_t,Delta_t,boundaries):
    """ HYBRID Hybrid stochastic/deterministic algorithm by Duncan et al (2016).
   HYBRID_duncan_next_reaction(S,S_ed,c,t0,x0,tf,delta_t,Delta_t,boundaries) computes simulated 
   reaction times t_S and the corresponding system state xs (rows: species,columns: timepoints) 
    --> each column is one state vector at the corresponding time.
    The algorithm combines the purely stochastical next-reaction-method with the diffusive CLE 
    approximation for a reaction kinetics system specified by:
     
   - a stoichiometric matrix S (rows: species; columns: reactions)
   - an educt matrix S_ed (rows: molecules; columns: reactions), where the
     educts are marked as abs(stochiometry)
   - a volume dependent reaction constant vector c
   - a starting time of simulation to 
   - an initial number of molecules x0 at t0 (as a row vector)
   - a final time of simulation tf
   - BIGGER time-step for the pure CLE regime delta_t
   - SMALLER time-step/upper bound for the CLE-simulation in the hybrid regime
   - 2*m - blending region boundaries for all m-molecule species
   """
    x = [x0]
    t = [t0]
    u = np.random.uniform(0,1,S.shape[1])
    F = -np.log(u)
    T = np.zeros(len(F))
    lambdas = generalised_blendings(x0,c,S,S_ed,boundaries,False)
    
    while t[-1] < tf:
        current_x = x[-1]
        betas = blendings(current_x, S, boundaries)
        
        # Recalculate lambdas for current state
        lambdas = generalised_blendings(current_x, c, S, S_ed, boundaries, False)

        if np.max(betas) == 0: # Pure CLE regime
            x_new = cle(current_x, c, delta_t, S, S_ed, boundaries)
            x.append(np.maximum(0.0, x_new))
            t.append(t[-1] + delta_t)
            # Note: The T vector is not updated here, which might be an issue if the system
            # can re-enter the hybrid regime. The paper is not explicit about this.
            continue

        # Hybrid or pure stochastic regime
        # To avoid division by zero for reactions with 0 propensity
        tau = np.full(S.shape[1], np.inf)
        non_zero_lambdas = lambdas > 0
        tau[non_zero_lambdas] = (F[non_zero_lambdas] - T[non_zero_lambdas]) / lambdas[non_zero_lambdas]

        r = np.argmin(tau)
        step = tau[r]

        if step == np.inf: # No reaction can fire, evolve with CLE until end
            t.append(tf)
            x.append(current_x)
            break

        if np.min(betas) == 1: # Pure stochastic regime (but using the hybrid framework)
            t_new = t[-1] + step
            x_new = current_x + S[:, r]
            T += lambdas * step
            F[r] -= np.log(np.random.uniform(0, 1))
        else: # Hybrid regime
            if step < Delta_t:
                x_star = cle(current_x, c, step, S, S_ed, boundaries)
                t_new = t[-1] + step
                x_new = x_star + S[:, r]
                T += lambdas * step
                F[r] -= np.log(np.random.uniform(0, 1))
            else: # step >= Delta_t
                t_new = t[-1] + Delta_t
                x_new = cle(current_x, c, Delta_t, S, S_ed, boundaries)
                T += lambdas * Delta_t
        
        x.append(np.maximum(0.0, x_new))
        t.append(t_new)

    return [np.array(t),np.array(x)]
    

def hybrid_duncan_thinning(S,S_ed,c,t0,x0,tf,delta_t,Delta_t,boundaries,lambda_bounds):
    """ HYBRID Hybrid stochastic/deterministic algorithm by Duncan et al (2016).
   HYBRID_duncan_thinning(S,S_ed,c,t0,x0,tf,delta_t,Delta_t,boundaries,lambda_bounds) computes simulated reaction times t_S
    and the corresponding system state xs (rows: species,columns: timepoints) --> each column 
    is one state vector at the corresponding time.
    The algorithm combines the purely stochastical SSA-approach(modified by the standard thinning method 
    for inhomogeneous poisson processes) with the diffusive CLE approximationfor a reaction kinetics
    system specified by:
     
   - a stoichiometric matrix S (rows: species; columns: reactions)
   - an educt matrix S_ed (rows: molecules; columns: reactions), where the
     educts are marked as abs(stochiometry)
   - a volume dependent reaction constant vector c
   - a starting time of simulation to 
   - an initial number of molecules x0 at t0 (as a row vector)
   - a final time of simulation tf
   - BIGGER time-step for the pure CLE regime delta_t
   - SMALLER time-step/upper bound for the CLE-simulation in the hybrid regime
   - 2*m - blending region boundaries for all m-molecule species
   - upper bounds for the possible lambda_prime (as required by the thinning method) as a reaction vector
   """
    x = [x0]
    t = [t0]
    while t[-1] < tf:
        current_x = x[-1]
        betas = blendings(current_x, S, boundaries)
        if np.max(betas) == 0:
            x_new = np.maximum(0.0, cle(x[-1],c,delta_t,S,S_ed,boundaries))
            x_new[x_new < 0.5] = 0.0 
            x.append(x_new) 
            t.append(t[-1] + delta_t)
        elif np.min(betas) == 1: 
            lambda_prime = generalised_blendings(current_x, c, S, S_ed, boundaries, False)
            lambda_0 = np.sum(lambda_prime)
            if lambda_0 == 0:
                t.append(tf)
                x.append(current_x)
                break
            else:
                tau = -np.log(np.random.uniform(0,1)) / lambda_0
                dist = lambda_prime / lambda_0
                dist[-1] = max(0.0, 1.0 - np.sum(dist[:-1]))
                next_reaction = np.random.choice(np.arange(S.shape[1]), p=np.maximum(0, dist))
                x.append(current_x + S[:, next_reaction])
                t.append(t[-1] + tau)
        else: 
            Lambda_0 = np.sum(lambda_bounds)
            u = np.random.uniform(0,1,2)
            tau = -np.log(u[0])/Lambda_0
            
            x_evolved = current_x
            t_var = 0
            while t_var < tau:
                current_dt = min(Delta_t, tau - t_var)
                x_evolved = cle(x_evolved, c, current_dt, S, S_ed, boundaries)
                t_var += current_dt 

            lambda_prime = generalised_blendings(x_evolved, c, S, S_ed, boundaries, False)
            if Lambda_0*u[1] < np.sum(lambda_prime):
                r = np.searchsorted(np.cumsum(lambda_prime), Lambda_0 * u[1])
                x_evolved[x_evolved < 0.5] = 0.0 
                x.append(np.maximum(0.0, x_evolved + S[:,r]))
            else: 
                x.append(np.maximum(0.0, x_evolved))
            t.append(t[-1] + tau)
            
    return [np.array(t),np.array(x)]