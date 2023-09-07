######################
### SmartRunner.py ###
######################

from math import *
import numpy as np
import random
import networkx as nx


###################################
### Main SmartRunner functions: ###
###################################


# Computes p_f for unweighted move sets
def get_pf(n,N,mp):
    if n==0:
        pf = 1./2
    else:
        Ntilde = N - mp + 1
        gamma = n/N
        try:
            pf = (1./N) * (exp(-gamma) - Ntilde*exp(-gamma*Ntilde) + \
                        exp(-gamma*(Ntilde+1))*(Ntilde - 1))/((1-exp(-gamma))*(1-exp(-gamma*Ntilde)))
        except:
            pf = 0.0 # this should enable jumping out of the 'problematic' node
    
    return pf


# Computes p_f for weighted move sets (Gaussian mixture model)
def get_pf_Gmix(n,N,mp,p_k,w_means,w_sigmas,Z):
    assert len(p_k) == len(w_means) and len(p_k) == len(w_sigmas), "Error: dim mismatch in get_pf_Gmix!"

    P = len(p_k)
    sq = sqrt(2)
    sq2 = sqrt(2*np.pi)

    if n==0:
        pf = 1./2
    else:
        Ntilde = N - mp + 1
        W = 0.0
        for j in range(P):
            W += p_k[j] * w_means[j]
        W *= N
        ####
        try:
            exp_minus_beta = 0.0
            for j in range(P):
                if w_sigmas[j] < 0.1 * w_means[j]: # narrow Gaussian, treat as a delta-function
                    alpha = (n*w_means[j])/W + log(Z)
                else:
                    c_k = (w_sigmas[j]**2 * n)/W

                    fac_arg = (c_k-w_means[j])/(sq*w_sigmas[j])

                    if fac_arg > 20 or fac_arg < -20: # asymptotic
                        alpha = (n*w_means[j])/W + log(sq2*w_sigmas[j]*n*Z/W)
                    else: # exact
                        fac = 0.5*erfc(fac_arg)
                        if fac < 1e-50:
                            fac = 1e-50
                        alpha = (n*w_means[j])/W - (n**2 * w_sigmas[j]**2)/(2*W**2) - log(fac) + log(Z)
                ####
                if alpha < 50:
                    exp_minus_beta += p_k[j] * exp(-alpha)
                else:
                    exp_minus_beta += 0.0
            ####
            pf = (1./N) * (exp_minus_beta - Ntilde * exp_minus_beta**Ntilde + \
                        exp_minus_beta**(Ntilde+1) * (Ntilde - 1))/((1 - exp_minus_beta)*(1 - exp_minus_beta**Ntilde))
        except:
            pf = 0.0 # this should enable jumping out of the 'problematic' node
    
    return pf


# Computes p_f for exponentially weighted move sets
def get_pf_exp(n,N,mp):
    if n==0:
        pf = 1./2
    else:
        Ntilde = N - mp + 1
        a = N/(n+N)
        try:
            pf = (1./N) * (a - Ntilde * a**Ntilde + \
                        (Ntilde - 1) * a**(Ntilde+1))/((1 - a) * (1 - a**Ntilde))
        except:
            pf = 0.0 # this should enable jumping out of the 'problematic' node
    
    return pf


# Computes approximate p_f(n) when N and m_p are either known or not
def get_pf_approx(n,N=None,mp=None):
    ####
    if N != None and mp != None:
        if mp > N:
            raise Exception("Illegal value of m_p={} > N={} in get_pf_approx!".format(mp,N))
        elif mp==N:
            return 0.0

    ####
    if n >= 0 and n < 5:
        pf = (1./250) * n * n - (2./25) * n + (1./2)
    elif n >= 5:
        pf = 1./n
    else:
        raise Exception("Illegal value of n={} in get_pf_approx!".format(n))

    return pf


#########
#L_EXTRA_MAX = int(1e12)
L_EXTRA_MAX = int(1e10)
#########


# Computes l_extra
def get_l_extra(n,N,mp):
    pf = get_pf(n,N,mp)
    #pf = get_pf_approx(n)
    #pf = get_pf_approx(n,N,mp)

    if pf <= 1e-12:
        l_extra = L_EXTRA_MAX
    else:
        #l_extra = int(floor(1./pf))
        l_extra = int(round(1./pf))
    
    return l_extra


# Computes approximate l_extra
def get_l_extra_approx(n,N=None,mp=None):
    pf = get_pf_approx(n,N,mp)

    if pf <= 1e-12:
        l_extra = L_EXTRA_MAX
    else:
        #l_extra = int(floor(1./pf))
        l_extra = int(round(1./pf))
    
    return l_extra


# Recursive computation of path lengths: now with the number of steps (jumps) computed
def get_path_lengths(G, DF_rate_mean, node2length, node2steps, lmax):
    ####
    if lmax == 0: # max path length exceeded
        return
    
    eps = 1e-04
    node2length_cur = {}
    node2steps_cur = {}

    ####
    for x in node2length:
        for y in G.successors(x):
            ####
            w_cur = G.edges[x,y]['weight']
            if x[-1] != "t" and y[-1] == "t":
                w_cur *= DF_rate_mean
            ####
            if not y in node2length:
                node2length_cur[y] = node2length[x] + w_cur
                ####
                if x[-1] != "t" and y[-1] == "t":
                    node2steps_cur[y] = node2steps[x]
                elif x[-1] != "t" and y[-1] != "t":
                    node2steps_cur[y] = node2steps[x] + 1
                else:
                    raise Exception("Illegal path!")
            else:
                l1 = node2length[y]
                l2 = node2length[x] + w_cur
                if abs(l1 - l2) > eps: # these two paths must have the same weights
                    raise Exception("Path weight mismatch!")
    ####
    l_added = len(node2length_cur)

    if l_added > 0:
        node2length.update(node2length_cur)
        node2steps.update(node2steps_cur)

    ####
    lmax -= 1

    ####
    if l_added == 0: # no new nodes added
        return
    else:
        get_path_lengths(G, DF_rate_mean, node2length, node2steps, lmax)


# This function returns the end-point of a "sideways" path that takes the walker out of
# the prevously explored region.
def jump_sideways(G, crd_tuple, crds_final, max_steps):
    
    node_cur = crd_tuple
    
    while 1:
        slist = list(G.successors(node_cur))
        
        if len(slist) == 0: # this should not happen as this function must be called for well-explored neighborhoods
            return
        
        node_cur = random.choice(slist)
        
        if node_cur[-1] == "t":
            node_cur = node_cur[:-1]
        
        crds_final.append(node_cur)
        
        max_steps -= 1
        
        # Max path length exceeded or 'non L_EXTRA_MAX' node found:
        if max_steps == 0 or G.nodes[node_cur]['lx'] != L_EXTRA_MAX:
            return


#######################################
# Top-level function, SmartRunner (v.2)
#######################################
def SmartRunner(crd_init_tuple,N,ltot,ranges,Delta_x,DeltaF_rate_mean,optimism, \
                fitness_f,move_f,l_max,mode,moveset_mode,params,mparams,sopt=2,max_feval=None):
    
    # Hyperparameter:
    M = 250 # stats of previous moves (both deleterious/neutral and beneficial); can also do M=ltot/50 or smth. like that
    steps = np.linspace(1, M, num=M, dtype=np.int64)
    if ltot <= M:
        print("WARNING: insufficient number of steps to reset DeltaF_rate_mean!")
        print("The initial value of {} will be used throughout the simulation ..".format(DeltaF_rate_mean))
    
    G = nx.DiGraph() # Initialize directed graph
    
    Ftraj = [] # node fitness at each step of the run
    traj_states = [] # corresponding node coordinates/states
    
    fvals = 0 # number of unique function evaluations
    
    crd_cur_tuple = crd_init_tuple
    
    # Record initial node stats (regular node):
    G.add_node(crd_cur_tuple, F=fitness_f(crd_cur_tuple,mode,params), nnb=N, trials=0, lx=get_l_extra(0,N,0))
    fvals += 1
    
    # Add a terminal (pseudo) node:
    crd_cur_tuple_term = crd_cur_tuple
    crd_cur_tuple_term += ("t",)
    G.add_node(crd_cur_tuple_term, F=-round(G.nodes[crd_cur_tuple]['lx'],4)) # remove DeltaF_rate_mean
    
    # Add a directed edge between regular and terminal initial nodes:
    G.add_edge(crd_cur_tuple, crd_cur_tuple_term, weight=-G.nodes[crd_cur_tuple_term]['F']) # -1 for the BF algorithm   
    
    Ftraj.append(G.nodes[crd_cur_tuple]['F'])
    traj_states.append(crd_cur_tuple)
    
    # Initialize best state/fitness for the entire run:
    Fglobal_best = G.nodes[crd_cur_tuple]['F']
    crd_global_best_tuple = crd_cur_tuple
    
    l = 0
    num_jumps = 0
    while l < ltot:
        ###
        crd_new_tuple = move_f(crd_cur_tuple,moveset_mode,mparams) # make a move
        crd_new_tuple_term = crd_new_tuple
        crd_new_tuple_term += ("t",)
        
        # Add a new node if not already present:
        if not crd_new_tuple in G:
            G.add_node(crd_new_tuple, F=fitness_f(crd_new_tuple,mode,params), nnb=N, trials=0, lx=get_l_extra(0,N,0))
            fvals += 1
            ####
            G.add_node(crd_new_tuple_term, F=-round(G.nodes[crd_new_tuple]['lx'],4))
            G.add_edge(crd_new_tuple, crd_new_tuple_term, weight=-G.nodes[crd_new_tuple_term]['F']) # -1 for the BF algorithm
        
        # Add a new edge if not already present:
        if not G.has_edge(crd_cur_tuple, crd_new_tuple):
            weight_cur = G.nodes[crd_new_tuple]['F'] - G.nodes[crd_cur_tuple]['F']
            G.add_edge(crd_cur_tuple, crd_new_tuple, weight=-weight_cur) # -1 for the BF algorithm
        
        # Update current node/edge info:
        G.nodes[crd_cur_tuple]['trials'] += 1
        G.nodes[crd_cur_tuple]['lx'] = \
            get_l_extra(G.nodes[crd_cur_tuple]['trials'], G.nodes[crd_cur_tuple]['nnb'], \
                   len(list(G.successors(crd_cur_tuple)))-1) # -1 for the "t" successor
        ####
        G.nodes[crd_cur_tuple_term]['F'] = -round(G.nodes[crd_cur_tuple]['lx'],4)
        G.edges[crd_cur_tuple, crd_cur_tuple_term]['weight'] = -G.nodes[crd_cur_tuple_term]['F'] # -1 for the BF algorithm (obsolete but kept)
        
        ##############################
        # Now compute the best move:
        ##############################
        path_length = {}
        path_length[crd_cur_tuple] = 0.0
        path_steps = {}
        path_steps[crd_cur_tuple] = 0
        ###
        get_path_lengths(G, DeltaF_rate_mean, path_length, path_steps, l_max)
        
        l_min = None
        node_min = None
        for x in path_length:
            if x[-1] == "t": # terminal node
                if l_min == None or (l_min != None and path_length[x] + DeltaF_rate_mean * path_steps[x] < l_min):
                    l_min = path_length[x] + DeltaF_rate_mean * path_steps[x]
                    #node_min = x[:4] # tuple crds of the best non-terminal node
                    node_min = x[:-1] # tuple crds of the best non-terminal node
        
        if G.nodes[node_min]['lx'] == L_EXTRA_MAX:

            # OPTION 1:
            if sopt == 1:
                MAX_STEPS = 1000
                crds_final = []
                jump_sideways(G, crd_cur_tuple, crds_final, MAX_STEPS)
                ####
                if len(crds_final) == 0:
                    raise Exception("No descendants found for {}".format(crd_cur_tuple))
                
                node_min = crds_final[-1]
                #############################
            
            # OPTION 2:
            elif sopt == 2:
                node_min = random.choice(list(path_length))
                if node_min[-1] == "t": # terminal node
                    node_min = node_min[:-1]
                #############################
            
            else:
                raise Exception("Unknown value of sopt: {}".format(sopt))
            
            num_jumps += 1
        
        # Process the move (the walker might stay on the same node or move to another node on the network):
        if node_min != crd_cur_tuple: # leave
            crd_cur_tuple = node_min
            crd_cur_tuple_term = node_min
            crd_cur_tuple_term += ("t",)
        
        #### Update best state/fitness for the entire run: ####
        if G.nodes[crd_cur_tuple]['F'] > Fglobal_best:
            Fglobal_best = G.nodes[crd_cur_tuple]['F']
            crd_global_best_tuple = crd_cur_tuple
        
        #### Record all moves, leave or stay: ####
        Ftraj.append(G.nodes[crd_cur_tuple]['F'])
        traj_states.append(crd_cur_tuple)
        
        if max_feval is not None and max_feval == fvals:
            print("TERMINATING - REACHED MAX FEVALS")
            # Build a global node_occ hash:
            node_trials = {}
            for nd in list(G.nodes):
                if nd[-1] != "t": # non-terminal node
                    node_trials[nd] = G.nodes[nd]['trials']
            return [fvals,Fglobal_best,crd_global_best_tuple,node_trials,Ftraj,traj_states]

        #### Figure out the slope of the last M moves ####
        if l > 0 and l%M == 0:
            model = np.polyfit(steps, Ftraj[-M:], deg=1)
            Fpred = np.polyval(model, steps)
            DeltaF_rate_actual = (Fpred[-1] - Fpred[0])/M
            ### rate -> rate if rate >=sf, otherwise rate -> sf*exp(rate - sf)
            sf = 0.001 # scaling factor
            if DeltaF_rate_actual >= sf:
                DeltaF_rate_rectified = DeltaF_rate_actual
            else:
                DeltaF_rate_rectified = sf*np.exp(DeltaF_rate_actual - sf)
            ####
            DeltaF_rate_mean = optimism * DeltaF_rate_rectified # multiply 'rectified' rate by the optimism level
            
        if l > 0 and l%50000 == 0:
            print("l = {} ..".format(l))
        
        l += 1 # global move counter
    
    print("Effectuated {} sideways jumps ..".format(num_jumps))

    # Build a global node_occ hash:
    node_trials = {}
    for nd in list(G.nodes):
        if nd[-1] != "t": # non-terminal node
            node_trials[nd] = G.nodes[nd]['trials']
    
    #return [fvals,Fglobal_best,crd_global_best_tuple,int(len(G.nodes)/2),Ftraj,traj_states] # divide by 2 not to count 't' nodes
    return [fvals,Fglobal_best,crd_global_best_tuple,node_trials,Ftraj,traj_states]

