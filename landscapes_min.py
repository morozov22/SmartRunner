#########################
### landscapes_min.py ###
#########################

from math import *
import numpy as np
import random
#import networkx as nx
#import sys
#import re


########################################
### Landscape parameter definitions: ###
########################################


# ranges = test function domains
def get_ranges(D,mode):

    if mode == "Rastrigin_4D":
        ranges = np.array([-5,5,-5,5,-5,5,-5,5]).reshape((D, 2))
    elif mode == "Ackley_4D":
        ranges = np.array([-32.8,32.8,-32.8,32.8,-32.8,32.8,-32.8,32.8]).reshape((D, 2))
    elif mode == "Griewank_4D":
        ranges = np.array([-600,600,-600,600,-600,600,-600,600]).reshape((D, 2))
    elif mode == "double_well_2D":
        ranges = np.array([-10,10,-10,10]).reshape((D, 2))
    elif mode[:2] == "SK" or mode[:2] == "NK":
        ranges = None
    else:
        raise Exception("Unknown mode: {} in get_ranges!".format(mode))
    ####
    return ranges


# Delta_x = step size on discretized fitness lanscapes
def get_Delta_x(mode):

    if mode == "Rastrigin_4D":
        Delta_x = 0.05
        #Delta_x = 0.01
    elif mode == "Ackley_4D":
        Delta_x = 0.2
        #Delta_x = 0.05
        #Delta_x = 0.3
    elif mode == "Griewank_4D":
        Delta_x = 1.0
    elif mode == "double_well_2D":
        Delta_x = 0.01
    elif mode[:2] == "SK" or mode[:2] == "NK":
        Delta_x = None
    else:
        raise Exception("Unknown mode: {} in get_Delta_x!".format(mode))
    ####
    return Delta_x


# D = number of landscape dimensions
# N = total number of nearest neighbors
def get_DN(mode,moveset_mode):

    ####
    if mode == "Rastrigin_4D":
        D = 4 # number of landscape dimensions
    elif mode == "Ackley_4D":
        D = 4 # number of landscape dimensions
    elif mode == "Griewank_4D":
        D = 4 # number of landscape dimensions
    elif mode == "double_well_2D":
        D = 2
    elif mode[:2] == "SK":
        D = int(mode[2:])
    elif mode[:2] == "NK":
        toks = mode[2:].split(".")
        assert len(toks) == 2, "Error: unknown NK mode format in get_DN: {}!".format(mode)
        D = int(toks[0])
    else:
        raise Exception("Unknown mode: {} in get_DN!".format(mode))

    ####
    if moveset_mode == "nnb":
        N = 2*D # total number of neighbors, assumed to be known for now (move set dependent)
    elif moveset_mode == "single_spin_flip": # this works for SK and NK models
        N = D
    elif moveset_mode == "spmut":
        ranges = get_ranges(D,mode)
        Delta_x = get_Delta_x(mode)
        N = 0
        for j in range(D):
            max_steps = int(floor((ranges[j,1] - ranges[j,0])/Delta_x))
            N += max_steps
    else:
        raise Exception("Unknown mode: {} in get_DN!".format(moveset_mode))

    return D,N



##########################
### Fitness functions: ###
##########################


# Overall fitness function caller.
# params is a tuple of parameters to be passed into the fitness function.
def get_fitness(crd,mode,params=None):

    ####
    if mode == "Rastrigin_4D":
        F = Rastrigin_4D(crd)
    elif mode == "Ackley_4D":
        F = Ackley_4D(crd)
    elif mode == "Griewank_4D":
        F = Griewank_4D(crd)
    elif mode == "double_well_2D":
        F = double_well_2D(crd)
    elif mode[:2] == "SK":
        F = H_SK(crd,params[0]) # params[0] = Jij_set
    elif mode[:2] == "NK":
        F = F_NK(crd,params[0],params[1])
    else:
        raise Exception("Unknown mode: {} in get_fitness!".format(mode))

    return F


# 2D double-well potential:
def double_well_2D(crd):

    mu11, mu12, sigma11, sigma12 = 3.5, 0.0, 2.0, 3.0
    mu21, mu22, sigma21, sigma22 = -3.5, 0.0, 3.0, 2.0 # -3.5, 0.0, 1.0, 2.0 for double_well_test4.dat
    A1, A2 = 75, 50

    f1 = np.exp(-(1.0/(2*sigma11*sigma11))*(crd[0] - mu11)**2 - (1.0/(2*sigma12*sigma12))*(crd[1] - mu12)**2)
    f2 = np.exp(-(1.0/(2*sigma21*sigma21))*(crd[0] - mu21)**2 - (1.0/(2*sigma22*sigma22))*(crd[1] - mu22)**2)

    return round(A1*f1 + A2*f2,4)


# 4D Rastrigin function:
def Rastrigin_4D(crd):
    
    for j in range(len(crd)):
        if crd[j] < -5.0 or crd[j] > 5.0:
            raise Exception("Coordinate {} is out of the [-5,5] range: {:.3f} in Rastrigin_4D!".format(j,crd[j]))
    
    fval = 4.0
    for j in range(len(crd)):
        fval += crd[j]*crd[j] - np.cos(18.*crd[j])
    
    return -round(fval,4)


# 4D Ackley function:
def Ackley_4D(crd):

    for j in range(len(crd)):
        if crd[j] < -32.8 or crd[j] > 32.8:
            raise Exception("Coordinate {} is out of the [-32.8,32.8] range: {:.3f} in Ackley_4D!".format(j,crd[j]))

    ####
    val1 = 0.0
    val2 = 0.0
    for j in range(len(crd)):
        val1 += crd[j]*crd[j]
        val2 += np.cos(2*np.pi*crd[j])
    ####
    fval = 20.0 + np.e - 20.0 * np.exp(-0.2 * np.sqrt(0.25*val1)) - np.exp(0.25*val2)
    
    return -round(fval,4)


# 4D Griewank function:
def Griewank_4D(crd):
    
    for j in range(len(crd)):
        if crd[j] < -600.0 or crd[j] > 600.0:
            raise Exception("Coordinate {} is out of the [-600,600] range: {:.3f} in Griewank_4D!".format(j,crd[j]))
    
    val1 = 0.0
    val2 = 1.0
    for j in range(len(crd)):
        val1 += crd[j]*crd[j]
        val2 *= np.cos(crd[j]/np.sqrt(j+1))

    fval = 1.0 + (1.0/4000.0) * val1 - val2
    
    return -round(fval,4)


# A set of parameters for the NK model:
def NKprms(Nsites,Nnb):
    if Nsites <= 0:
        raise Exception("Provide a positive number of sites in NKprms - {} is invalid!".format(Nsites))

    if Nnb < 0 or Nnb > Nsites-1:
        raise Exception("Provide a valiud number of neighbors in NKprms - {} is invalid for Nsites={}!".format(Nnb,Nsites))

    dim = Nsites * (Nnb + 1)
    # N x (K+1) matrix of 0-based nb indices for each site (including self):
    nb_ind = np.zeros(dim,dtype=np.int64).reshape((Nsites,Nnb+1))
    # N x (K+1) matrix of fitness values for each site (including self):
    Fvals = np.random.uniform(size=(Nsites,int(2**(Nnb+1))))

    site_arr = list(range(Nsites))

    for j in site_arr:
        site_arr_cur = site_arr.copy()
        site_arr_cur.remove(j)
        # Generate and save K nb indices:
        nnb_ind_cur = random.sample(site_arr_cur,Nnb)
        nb_ind[j,0] = j # self first ..
        for k in range(len(nnb_ind_cur)):
            nb_ind[j,k+1] = nnb_ind_cur[k]

    return [nb_ind,Fvals]


# NK model fitness:
def F_NK(crd,nb_ind,Fvals):

    if nb_ind.shape[0] != Fvals.shape[0]:
        raise Exception("Matrix dimension mismatch in F_NK!")

    Nsites = nb_ind.shape[0]
    Nnb = nb_ind.shape[1]

    if len(crd) != Nsites:
        raise Exception("Mismatch between spin_state and interaction_matrix dims in F_NK!")

    F = 0.0

    for i in range(0, Nsites):
        bits = []
        for j in range(0, Nnb):
            cur_bit = crd[nb_ind[i,j]]
            if cur_bit == -1:
                cur_bit = '0' # switch to binary
            elif cur_bit == 1:
                cur_bit = '1'
            else:
                raise Exception("Unknown bit value: {}!".format(cur_bit))
            ####
            bits.append(cur_bit)
        bit_str = ''.join(bits)
        ####
        arr_pos = int(bit_str,2)
        F += Fvals[i,arr_pos]

    return round(F/Nsites,4)


# A set of spin glass constants Jij:
def Jij(Nspins, discrete=False):

    if Nspins <= 0:
        raise Exception("Provide a positive number of spins in Jij - {} is invalid!".format(Nspins))

    if discrete == False:
        a = np.random.normal(size=int(Nspins*(Nspins-1)/2))
    else:
        a = np.random.choice([-1.0,1.0], size=int(Nspins*(Nspins-1)/2))

    Jij_set = np.zeros((Nspins,Nspins))


    cnt = 0
    for i in range(0, Nspins):
        for j in range(i+1, Nspins):
            Jij_set[i,j] = a[cnt]
            cnt += 1

    return Jij_set


# SK spin glass 'fitness' (-energy):
def H_SK(crd,Jij_set):

    if Jij_set.shape[0] != Jij_set.shape[1]:
        raise Exception("Expected a square matrix in H_SK!")

    Nspins = Jij_set.shape[0]

    if len(crd) != Nspins:
        raise Exception("Mismatch between spin_state and interaction_matrix in H_SK!")

    E_SK = 0.0

    for i in range(0, Nspins):
        for j in range(i+1, Nspins):
            E_SK -= Jij_set[i,j] * crd[i] * crd[j]

    return -round(E_SK/(Nspins*sqrt(Nspins)),4)


#############################
### Move set definitions: ###
#############################


# High-level moveset selector.
def make_move(old_crd,moveset_mode,params=None):

    ####
    if moveset_mode == "nnb":
        #new_crd_tuple = make_move_nnb(old_crd,Delta_x,ranges,dirn,dimn)
        new_crd_tuple = make_move_nnb(old_crd,params[0],params[1],params[2],params[3])
    elif moveset_mode == "single_spin_flip":
        new_crd_tuple = make_spin_flip(old_crd)
    elif moveset_mode == "spmut":
        new_crd_tuple = make_mutation(old_crd,params[0],params[1],params[3])
    else:
        raise Exception("Unknown mode: {} in make_move!".format(moveset_mode))

    return new_crd_tuple


# Random move on a 2D/4D lattice by +- Delta_x (with periodic BCs).
# This version returns a tuple rather than a list.
def make_move_nnb(old_crd,Delta_x,ranges,dirn=None,dimn=None):
    
    assert len(old_crd) == len(ranges), "Error: dim mismatch in make_move_nnb!"
    assert len(np.shape(ranges)) == 2 and np.shape(ranges)[1] == 2, \
                            "Error: unexpected dims in ranges in make_move_nnb!"
    
    direction = [-1,1]
    dimension = list(range(0, len(old_crd)))
    
    if dirn == None:
        dir_cur = random.choice(direction)
    else:
        dir_cur = dirn
        assert dir_cur in direction, "Error: illegal dir={} in make_move_nnb!".format(dir_cur)
    ###
    if dimn == None:
        dim_cur = random.choice(dimension)
    else:
        dim_cur = dimn
        assert dim_cur in dimension, "Error: illegal dim={} in make_move_nnb!".format(dim_cur)
    
    new_crd = list(old_crd)
    new_crd[dim_cur] = old_crd[dim_cur] + dir_cur*Delta_x
    
    # Periodic BCs:
    if new_crd[dim_cur] < ranges[dim_cur][0]:
        new_crd[dim_cur] += (ranges[dim_cur][1] - ranges[dim_cur][0])
    
    if new_crd[dim_cur] > ranges[dim_cur][1]:
        new_crd[dim_cur] -= (ranges[dim_cur][1] - ranges[dim_cur][0])
    
    for j in range(len(new_crd)):
        new_crd[j] = round(new_crd[j],4)

    return tuple(new_crd)


# Make a single-point mutation on a 2D/4D lattice.
def make_mutation(old_crd,Delta_x,ranges,dimn=None):

    assert len(old_crd) == len(ranges), "Error: dim mismatch in make_mutation!"
    assert len(np.shape(ranges)) == 2 and np.shape(ranges)[1] == 2, \
                            "Error: unexpected dims in ranges in make_mutation!"

    dimension = list(range(0, len(old_crd)))
    ###
    if dimn == None:
        dim_cur = random.choice(dimension)
    else:
        dim_cur = dimn
        assert dim_cur in dimension, "Error: illegal dim={} in make_mutation!".format(dim_cur)

    new_crd = list(old_crd)

    max_steps = int(floor((ranges[dim_cur,1] - ranges[dim_cur,0])/Delta_x))

    while 1:
        crd_mut = ranges[dim_cur,0] + Delta_x*random.randint(0,max_steps)
        crd_mut = round(crd_mut,4)
        if crd_mut != new_crd[dim_cur]:
            new_crd[dim_cur] = crd_mut
            break

    return tuple(new_crd)


# Random or pre-defined single-spin flip:
def make_spin_flip(old_crd,ind=None):

    new_crd = list(old_crd)

    if ind == None:
        indices = range(len(old_crd))
        rind = random.choice(indices)
        new_crd[rind] *= -1
        new_crd[rind] = int(new_crd[rind])
    else:
        assert ind >= 0 and ind < len(old_crd), "Error: invalid index in make_spin_flip!"
        new_crd[ind] *= -1
        new_crd[ind] = int(new_crd[ind])

    return tuple(new_crd)


# This functions returns a list of all neighbors of the current node, or a list of
# all spin state neighbors connected by a single spin flip:
def get_all_neighbors(crd_cur,moveset_mode,params=None):
    
    nnb = []

    if moveset_mode == "single_spin_flip":
        for i in range(len(crd_cur)):
            nnb.append(make_spin_flip(crd_cur, ind=i))
    elif moveset_mode == "nnb": # various 2D,4D landscapes with nnb moves 
        direction = [-1,1]
        dimension = list(range(0, len(crd_cur)))
        
        for dr in direction:
            for dm in dimension:
                nnb.append(make_move_nnb(crd_cur,params[0],params[1],dirn=dr,dimn=dm))
    elif moveset_mode == "spmut":
        dimension = list(range(0, len(crd_cur)))
        Delta_x = params[0]
        ranges = params[1]
        for j in dimension:
            max_steps = int(floor((ranges[j,1] - ranges[j,0])/Delta_x))
            for k in range(max_steps+1):
                crd_nnb = list(crd_cur)
                crd_tmp = ranges[j,0] + k*Delta_x
                crd_tmp = round(crd_tmp,4)
                if crd_tmp != crd_nnb[j]:
                    crd_nnb[j] = crd_tmp
                    nnb.append(tuple(crd_nnb))

    else:
        raise Exception("Unknown mode: {} in get_all_neighbors!".format(moveset_mode))
    
    return nnb


############################
### Auxiliary functions: ###
############################


# This is a high-level random state generator.
# params = (ranges,Delta_x)
def generate_random_state(dim,mode,params=None):

    if mode[:2] == "SK" or mode[:2] == "NK":
        random_crd = generate_random_spin_state(dim)
    else:
        random_crd = generate_random_crd(dim,params[0],params[1])

    return random_crd


# This function generates a random coord, e.g. to start the run or respawn in a random place.
# This version returns a tuple rather than a list.
def generate_random_crd(dim,ranges,Delta_x):
    
    assert len(ranges) == dim, "Error: dim mismatch in generate_random_crd!"
    
    crd_rand = np.zeros(dim)
    
    for j in range(dim):
        max_steps = int(floor((ranges[j,1] - ranges[j,0])/Delta_x))
        crd_rand[j] = ranges[j,0] + Delta_x*random.randint(0,max_steps)
    
    for j in range(dim):
        crd_rand[j] = round(crd_rand[j],4)

    return tuple(crd_rand)


# This function generates a random spin configuration.
def generate_random_spin_state(dim):
    
    spins = [-1,1]

    return tuple(random.choices(spins, k=dim))

