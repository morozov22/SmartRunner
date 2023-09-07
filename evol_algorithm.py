##############################
### Evolutionary Algorithm ###
##############################

############################################################
# Adapted from Solid, a gradient-free Python optimization package:
# https://github.com/100/Solid
############################################################

from landscapes_min import *
from SmartRunner import get_l_extra_approx

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from random import random, randint, shuffle


class EvolutionaryAlgorithm:
    """
    Conducts evolutionary algorithm
    """
    __metaclass__ = ABCMeta

    population = None
    fitnesses = None

    crossover_rate = None

    mutation_rate = None

    cur_steps = None
    best_fitness = None
    best_member = None

    max_steps = None
    max_fitness = None
    max_feval = None

    params = None
    mparams = None
    rparams = None

    mode = None
    moveset_mode = None

    Npop = None
    dim = None

    Rbar = None

    node_fitness = None # global hash for saving already computed fitness values
    node_trials = None # global hash for saving the number of trials at each node

    Ftraj = None # global hash for saving best_fitness at east step in the algorithm
    traj_states = None # global hash for saving best_member at east step in the algorithm


    def __init__(self, crossover_rate, mutation_rate, max_steps, params, mparams, rparams, \
                 mode, moveset_mode, Npop, dim, Rbar, max_fitness=None, max_feval=None):
        """
        :param crossover_rate: probability of crossover
        :param mutation_rate: probability of mutation
        :param max_steps: maximum steps to run genetic algorithm for
        :param max_fitness: fitness value to stop algorithm once reached
        """
        if isinstance(crossover_rate, float):
            if 0 <= crossover_rate <= 1:
                self.crossover_rate = crossover_rate
            else:
                raise ValueError('Crossover rate must be a float between 0 and 1')
        else:
            raise ValueError('Crossover rate must be a float between 0 and 1')

        if isinstance(mutation_rate, float):
            if 0 <= mutation_rate <= 1:
                self.mutation_rate = mutation_rate
            else:
                raise ValueError('Mutation rate must be a float between 0 and 1')
        else:
            raise ValueError('Mutation rate must be a float between 0 and 1')

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise ValueError('Maximum steps must be a positive integer')

        if max_fitness is not None:
            if isinstance(max_fitness, (int, float)):
                self.max_fitness = float(max_fitness)
            else:
                raise ValueError('Maximum fitness must be a numeric type')

        if max_feval is not None:
            if isinstance(max_feval, int) and max_feval > 0:
                self.max_feval = max_feval
            else:
                raise ValueError('Maximum number of fevals must be a positive integer')

        ######

        self.params = params
        self.mparams = mparams
        self.rparams = rparams

        self.mode = mode
        self.moveset_mode = moveset_mode

        if isinstance(Npop, int) and Npop > 0:
            self.Npop = Npop
        else:
            raise ValueError('Population size must be a positive integer')

        if isinstance(dim, int) and dim > 0:
            self.dim = dim
        else:
            raise ValueError('Fitness landscape dimension must be a positive integer')

        if isinstance(Rbar, (int,float)) and Rbar >= 0.0:
            self.Rbar = Rbar
        else:
            raise ValueError('Rbar must be a positive integer or float')

        self.node_fitness = {}
        self.node_trials = {}

        self.Ftraj = []
        self.traj_states = []


    def __str__(self):
        return ('EVOLUTIONARY ALGORITHM: \n' +
                'CURRENT STEPS: %d \n' +
                'BEST FITNESS: %f \n' +
                'BEST MEMBER: %s \n\n') % \
               (self.cur_steps, self.best_fitness, str(self.best_member))

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm
        :return: None
        """
        self.cur_steps = 0
        self.population = None
        self.fitnesses = None
        self.best_member = None
        self.best_fitness = None

        self.node_fitness = {}
        self.node_trials = {}

        self.Ftraj = []
        self.traj_states = []

        #self.offset = None
        #self.num_rect = 0


    @abstractmethod
    def _initial_population(self):
        """
        Generates initial population
        :return: list of members of population
        """
        pass

    @abstractmethod
    def _fitness(self, member):
        """
        Evaluates fitness of a given member
        :param member: a member
        :return: fitness of member
        """
        pass

    def _populate_fitness(self):
        """
        Calculates fitness of all members of current population
        :return: None
        """
        self.fitnesses = [self._fitness(x) for x in self.population]

    def _update_occ(self):
        """
        Updates occupancies for all members of current population
        :return: None
        """
        for x in self.population:
        	if x in self.node_trials:
        		self.node_trials[x] += 1
        	else:
        		self.node_trials[x] = 0

    def _most_fit(self):
        """
        Finds most fit member of current population
        :return: most fit member and most fit member's fitness
        """
        best_idx = 0
        cur_idx = 0
        for x in self.fitnesses:
            if x > self.fitnesses[best_idx]:
                best_idx = cur_idx
            cur_idx += 1
        return self.population[best_idx], self.fitnesses[best_idx]

    def _select_n(self, n):
        """
        Probabilistically selects n members from current population using
        roulette-wheel selection
        :param n: number of members to select
        :return: n members
        """
        assert n > 0, "Error: provide a positive integer instead of {} in _select_n!".format(n)
        shuffle(self.population)
        total_fitness = sum(self.fitnesses)
        min_fitness = min(self.fitnesses)
        total_fitness -= len(self.fitnesses) * min_fitness # subtract min(F) from the sum

        #if total_fitness != 0:
        if abs(total_fitness) > 1e-06:
            probs = list([(self._fitness(x) - min_fitness) / total_fitness for x in self.population])
            psum = sum(probs)
            
            if abs(psum) < 1e-06: # all sequences have the same fitness, return first n sequences
                return self.population[0:n]

            if abs(psum - 1.0) > 1e-06:
            	raise Exception("Invalid probs: sum={} in _select_n!".format(psum))

        else:
            return self.population[0:n] # all sequences have zero fitness, return first n sequences
        
        res = []
        for _ in range(n):
            r = random()
            sum_ = 0
            for i, x in enumerate(probs):
                sum_ += probs[i]
                if r <= sum_:
                    res.append(deepcopy(self.population[i]))
                    break

        if len(res) != n:
            raise Exception("Error: selected {} instead of {} population members in _select_n({})!".format(len(res),n,n))

        return res

    @abstractmethod
    def _crossover(self, parent1, parent2):
        """
        Creates new member of population by combining two parent members
        :param parent1: a member
        :param parent2: a member
        :return: member made by combining elements of both parents
        """
        pass

    @abstractmethod
    def _mutate(self, member):
        """
        Randomly mutates a member
        :param member: a member
        :return: mutated member
        """
        pass

    def run(self, verbose=False):
        """
        Conducts evolutionary algorithm
        :param verbose: indicates whether or not to print progress regularly
        :return: best state and best objective function value
        """
        self._clear()

        self.population = self._initial_population()
        self._update_occ() # new
        self._populate_fitness()
        self.best_member, self.best_fitness = self._most_fit()
        self.Ftraj.append(self.best_fitness) # new
        self.traj_states.append(deepcopy(self.best_member)) # new
        ######
        num_copy = max(int((1 - self.crossover_rate) * len(self.population)), 2)
        num_crossover = len(self.population) - num_copy
        
        for i in range(self.max_steps):
            self.cur_steps += 1

            if verbose and ((i + 1) % 100 == 0):
                print(self)

            self.population = self._select_n(num_copy)
            self._populate_fitness()

            parents = self._select_n(2)
            for _ in range(num_crossover):
                self.population.append(self._crossover(*parents))

            self.population = list([self._mutate(x) for x in self.population])
            self._update_occ() # new
            self._populate_fitness()

            best_member, best_fitness = self._most_fit()
            self.Ftraj.append(best_fitness) # new
            self.traj_states.append(deepcopy(best_member)) # new
            ####
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_member = deepcopy(best_member)
            
            if self.max_fitness is not None and self.best_fitness >= self.max_fitness:
                print("TERMINATING - REACHED MAXIMUM FITNESS")
                return [len(self.node_fitness), self.best_fitness, self.best_member, self.node_trials, self.Ftraj, self.traj_states]

            if self.max_feval is not None and self.max_feval <= len(self.node_fitness):
                print("TERMINATING - REACHED MAX FEVALS")
                return [len(self.node_fitness), self.best_fitness, self.best_member, self.node_trials, self.Ftraj, self.traj_states]

        print("TERMINATING - REACHED MAXIMUM STEPS")


        return [len(self.node_fitness), self.best_fitness, self.best_member, self.node_trials, self.Ftraj, self.traj_states]
        

################################

class Algorithm_EA(EvolutionaryAlgorithm):

    def _initial_population(self):
        return list(generate_random_state(self.dim,self.mode,self.rparams) for _ in range(self.Npop))

    def _fitness(self, member):
        if member in self.node_fitness:
            Fcur = self.node_fitness[member]
        else:
            Fcur = get_fitness(member,self.mode,self.params)
            self.node_fitness[member] = Fcur

        if member in self.node_trials:
            n = self.node_trials[member]
        else:
            n = 0

        Fcur -= self.Rbar * (get_l_extra_approx(n) - 2) # occupancy penalty; note that get_l_extra_approx(0) = 2 is subtracted off

        return Fcur

    def _crossover(self, parent1, parent2):
        partition = randint(0, len(self.population[0]) - 1)
        return parent1[0:partition] + parent2[partition:] # should work on tuples

    # Note that mutation is actually implemented here via existing move sets
    def _mutate(self, member):

        if self.mutation_rate >= random():
            new_member = make_move(member,self.moveset_mode,self.mparams)
        else:
        	new_member = member

        return new_member

################################

