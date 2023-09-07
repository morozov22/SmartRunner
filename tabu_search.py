###################
### Tabu Search ###
###################

############################################################
# Adapted from Solid, a gradient-free Python optimization package:
# https://github.com/100/Solid
############################################################

from landscapes_min import *

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
from numpy import argmax

class TabuSearch:
    """
    Conducts tabu search
    """
    __metaclass__ = ABCMeta

    cur_steps = None

    tabu_size = None
    tabu_list = None

    initial_state = None
    current = None
    best = None

    max_steps = None
    max_score = None
    max_feval = None

    params = None
    mparams = None

    mode = None
    moveset_mode = None

    node_fitness = None # global hash for saving already computed fitness values
    node_trials = None # global hash for saving the number of trials at each node

    Ftraj = None # global hash for saving best_fitness at east step in the algorithm
    traj_states = None # global hash for saving best_member at east step in the algorithm

    def __init__(self, initial_state, tabu_size, max_steps, params, mparams, \
                 mode, moveset_mode, max_score=None, max_feval=None):
        """
        :param initial_state: initial state, should implement __eq__ or __cmp__
        :param tabu_size: number of states to keep in tabu list
        :param max_steps: maximum number of steps to run algorithm for
        :param max_score: score to stop algorithm once reached
        """
        self.initial_state = initial_state

        if isinstance(tabu_size, int) and tabu_size > 0:
            self.tabu_size = tabu_size
        else:
            raise TypeError('Tabu size must be a positive integer')

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise TypeError('Maximum steps must be a positive integer')

        if max_feval is not None:
            if isinstance(max_feval, int) and max_feval > 0:
                self.max_feval = max_feval
            else:
                raise ValueError('Maximum number of fevals must be a positive integer')

        if max_score is not None:
            if isinstance(max_score, (int, float)):
                self.max_score = float(max_score)
            else:
                raise TypeError('Maximum score must be a numeric type')

        ######

        self.params = params
        self.mparams = mparams

        self.mode = mode
        self.moveset_mode = moveset_mode

        self.node_fitness = {}
        self.node_trials = {}

        self.Ftraj = []
        self.traj_states = []


    def __str__(self):
        return ('TABU SEARCH: \n' +
                'CURRENT STEPS: %d \n' +
                'BEST SCORE: %f \n' +
                'BEST MEMBER: %s \n\n') % \
               (self.cur_steps, self._score(self.best), str(self.best))

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm
        :return: None
        """
        self.cur_steps = 0
        self.tabu_list = deque(maxlen=self.tabu_size)
        self.current = self.initial_state
        self.best = self.initial_state

        self.node_fitness = {}
        self.node_trials = {}

        self.Ftraj = []
        self.traj_states = []

    @abstractmethod
    def _score(self, state):
        """
        Returns objective function value of a state
        :param state: a state
        :return: objective function value of state
        """
        pass

    @abstractmethod
    def _neighborhood(self):
        """
        Returns list of all members of neighborhood of current state, given self.current
        :return: list of members of neighborhood
        """
        pass

    def _best(self, neighborhood):
        """
        Finds the best member of a neighborhood
        :param neighborhood: a neighborhood
        :return: best member of neighborhood
        """
        return neighborhood[argmax([self._score(x) for x in neighborhood])]

    def _update_occ(self, state):
        """
        Updates occupancies for the input state
        :return: None
        """
        if state in self.node_trials:
            self.node_trials[state] += 1
        else:
            self.node_trials[state] = 0

    def run(self, verbose=False):
        """
        Conducts tabu search
        :param verbose: indicates whether or not to print progress regularly
        :return: best state and objective function value of best state
        """
        self._clear() # this also inits things ..

        for i in range(self.max_steps):
            self.cur_steps += 1

            self._update_occ(self.current) # new
            self.Ftraj.append(self._score(self.current)) # new
            self.traj_states.append(deepcopy(self.current)) # new
            
            if ((i + 1) % 100 == 0) and verbose:
                print(self)
            
            neighborhood = self._neighborhood()
            neighborhood_best = self._best(neighborhood)
            

            while True:
                if all([x in self.tabu_list for x in neighborhood]):
                    print("TERMINATING - NO SUITABLE NEIGHBORS")
                    return [len(self.node_fitness), self._score(self.best), self.best, self.node_trials, self.Ftraj, self.traj_states]
                if neighborhood_best in self.tabu_list:
                    if self._score(neighborhood_best) > self._score(self.best):
                        self.tabu_list.append(neighborhood_best)
                        self.best = deepcopy(neighborhood_best)
                        break
                    else:
                        neighborhood.remove(neighborhood_best)
                        neighborhood_best = self._best(neighborhood)
                else:
                    self.tabu_list.append(neighborhood_best)
                    self.current = neighborhood_best
                    if self._score(self.current) > self._score(self.best):
                        self.best = deepcopy(self.current)
                    break

            if self.max_score is not None and self._score(self.best) > self.max_score:
                print("TERMINATING - REACHED MAXIMUM SCORE")
                return [len(self.node_fitness), self._score(self.best), self.best, self.node_trials, self.Ftraj, self.traj_states]

            if self.max_feval is not None and self.max_feval <= len(self.node_fitness):
                print("TERMINATING - REACHED MAX FEVALS")
                return [len(self.node_fitness), self._score(self.best), self.best, self.node_trials, self.Ftraj, self.traj_states]
        
        print("TERMINATING - REACHED MAXIMUM STEPS")

        return [len(self.node_fitness), self._score(self.best), self.best, self.node_trials, self.Ftraj, self.traj_states]


################################

class Algorithm_TB(TabuSearch):
    
    def _neighborhood(self):
        crd_cur = tuple(self.current)
        return get_all_neighbors(crd_cur,self.moveset_mode,self.mparams)

    def _score(self, state):
        
        if state in self.node_fitness:
            Fcur = self.node_fitness[state]
        else:
            Fcur = get_fitness(state,self.mode,self.params)
            self.node_fitness[state] = Fcur
            
        return Fcur

################################


