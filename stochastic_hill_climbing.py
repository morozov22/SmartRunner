################################
### Stochastic Hill Climbing ###
################################

############################################################
# Adapted from Solid, a gradient-free Python optimization package:
# https://github.com/100/Solid
############################################################

from landscapes_min import *
from SmartRunner import get_l_extra_approx

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from math import exp
from random import random


class StochasticHillClimb:
    """
    Conducts stochastic hill climb
    """
    __metaclass__ = ABCMeta

    initial_state = None
    current_state = None
    best_state = None

    cur_steps = 0
    max_steps = None

    best_objective = None
    max_objective = None
    max_feval = None

    temp = None

    Rbar = None

    params = None
    mparams = None

    mode = None
    moveset_mode = None

    node_fitness = None # global hash for saving already computed fitness values
    node_trials = None # global hash for saving the number of trials at each node

    Ftraj = None # global hash for saving best_fitness at east step in the algorithm
    traj_states = None # global hash for saving best_member at east step in the algorithm

    def __init__(self, initial_state, temp, max_steps, params, mparams, \
                 mode, moveset_mode, Rbar, max_objective=None, max_feval=None):
        """
        :param initial_state: initial state of hill climbing
        :param max_steps: maximum steps to run hill climbing for
        :param temp: temperature in probabilistic acceptance of transition
        :param max_objective: objective function to stop algorithm once reached
        """
        self.initial_state = initial_state

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise ValueError('Max steps must be a positive integer')

        if max_objective is not None:
            if isinstance(max_objective, (float, int)):
                self.max_objective = float(max_objective)
            else:
                raise ValueError('Maximum objective must be a numeric type')

        if max_feval is not None:
            if isinstance(max_feval, int) and max_feval > 0:
                self.max_feval = max_feval
            else:
                raise ValueError('Maximum number of fevals must be a positive integer')

        if isinstance(temp, (float, int)) and temp > 0:
            self.temp = float(temp)
        else:
            raise ValueError('Temperature must be a positive float or integer')

        if isinstance(Rbar, (int,float)) and Rbar >= 0.0:
            self.Rbar = Rbar
        else:
            raise ValueError('Rbar must be a positive integer or float')

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
        return ('STOCHASTIC HILL CLIMB: \n' +
                'CURRENT STEPS: %d \n' +
                'BEST OBJECTIVE: %f \n' +
                'BEST STATE: %s \n\n') % \
               (self.cur_steps, self.best_objective, str(self.best_state))

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm
        :return: None
        """
        self.cur_steps = 0
        self.current_state = None
        self.best_state = None
        self.best_objective = None

        self.node_fitness = {}
        self.node_trials = {}

        self.Ftraj = []
        self.traj_states = []

    @abstractmethod
    def _neighbor(self):
        """
        Returns a random member of the neighbor of the current state
        :return: a random neighbor, given access to self.current_state
        """
        pass

    @abstractmethod
    def _objective(self, state):
        """
        Evaluates a given state
        :param state: a state
        :return: objective function value of state
        """
        pass

    def _accept_neighbor(self, neighbor):
        """
        Probabilistically determines whether or not to accept a transition to a neighbor
        :param neighbor: a state
        :return: boolean indicating whether or not transition was accepted
        """
        try:
            p = 1. / (1 + (exp((self._objective(self.current_state) - (self._objective(neighbor) - self.Rbar)) / self.temp))) # -Rbar*1 "jump" penalty in F
        except OverflowError:
            return True
        return True if p >= 1 else p >= random()

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
        Conducts hill climb
        :param verbose: indicates whether or not to print progress regularly
        :return: best state and best objective function value
        """
        self._clear() # this also inits things ..

        self.current_state = self.initial_state
        self._update_occ(self.current_state) # new

        self.best_objective = self._objective(self.current_state)
        self.best_state = deepcopy(self.current_state)

        self.Ftraj.append(self.best_objective) # new
        self.traj_states.append(deepcopy(self.current_state)) # new

        for i in range(self.max_steps):
            self.cur_steps += 1

            if ((i + 1) % 100 == 0) and verbose:
                print(self)

            neighbor = self._neighbor()

            if self._accept_neighbor(neighbor):
                self.current_state = neighbor

            self._update_occ(self.current_state) # new
            self.Ftraj.append(self._objective(self.current_state)) # new
            self.traj_states.append(deepcopy(self.current_state)) # new

            if self._objective(self.current_state) > self.best_objective:
                self.best_objective = self._objective(self.current_state)
                self.best_state = deepcopy(self.current_state)

            if self.max_objective is not None and self.best_objective > self.max_objective:
                print("TERMINATING - REACHED MAXIMUM OBJECTIVE")
                return [len(self.node_fitness), self.best_objective, self.best_state, self.node_trials, self.Ftraj, self.traj_states]

            if self.max_feval is not None and self.max_feval == len(self.node_fitness):
                print("TERMINATING - REACHED MAX FEVALS")
                return [len(self.node_fitness), self.best_objective, self.best_state, self.node_trials, self.Ftraj, self.traj_states]

        print("TERMINATING - REACHED MAXIMUM STEPS")
        return [len(self.node_fitness), self.best_objective, self.best_state, self.node_trials, self.Ftraj, self.traj_states]


################################ 

class Algorithm_SHC(StochasticHillClimb):
    
    def _neighbor(self):
        return make_move(self.current_state,self.moveset_mode,self.mparams)

    def _objective(self, state):

        if state in self.node_fitness:
            Fcur = self.node_fitness[state]
        else:
            Fcur = get_fitness(state,self.mode,self.params)
            self.node_fitness[state] = Fcur

        ####
        if state in self.node_trials:
            n = self.node_trials[state]
        else:
            n = 0

        Fcur -= self.Rbar * (get_l_extra_approx(n) - 2) # occupancy penalty; note that get_l_extra_approx(0) = 2 is subtracted off

        return Fcur

################################

