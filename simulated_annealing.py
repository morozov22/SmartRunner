###########################
### Simulated Annealing ###
###########################

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
import numpy as np

class SimulatedAnnealing:
    """
    Conducts simulated annealing algorithm
    """
    __metaclass__ = ABCMeta

    initial_state = None
    current_state = None
    best_state = None

    cur_steps = 0
    max_steps = None

    current_energy = None
    best_energy = None
    min_energy = None
    max_feval = None

    start_temp = None
    finish_temp = None
    current_temp = None
    adjust_temp = None

    params = None
    mparams = None
    #rparams = None

    mode = None
    moveset_mode = None

    Rbar = None

    node_energy = None # global hash for saving already computed fitness values
    node_trials = None # global hash for saving the number of trials at each node

    Ftraj = None # global hash for saving best_fitness at east step in the algorithm
    traj_states = None # global hash for saving best_member at east step in the algorithm

    def _exponential(self, schedule_constant):
        def f():
            self.current_temp *= schedule_constant
        return f

    def _linear(self, schedule_constant):
        def f():
            self.current_temp -= schedule_constant
        return f

    def _get_schedule(self, schedule_str, schedule_constant):
        if schedule_str == 'exponential':
            return self._exponential(schedule_constant)
        elif schedule_str == 'linear':
            return self._linear(schedule_constant)
        else:
            raise ValueError('Annealing schedule must be either "exponential" or "linear"')

    def __init__(self, temp_begin, temp_end, max_steps, params, mparams, \
                 mode, moveset_mode, initial_state, Rbar, min_energy=None, max_feval=None, schedule='linear'):
        """
        :param initial_state: initial state of annealing algorithm
        :param max_steps: maximum number of iterations to conduct annealing for
        :param temp_begin: beginning temperature
        :param schedule_constant: constant value in annealing schedule function
        :param min_energy: energy value to stop algorithm once reached
        :param schedule: 'exponential' or 'linear' annealing schedule
        """

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise ValueError('Max steps must be a positive integer')

        if min_energy is not None:
            if isinstance(min_energy, (float, int)):
                self.min_energy = float(min_energy)
            else:
                raise ValueError('Minimum energy must be a numeric type')

        if max_feval is not None:
            if isinstance(max_feval, int) and max_feval > 0:
                self.max_feval = max_feval
            else:
                raise ValueError('Maximum number of fevals must be a positive integer')

        if isinstance(temp_begin, (float, int)) and temp_begin > 0:
            self.start_temp = float(temp_begin)
        else:
            raise ValueError('Initial temperature must be a positive numeric type')

        if isinstance(temp_end, (float, int)) and temp_end >= 0:
            self.finish_temp = float(temp_end)
        else:
            raise ValueError('Final temperature must be a non-negative numeric type')

        if temp_end > temp_begin:
            raise ValueError('Final temperature must be lower than or equal to the initial temperature')

        if schedule == "linear":
            schedule_constant = (temp_begin - temp_end)/max_steps
        elif schedule == "exponential":
            schedule_constant_log = (1./max_steps) * np.log(temp_end/temp_begin)
            schedule_constant = np.exp(schedule_constant_log)
        else:
            raise ValueError('Unknown schedule type')

        self.adjust_temp = self._get_schedule(schedule, schedule_constant)

        ######

        self.params = params
        self.mparams = mparams

        self.mode = mode
        self.moveset_mode = moveset_mode

        if isinstance(Rbar, (int,float)) and Rbar >= 0.0:
            self.Rbar = Rbar
        else:
            raise ValueError('Rbar must be a positive integer or float')

        self.node_energy = {}
        self.node_trials = {}

        self.Ftraj = []
        self.traj_states = []

        self.initial_state = initial_state

    def __str__(self):
        return ('SIMULATED ANNEALING: \n' +
                'CURRENT STEPS: %d \n' +
                'CURRENT TEMPERATURE: %f \n' +
                'BEST ENERGY: %f \n' +
                'BEST STATE: %s \n\n') % \
               (self.cur_steps, self.current_temp, self.best_energy, str(self.best_state))

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
        self.current_energy = None
        self.best_energy = None

        self.node_energy = {}
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
    def _energy(self, state):
        """
        Finds the energy of a given state
        :param state: a state
        :return: energy of state
        """
        pass

    def _accept_neighbor(self, neighbor):
        """
        Probabilistically determines whether or not to accept a transition to a neighbor
        :param neighbor: a state
        :return: boolean indicating whether or not transition is accepted
        """
        try:
            p = exp(-((self._energy(neighbor) + self.Rbar) - self._energy(self.current_state)) / self.current_temp) # +Rbar*1 "jump" penalty in E
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
        Conducts simulated annealing
        :param verbose: indicates whether or not to print progress regularly
        :return: best state and best energy
        """
        self._clear() # this also inits things ..

        self.current_state = self.initial_state
        self._update_occ(self.current_state) # new

        self.current_temp = self.start_temp
        self.best_energy = self._energy(self.current_state)

        self.Ftraj.append(-self.best_energy) # new
        self.traj_states.append(deepcopy(self.current_state)) # new

        for i in range(self.max_steps):
            self.cur_steps += 1

            if verbose and ((i + 1) % 100 == 0):
                print(self)

            neighbor = self._neighbor()

            if self._accept_neighbor(neighbor):
                self.current_state = neighbor
            self.current_energy = self._energy(self.current_state)
            self._update_occ(self.current_state) # new

            self.Ftraj.append(-self.current_energy) # new
            self.traj_states.append(deepcopy(self.current_state)) # new

            if self.current_energy < self.best_energy:
                self.best_energy = self.current_energy
                self.best_state = deepcopy(self.current_state)

            if self.min_energy is not None and self.current_energy < self.min_energy:
                print("TERMINATING - REACHED MINIMUM ENERGY")
                return [len(self.node_energy), -self.best_energy, self.best_state, self.node_trials, self.Ftraj, self.traj_states]
                #return self.best_state, self.best_energy

            if self.max_feval is not None and self.max_feval == len(self.node_energy):
                print("TERMINATING - REACHED MAX FEVALS")
                return [len(self.node_energy), -self.best_energy, self.best_state, self.node_trials, self.Ftraj, self.traj_states]

            self.adjust_temp()
            if self.current_temp < 0.000001:
                print("TERMINATING - REACHED TEMPERATURE OF 0")
                return [len(self.node_energy), -self.best_energy, self.best_state, self.node_trials, self.Ftraj, self.traj_states]

        print("TERMINATING - REACHED MAXIMUM STEPS")
        return [len(self.node_energy), -self.best_energy, self.best_state, self.node_trials, self.Ftraj, self.traj_states]


################################

class Algorithm_SA(SimulatedAnnealing):
    
    def _neighbor(self):
        return make_move(self.current_state,self.moveset_mode,self.mparams)

    def _energy(self, member):
        
        if member in self.node_energy:
            Ecur = self.node_energy[member]
        else:
            Ecur = -get_fitness(member,self.mode,self.params) # -1 for the energy
            self.node_energy[member] = Ecur


        if member in self.node_trials:
            n = self.node_trials[member]
        else:
            n = 0

        Ecur += self.Rbar * (get_l_extra_approx(n) - 2) # occupancy penalty; note that get_l_extra_approx(0) = 2 is subtracted off

        return Ecur

################################


    