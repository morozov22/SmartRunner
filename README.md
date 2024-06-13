# SmartRunner Manual

## Developer: Alexandre V. Morozov (morozov@physics.rutgers.edu)

## Paper: J. Yu and A.V. Morozov, "An adaptive Bayesian approach to gradient-free global optimization" ([Yu, J. and Morozov, A.V. _New J Phys_ 26 (2004) 023027](https://iopscience.iop.org/article/10.1088/1367-2630/ad23a3))

## SmartRunner code:

* <span style="color:blue"><font size="4"> __run_optimizers.py__ (high-level Python3 script for running global optimizers) </font></span>

* <span style="color:blue"><font size="4"> __SmartRunner.py__ (SmartRunner code library) </font></span>

* <span style="color:blue"><font size="4"> __evol_algorithm.py__ (Evolutionary Algorithm code library) </font></span>

* <span style="color:blue"><font size="4"> __simulated_annealing.py__ (Simulated Annealing code library) </font></span>

* <span style="color:blue"><font size="4"> __stochastic_hill_climbing.py__ (Stochastic Hill Climbing code library) </font></span>

* <span style="color:blue"><font size="4"> __tabu_search.py__ (Taboo Search code library) </font></span>

* <span style="color:blue"><font size="4"> __extremal_optimization.py__ (Extremal Optimization code library) [as of June '24] </font></span>

* <span style="color:blue"><font size="4"> __landscapes_min.py__ (landscape and moveset definitions) </font></span>

* <span style="color:blue"><font size="4"> __utilities_min.py__ (auxiliary functions) </font></span>

`run_optimizers.py` is a high-level script for running the following five optimizers: __SmartRunner__, __Evolutionary Algorithm__, __Simulated Annealing__, __Stochastic Hill Climbing__, __Taboo Search__, and __Extremal Optimization__.
__Evolutionary Algorithm__, __Simulated Annealing__, __Stochastic Hill Climbing__, and __Taboo Search__ use code adapted from a gradient-free optimization library Solid:
[https://github.com/100/Solid](https://github.com/100/Solid). __Extremal Optimization__ is implemented following Stefan Boettcher (Boettcher, S. and Percus, A. _Artificial Intelligence_ 119 (2000) 275-286).

<br/><br/>

Run this command:
```
python3 run_optimizers.py -h
```
or
```
python3 run_optimizers.py --help
```
to obtain the list of allowed algorithm types.

<br/><br/>

Run this command:
```
python3 run_optimizers.py -h -amode=algorithm_type
```
or
```
python3 run_optimizers.py --help -amode=algorithm_type,
```
where `algorithm_type = {EvolutionaryAlgorithm,SimulatedAnnealing,TabuSearch,StochasticHillClimbing,SmartRunner,ExtremalOptimization}` to
obtain an algorithms-specific list of all available options. Most of these options are self-explanatory and the notation
corresponds to that used in the SmartRunner manuscript ([Yu, J. and Morozov, A.V. _New J Phys_ 26 (2004) 023027](https://iopscience.iop.org/article/10.1088/1367-2630/ad23a3)).

Please note that some options accept a single value or an array of
values either on a linear or $\log_{10}$ scale. For example, with StochasticHillClimbing `-T=0.2` will simply set $T=0.2$ in all runs,
`-T=\[0.1,1.0,10\]` will perform a scan over 10 uniformly spaced temperature values: $T = \{0.1, 0.2, \dots , 1.0\}$,
while `-T=\[0.1,10.0,3,\"log10\"\]` will result in $T = \{0.1, 1.0, 10.0\}$.
In all options, parentheses `()` can be used instead of brackets `[]`. The code tries to guess intelligently whether the resulting array values
should be *int* or *float*: for example, `-ltot=\(10000,50000,5\)` will result in a scan over
$l_\mathrm{tot} = \{ 10000, 20000, 30000, 40000, 50000 \}$.

The `-mode=landscape_type` option determines the fitness function to be optimized and the `-moveset_mode=moveset_type` option
determines the move type. Currently, the code supports
`landscape_type = {Rastrigin_4D,Ackley_4D,Griewank_4D,double_well_2D}` with `moveset_type = {nnb,spmut}` and
`landscape_type = {EAX1xX2,SKZ,NKX.Y}` (where X1xX2 denotes a $X_1 \times X_2$ 2D lattice with periodic boundary conditions in the Edwards-Anderson (EA) spin glass model, Z is the number of spins in the Sherrington-Kirkpatrick (SK) spin glass model, and X,Y are the number of sites and neighbors in the Kauffman's NK model) with `moveset_type = single_spin_flip`.
The code is designed to be flexible - new systems and movesets can be added in a straighforward manner
by modifying `landscapes_min.py`.

Note that if the `-max_feval=max_fevals` flag is specified, all runs will terminate once the user-provided number of
__unique__ function calls, `max_fevals`, has been reached. The total number of steps `ltot_val` should then
be set to a large value in the `-ltot=ltot_val` flag, so that the code does not terminate prematurely.

If the `-init_each={T,F}` flag is specified, each individual run with a given set of input parameter values will
start from a randomly chosen state. If the `-init=\(s1,s2,...\)` flag is specified ($(s_1,s_2,\dots)$
must correspond to a valid state on the fitness landscape), the `-init_each` flag will be ignored and all runs will start from the user-provided state.

The sets of randomly generated EA/SK and NK model parameters are stored in auxiliary files. The files are output
using `-Jout=Jij_out.dat` and `-NKout=NK_out.dat` flags respectively, where the `*.dat` are the output filenames. Note that
for the NK model, two files are actually generated: `NK_out.dat.1` and `NK_out.dat.2`. For the subsequent runs in the
quenched disorder mode, the model parameter files must be read in using the `-Jin=Jij_in.dat` flag for the EA/SK model and the `-NKin=NK_in.dat` flag for the NK model, otherwise the model parameters will be generated *de novo*. For all models, multiple runs with the same parameter settings can be specified using the `-nruns=nr_val` option.

The `-discrete={T,F}` flag is only valid for the EA/SK model and refers to the probabilistic model for spin couplings.
When `-discrete=T`, the couplings are drawn from $\{−1,+1\}$ with equal probability. Otherwise, the couplings are sampled
from a standard normal distribution.

The `-schedule={linear, exponential}` refers to the Simulated Annealing schedule. The default is 'linear'.

The `-Smax` flag in the TabuSearch mode refers to the maximum allowed fitness value. The run will terminate once
this value is exceeded. For this option to have no effect, it should be set to a large positive number, e.g.
`-Smax=10000.0`.

If the `-straj={0,1}` is set to $1$, a fitness trajectory will be output in a separate file for each run in the series. All filenames will be created automatically based on the user-provided output filename.

If the `-socc={0,1}` is set to $1$, a sorted list of state occupancies will be output in a separate file
for each run in the series. As with the fitness trajectories, all filenames will be created automatically based on the user-provided output filename. Both `-straj` and `-socc` options are off by default.

Finally, the `-sopt={1,2}` flag is only relevant in the SmartRunner mode. It is an advanced option having to
do with how SmartRunner avoids being trapped in a fully explored region. Since both options produce nearly identical results, the default setting, `-sopt=2`, should suffice for most users.

<br/><br/>

The main output file of `run_optimizers.py` should look like this:
```
# amode = SmartRunner, mode = NK200.8, moveset_mode = single_spin_flip
# 1x5x2x2 array, D = 200
# l_max = 2, nruns = 2, sopt = 2, straj = 0
ltot         Opt          dF           Fbest         feval         Sbest
1500000    1.0000e-04    1.0000e-03    7.7250e-01    406617    (-1,-1,-1,1,-1,1,-1,-1,1,1,-1,1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,1)
1500000    1.0000e-04    1.0000e-03    7.6540e-01    401853    (1,-1,-1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,-1,1,1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,1)
1500000    1.0000e-04    1.0000e-02    7.7240e-01    388342    (1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,1,-1,-1,1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,1,1,1,1)
1500000    1.0000e-04    1.0000e-02    7.6210e-01    400941    (-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,1,-1,1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1)
1500000    1.0000e-03    1.0000e-03    7.7180e-01    702217    (-1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1)
1500000    1.0000e-03    1.0000e-03    7.8710e-01    681292    (-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,1,-1,-1,1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,-1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,1,-1,1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,1,1,1,-1,1,1,-1,-1)
1500000    1.0000e-03    1.0000e-02    7.7590e-01    684432    (-1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,1,-1,1,-1,1,1,-1,-1,-1,1,-1)
1500000    1.0000e-03    1.0000e-02    7.8540e-01    672067    (-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,1,1,-1,1,1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1)
1500000    1.0000e-02    1.0000e-03    7.7470e-01    1141421    (-1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1)
1500000    1.0000e-02    1.0000e-03    7.8040e-01    1100435    (-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,1,-1,1,1,-1,-1,-1,1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,1,-1,1,-1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,1,-1)
1500000    1.0000e-02    1.0000e-02    7.7980e-01    1125987    (-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,1,1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1)
1500000    1.0000e-02    1.0000e-02    7.8160e-01    1144962    (1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,-1,1,1,-1,-1,1,1,1,1,1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,1,1,1,-1,1,-1,1,1,1)
1500000    1.0000e-01    1.0000e-03    7.9080e-01    1415809    (-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,-1,1)
1500000    1.0000e-01    1.0000e-03    7.9370e-01    1414682    (1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,1,-1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1)
1500000    1.0000e-01    1.0000e-02    7.9150e-01    1414889    (-1,1,1,1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,1,1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1)
1500000    1.0000e-01    1.0000e-02    7.9200e-01    1415424    (1,1,1,1,1,-1,1,1,1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,-1,1,1,1,-1,1,-1)
1500000    1.0000e+00    1.0000e-03    7.4400e-01    1474413    (-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,1,1)
1500000    1.0000e+00    1.0000e-03    7.4220e-01    1474225    (1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,-1,-1,1,-1,-1,1,1,-1)
1500000    1.0000e+00    1.0000e-02    7.2930e-01    1474068    (-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,-1,1,1)
1500000    1.0000e+00    1.0000e-02    7.3170e-01    1474566    (-1,-1,-1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,-1,1,1,1,-1,1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,1,-1,1,1,1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1,1,-1,1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1)
```

This is a series of SmartRunner runs for the NK model with $200$ sites and $8$ neighbors per site.
Each random move is a single "spin flip". The SmartRunner was run with $l_\mathrm{max} = 2$ and
$s_\mathrm{opt} = 2$. The fitness trajectory files were not requested. The user-provided input parameter ranges
resulted in a series of $1 \times 5 \times 2 \times 2 = 20$ runs. For each run, the best fitness (`Fbest`)
and the corresponding system state (`Sbest`) are output (note that $\{-1,1\} \to \{0,1\}$),
along with the total number of unique function evaluations (`feval`). The input parameters were `ltot` (the
total number of steps), `Opt` (optimism $\alpha$), and `dF` (the initial guess
for the expected rate of fitness gain per step $\bar{R}_\mathrm{init}$).

The output file was generated with the following command:
```
python3 run_optimizers.py -amode=SmartRunner -ltot=1500000 -DeltaF_rate_mean=\[0.001,0.01,2,\"log10\"\] -optimism=\[0.0001,1.0,5,\"log10\"\] -mode=NK200.8 -moveset_mode=single_spin_flip -nruns=2 -out=sr.dat -straj=0 -socc=0 -init_each=T -l_max=2 -NKin=NK200.8.dat
```

All runs used a previously generated set of NK model parameters stored in the `NK200.8.dat.1` and `NK200.8.dat.2` files. Two independent runs were carried out for each of the $10$ unique input parameter combinations. Each of the $20$ runs started from a randomly generated state.

<br/><br/>                                                                             
                                                                               
Finally, `utilities_min.py` contains several functions designed to work with the `run_optimizers.py` output.

* <span style="color:green"><font size="3"> __read_data_generic(*filename*)__ (reads a single main output file into Numpy matrices) </font></span>

* <span style="color:green"><font size="3"> __read_multiple_files(*filenames*)__ (reads multiple main output files and concatenates the data; typically used to combine output of multiple parallel runs) </font></span>

* <span style="color:green"><font size="3"> __read_fitness_data(*datafile*)__ (reads a single output file with a fitness trajectory) </font></span>

* <span style="color:green"><font size="3"> __read_occ_data(*datafile*)__ (reads a single output file with state occupancies) </font></span>

<br/><br/>

The `examples/` folder contains five representative runs that feature each global optimization algorithm and each fitness landscape once. These runs are provided only as guidance and do not contain any actual results reported in the SmartRunner paper. The `commands.txt` file has the commands used to generate the output files in the `examples/` folder.



```python

```
