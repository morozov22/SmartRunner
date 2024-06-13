#######################
#  run_optimizers.py  #
#######################

from utilities_min import *
from SmartRunner import *
from evol_algorithm import *
from simulated_annealing import *
from tabu_search import *
from stochastic_hill_climbing import *
from extremal_optimization import *
from time import perf_counter

# Initial time in ns:
time_i = int(perf_counter())

# Get command line args:
args = parse_command_line_generic()


amode = args['amode'] # EvolutionaryAlgorithm/SimulatedAnnealing/TabuSearch/StochasticHillClimbing/SmartRunner/ExtremalOptimization


if amode == "EvolutionaryAlgorithm":

	# Process 'array coordinates':
	mu_rate = postprocess_val('mu_rate',float,args)
	x_rate = postprocess_val('x_rate',float,args)
	Rbar = postprocess_val('Rbar',float,args)

	prm1 = mu_rate
	prm2 = x_rate
	prm3 = Rbar
	prm_labels = ["mu_rate","x_rate","Rbar"]

	# Process 'non-array coordinates':
	# Population size:
	Npop = args['Npop']
	assert type(Npop) is int and Npop > 0, "Error: provide a positive integer for Npop!"

	# Total number of moves:
	ltot = args['ltot']
	assert type(ltot) is int and ltot > 0, "Error: provide a positive integer for ltot!"

elif amode == "SimulatedAnnealing":

	# Process 'array coordinates':
	Ti = postprocess_val('Ti',float,args)
	Tf = postprocess_val('Tf',float,args)
	Rbar = postprocess_val('Rbar',float,args)

	prm1 = Ti
	prm2 = Tf
	prm3 = Rbar
	prm_labels = ["Ti","Tf","Rbar"]

	# Process 'non-array coordinates':

	# Total number of moves:
	ltot = args['ltot']
	assert type(ltot) is int and ltot > 0, "Error: provide a positive integer for ltot!"

	# Schedule type:
	if 'schedule' in args:
		schedule_type = args['schedule']
		assert schedule_type == "linear" or schedule_type == "exponential", \
			"Error: illegal schedule_type = {}!".format(schedule_type)
	else:
		schedule_type = "linear"

elif amode == "TabuSearch":

	# Process 'array coordinates':
	ltot = postprocess_val('ltot',int,args)
	tabu_size = postprocess_val('tabu',int,args)
	Smax = postprocess_val('Smax',float,args)

	prm1 = ltot
	prm2 = tabu_size
	prm3 = Smax
	prm_labels = ["ltot","tabu_size","Smax"]

	# Process 'non-array coordinates':

elif amode == "StochasticHillClimbing":

	# Process 'array coordinates':
	ltot = postprocess_val('ltot',int,args)
	T = postprocess_val('T',float,args)
	Rbar = postprocess_val('Rbar',float,args)

	prm1 = ltot
	prm2 = T
	prm3 = Rbar
	prm_labels = ["ltot","T","Rbar"]

	# Process 'non-array coordinates':

elif amode == "SmartRunner":

	# Process 'array coordinates':
	ltot = postprocess_val('ltot',int,args)
	optimism = postprocess_val('optimism',float,args)
	DeltaF_rate_mean = postprocess_val('DeltaF_rate_mean',float,args)

	prm1 = ltot
	prm2 = optimism
	prm3 = DeltaF_rate_mean
	prm_labels = ["ltot","optimism","DeltaF_rate_mean"]

	# Process 'non-array coordinates':
	if 'l_max' in args:
		l_max = args['l_max']
		assert type(l_max) is int and l_max > 0, "Error: provide a positive integer for l_max!"
	else:
		l_max = 3

	if 'sopt' in args:
		sopt = args['sopt']
		assert type(sopt) is int and (sopt == 1 or sopt == 2), "Error: provide a {1,2} integer for sopt!"
	else:
		sopt = 2

elif amode == "ExtremalOptimization":

	# Process 'array coordinates':
	ltot = postprocess_val('ltot',int,args)
	tau = postprocess_val('tau',float,args)
	Rbar = postprocess_val('Rbar',float,args)

	if len(Rbar) > 1 or Rbar[0] != 0.0:
		print("WARNING: expected Rbar=0.0 in amode = {}!".format(amode))
		print("Resetting ..")
		Rbar = [0.0]

	prm1 = ltot
	prm2 = tau
	prm3 = Rbar
	prm_labels = ["ltot","tau","Rbar"]

else:
	print("Unknown amode = {}!".format(amode))
	sys.exit("\nTerminating ..")

l1 = len(prm1)
l2 = len(prm2)
l3 = len(prm3)


# Process 'non-array' coordinates (amode-independent):
if 'nruns' in args:
	nruns = args['nruns']
	assert type(nruns) is int and nruns > 0, "Error: provide a positive integer for nruns!"
else:
	nruns = 20

# Set up auxiliary parameters of the landscape and move set:

# Landscape type and moveset modes:
mode = args['mode'] # SKZZZ/EAXxY/NKXX.YY/Rastrigin_4D/Ackley_4D/Griewank_4D/double_well_2D
moveset_mode = args['moveset_mode'] # nnb/spmut/single_spin_flip

if amode == "ExtremalOptimization":
	if mode[:2] != "SK" and mode[:2] != "EA":
		print("mode = {} is not supported in amode = {}!".format(mode,amode))
		sys.exit("\nTerminating ..")
	####
	if moveset_mode != "single_spin_flip":
		print("moveset_mode = {} is not supported in amode = {}!".format(moveset_mode,amode))
		sys.exit("\nTerminating ..")

# Output file:
fout = args['out']

# Save trajectory option:
if 'straj' in args:
	straj = args['straj']
	assert type(straj) is int and (straj == 0 or straj == 1), "Error: provide a {0,1} integer for straj!"
else:
	straj = 0 # off by default

# Save occupancy option:
if 'socc' in args:
	socc = args['socc']
	assert type(socc) is int and (socc == 0 or socc == 1), "Error: provide a {0,1} integer for socc!"
else:
	socc = 0 # off by default

# Max number of fevals:
if 'max_feval' in args:
	max_feval = args['max_feval']
	assert type(max_feval) is int and max_feval > 0, "Error: provide a positive integer for max_feval!"
else:
	max_feval = None

######
D,N = get_DN(mode,moveset_mode)
ranges = get_ranges(D,mode)
Delta_x = get_Delta_x(mode)

if mode[:2] == "SK" or mode[:2] == "EA" or mode[:2] == "NK":
	rparams = None
else:
	rparams = (ranges,Delta_x,)

# Set up mode-dependent parameters:
if mode[:2] == "SK" or mode[:2] == "EA":
	if mode[:2] == "SK":
		Nspins = int(mode[2:])
	elif mode[:2] == "EA":
		toks = mode[2:].split("x")
		assert len(toks) == 2, "Error: unknown EA mode format: {}!".format(mode)
		Nx = int(toks[0])
		Ny = int(toks[1])
		Nspins = Nx * Ny
	######
	assert Nspins == D, "Error: mismatch between Nspins={} and D={}!".format(Nspins,D)
	# Sampling mode for Jij couplings:
	if 'discrete' in args:
		assert args['discrete'] == "True" or args['discrete'] == "T" or \
				args['discrete'] == "False" or args['discrete'] == "F", \
				"Error: invalid -discrete value: {}!".format(args['discrete'])
		######
		if args['discrete'] == "True" or args['discrete'] == "T":
			discr = True
		else:
			discr = False
	else:
		discr = False # False by default
	####
	if 'Jin' in args:
		if 'discrete' in args:
			print("WARNING: -discrete option is ignored because -Jin option is provided!")
		####
		Jin = args['Jin']
		Jij_set = np.loadtxt(Jin, dtype=np.float64)
		assert Jij_set.shape[0] == Nspins and Jij_set.shape[1] == Nspins, "Error: Jij matrix dim mismatch!"
	else:
		if mode[:2] == "SK":
			Jij_set = Jij(Nspins,discrete=discr) # generate spin couplings *once*
		elif mode[:2] == "EA":
			Jij_set = Jij_EA(Nx,Ny,discrete=discr) # generate spin couplings *once*
	######
	if 'Jout' in args:
		Jout = args['Jout']
		np.savetxt(Jout, Jij_set, delimiter=' ', fmt='%12.7f')
	######
	params = (Jij_set,)
	mparams = None
elif mode[:2] == "NK":
	toks = mode[2:].split(".")
	assert len(toks) == 2, "Error: unknown NK mode format: {}!".format(mode)
	Nsites = int(toks[0])
	Nnb = int(toks[1])
	assert Nsites == D, "Error: mismatch between Nsites={} and D={}!".format(Nsites,D)
	if 'NKin' in args:
		NKin = args['NKin']
		NKin1 = NKin + '.1'
		NKin2 = NKin + '.2'
		nb_ind = np.loadtxt(NKin1, dtype=np.int64)
		Fvals = np.loadtxt(NKin2, dtype=np.float64)
		assert nb_ind.shape[0] == Nsites and nb_ind.shape[1] == Nnb+1, "Error: nb_ind matrix dim mismatch!"
		assert Fvals.shape[0] == Nsites and Fvals.shape[1] == 2**(Nnb+1), "Error: Fvals matrix dim mismatch!"
	else:
		nb_ind,Fvals = NKprms(Nsites,Nnb) # generate NK model prms *once*
	######
	if 'NKout' in args:
		NKout = args['NKout']
		NKout1 = NKout + '.1'
		NKout2 = NKout + '.2'
		np.savetxt(NKout1, nb_ind, delimiter=' ', fmt='%12d')
		np.savetxt(NKout2, Fvals, delimiter=' ', fmt='%12.7f')
	######
	params = (nb_ind,Fvals,)
	mparams = None
else:
	params = None
	mparams = (Delta_x,ranges,None,None,)

############################################################

# Set up the initial state:
if 'init' in args:
	crd_init_tuple = tuple(args['init'])
	if amode == "EvolutionaryAlgorithm":
		print("WARNING: -init option is ignored in amode = {}!".format(amode))
else:
	if amode == "EvolutionaryAlgorithm":
		crd_init_tuple = None
	else:
		crd_init_tuple = generate_random_state(D,mode,rparams) # NOT used in the EvolutionaryAlgorithm mode

if 'init_each' in args:
	assert args['init_each'] == "True" or args['init_each'] == "T" or \
			args['init_each'] == "False" or args['init_each'] == "F", \
			"Error: invalid -init_each value: {}!".format(args['init_each'])

prm1_glob = []
prm2_glob = []
prm3_glob = []

fvals_glob = []
Fbest_glob = []
state_glob = []

run_cnt = 1
for m in range(l1):

	print("{}. {} = {}".format(m+1,prm_labels[0],prm1[m]))

	for k in range(l2):
	
		#print("  {}. {} = {:.2f}".format(k+1,prm_labels[1],prm2[k]))
		print("  {}. {} = {}".format(k+1,prm_labels[1],prm2[k]))

		for i in range(l3):
				    
			#print("     {}. {} = {:.2f}".format(i+1,prm_labels[2],prm3[i]))
			print("     {}. {} = {}".format(i+1,prm_labels[2],prm3[i]))

			print("       Starting {} {} run(s) ..".format(nruns,amode))

			step = 10

			for j in range(nruns):

				if j > 0 and j%step==0:
					print("        run {} ..".format(j))

				# Generate random initial state for each individual run:
				if 'init_each' in args and (args['init_each'] == "T" or args['init_each'] == "True"):
					if amode == "EvolutionaryAlgorithm":
						print("WARNING: -init_each option is ignored in amode = {}!".format(amode))
					else:
						if 'init' in args:
							print("WARNING: -init_each option is ignored because -init option is provided!")
						else:
							crd_init_tuple = generate_random_state(D,mode,rparams)

				### Main optimizer function ###
				if amode == "EvolutionaryAlgorithm":
					# crossover_rate, mutation_rate, max_steps, params, mparams, rparams, mode, moveset_mode, Npop, dim, Rbar, max_fitness=None
					algorithm = Algorithm_EA(x_rate[k], mu_rate[m], ltot, params, mparams, rparams, \
												mode, moveset_mode, Npop, D, Rbar[i], max_fitness=None, max_feval=max_feval)
				
					[fvals,Fglobal_best,crd_global_best_tuple,node_occ,Ftraj,traj_states] = algorithm.run(verbose=False)
				elif amode == "SimulatedAnnealing":
					algorithm = Algorithm_SA(Ti[m], Tf[k], ltot, params, mparams, \
											mode, moveset_mode, crd_init_tuple, Rbar[i], \
											min_energy=None, max_feval=max_feval, schedule=schedule_type)
				
					[fvals,Fglobal_best,crd_global_best_tuple,node_occ,Ftraj,traj_states] = algorithm.run(verbose=False)
				elif amode == "TabuSearch":
					algorithm = Algorithm_TB(crd_init_tuple, tabu_size[k], ltot[m], params, mparams, \
												mode, moveset_mode, max_score=Smax[i], max_feval=max_feval)
				
					[fvals,Fglobal_best,crd_global_best_tuple,node_occ,Ftraj,traj_states] = algorithm.run(verbose=False)
				elif amode == "StochasticHillClimbing":
					algorithm = Algorithm_SHC(crd_init_tuple, T[k], ltot[m], params, mparams, \
												mode, moveset_mode, Rbar[i], max_objective=None, max_feval=max_feval)
				
					[fvals,Fglobal_best,crd_global_best_tuple,node_occ,Ftraj,traj_states] = algorithm.run(verbose=False)
				elif amode == "SmartRunner":
					#crd_init_tuple = generate_random_state(D,mode,rparams)
					### Main function of the SmartRunner algorithm ###
					[fvals,Fglobal_best,crd_global_best_tuple,node_occ,Ftraj,traj_states] = \
						SmartRunner(crd_init_tuple,N,ltot[m],ranges,Delta_x,DeltaF_rate_mean[i],optimism[k], \
									get_fitness,make_move,l_max,mode,moveset_mode,params,mparams,sopt=sopt,max_feval=max_feval)
				elif amode == "ExtremalOptimization":
					[fvals,Fglobal_best,crd_global_best_tuple,node_occ,Ftraj,traj_states] = \
						ExtremalOpt(crd_init_tuple, tau[k], ltot[m], params, mode, max_feval=max_feval)
				else:
					print("Unknown amode = {}!".format(amode))
					sys.exit("\nTerminating ..")

				prm1_glob.append(prm1[m])
				prm2_glob.append(prm2[k])
				prm3_glob.append(prm3[i])

				state_glob.append(str(crd_global_best_tuple).replace(' ', ''))
				fvals_glob.append(int(fvals))
				Fbest_glob.append(float(Fglobal_best))

				# Output fitness trajectories:
				if straj == 1:
					ftraj_cur = fout + '.f.' + str(run_cnt)
					####
					comment_str = "# run = {}\n".format(run_cnt)
					header_arr = ['ltot','F','S']
					#ltot_arr = list(range(1,len(Ftraj)+1)) # algorithms may terminate early sometimes ..
					ltot_arr = list(range(len(Ftraj))) # algorithms may terminate early sometimes ..
					
					for j in range(len(Ftraj)):
						Ftraj[j] = float(Ftraj[j]) # cast from numpy.float64 to float
					###
					for j in range(len(traj_states)):
						traj_states[j] = str(traj_states[j]).replace(' ', '')
					data_arr = [ltot_arr,Ftraj,traj_states]
					write_data(ftraj_cur,comment_str,header_arr,data_arr)

				# Output node occupancies, sorted by occ:
				if socc == 1:
					focc_cur = fout + '.occ.' + str(run_cnt)
					####
					comment_str = "# run = {}\n".format(run_cnt)
					header_arr = ['occ','S']
					node_arr = []
					occ_arr = []
					node_occ_sorted = sorted(node_occ.items(), key=lambda item: item[1])
					for x in node_occ_sorted:
						node_arr.append(str(x[0]).replace(' ', ''))
						occ_arr.append(int(x[1]))

					data_arr = [occ_arr,node_arr]
					write_data(focc_cur,comment_str,header_arr,data_arr)

				run_cnt += 1

	print("\nElapsed time =",convert(int(perf_counter())-time_i))
	print("\n")


print("=== Finished {} {} run(s) .. ===".format(run_cnt-1,amode))

#############
if amode == "EvolutionaryAlgorithm":
	comment_str = "# amode = {}, mode = {}, moveset_mode = {}\n# {}x{}x{}x{} array, D = {}\n# nruns = {}, ltot = {}, straj = {}, Npop = {}\n". \
				format(amode,mode,moveset_mode,l1,l2,l3,nruns,D,nruns,ltot,straj,Npop)

	header_arr = ['Mu',' X','  Rbar',' Fbest',' feval','Sbest']
elif amode == "SimulatedAnnealing":
	comment_str = "# amode = {}, mode = {}, moveset_mode = {}\n# {}x{}x{}x{} array, D = {}\n# nruns = {}, ltot = {}, straj = {}, schedule = {}\n". \
				format(amode,mode,moveset_mode,l1,l2,l3,nruns,D,nruns,ltot,straj,schedule_type)

	header_arr = ['Ti',' Tf','  Rbar',' Fbest',' feval','Sbest']
elif amode == "TabuSearch":
	comment_str = "# amode = {}, mode = {}, moveset_mode = {}\n# {}x{}x{}x{} array, D = {}\n# nruns = {}, straj = {}\n". \
				format(amode,mode,moveset_mode,l1,l2,l3,nruns,D,nruns,straj)

	header_arr = ['ltot','Tsize','Smax','Fbest','feval','Sbest']
elif amode == "StochasticHillClimbing":
	comment_str = "# amode = {}, mode = {}, moveset_mode = {}\n# {}x{}x{}x{} array, D = {}\n# nruns = {}, straj = {}\n". \
				format(amode,mode,moveset_mode,l1,l2,l3,nruns,D,nruns,straj)

	header_arr = ['ltot','    T','Rbar','Fbest','feval','Sbest']
elif amode == "SmartRunner":
	comment_str = "# amode = {}, mode = {}, moveset_mode = {}\n# {}x{}x{}x{} array, D = {}\n# l_max = {}, nruns = {}, sopt = {}, straj = {}\n". \
				format(amode,mode,moveset_mode,l1,l2,l3,nruns,D,l_max,nruns,sopt,straj)

	header_arr = ['ltot','Opt','dF','Fbest','feval','Sbest']
elif amode == "ExtremalOptimization":
	comment_str = "# amode = {}, mode = {}, moveset_mode = {}\n# {}x{}x{}x{} array, D = {}\n# nruns = {}, straj = {}\n". \
				format(amode,mode,moveset_mode,l1,l2,l3,nruns,D,nruns,straj)

	header_arr = ['ltot','  tau','Rbar','Fbest','feval','Sbest']
else:
	print("Unknown amode = {}!".format(amode))
	sys.exit("\nTerminating ..")

data_arr = [prm1_glob,prm2_glob,prm3_glob,Fbest_glob,fvals_glob,state_glob]

write_data(fout,comment_str,header_arr,data_arr)


# Final time in ns:
time_f = int(perf_counter())
print("\nTotal elapsed time =",convert(time_f-time_i))

