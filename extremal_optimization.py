#############################
### Extremal Optimization ###
#############################

from landscapes_min import *
from scipy.stats import powerlaw
#from SmartRunner import get_l_extra_approx

###########################################
# Top-level function, ExtremalOptimization
###########################################

def ExtremalOpt(crd_init_tuple, tau, ltot, params, mode, max_feval=None):
	
	assert tau > 0, "Error: provide a positive tau in ExtremalOpt!"

	Ftraj = [] # node fitness at each step of the run
	traj_states = [] # corresponding node coordinates/states

    # Build a global node_occ hash:
	node_trials = {}

    # Build a global node_F hash:
	node_F = {}
    
	fvals = 0 # number of unique function evaluations
    
	crd_cur_tuple = crd_init_tuple
	Nspins = len(crd_cur_tuple)

	if mode[:2] == "SK":
		Fcur = H_SK(crd_cur_tuple,params[0]) # total fitness value
	elif mode[:2] == "EA":
		Fcur = H_EA(crd_cur_tuple,params[0]) # total fitness value
	else:
		print("Unknown mode = {} in ExtremalOpt!".format(mode))
		sys.exit("\nTerminating ..")

	node_trials[crd_cur_tuple] = 0
	node_F[crd_cur_tuple] = Fcur
	fvals += 1

	Ftraj.append(Fcur)
	traj_states.append(crd_cur_tuple)

	# Initialize best state/fitness for the entire run:
	Fglobal_best = Fcur
	crd_global_best_tuple = crd_cur_tuple

	l = 0
	while l < ltot:

		######
		if mode[:2] == "SK":
			s_en = H_SK_site(crd_cur_tuple,params[0]) # single-spin fitness values
		elif mode[:2] == "EA":
			s_en = H_EA_site(crd_cur_tuple,params[0]) # single-spin fitness values
		else:
			print("Unknown mode = {} in ExtremalOpt!".format(mode))
			sys.exit("\nTerminating ..")

		s_en_ind = np.argsort(s_en) # indices of the sorted array ([a_1 <= a_2 <= .. <= a_N])

		# Choose a sorted index randomly acc. to the power-law distribution:
		r = powerlaw.rvs(tau, size=1) # random number in the [0,1] range
		chosen_ind = int(r*Nspins)
		if chosen_ind == Nspins: # boundary special case when r=1.0
			chosen_ind -= 1

		#debug
		#print("chosen_ind =",chosen_ind)
		#print("chosen Es =",s_en[s_en_ind[chosen_ind]],"; <Es> =",np.mean(s_en))

		crd_cur_tuple = make_spin_flip(crd_cur_tuple, ind=s_en_ind[chosen_ind])

		if crd_cur_tuple in node_F:
			Fcur = node_F[crd_cur_tuple]
			node_trials[crd_cur_tuple] += 1
		else:
			if mode[:2] == "SK":
				Fcur = H_SK(crd_cur_tuple,params[0]) # total fitness value
			elif mode[:2] == "EA":
				Fcur = H_EA(crd_cur_tuple,params[0]) # total fitness value
			else:
				print("Unknown mode = {} in ExtremalOpt!".format(mode))
				sys.exit("\nTerminating ..")

			node_trials[crd_cur_tuple] = 0
			node_F[crd_cur_tuple] = Fcur
			fvals += 1

		Ftraj.append(Fcur)
		traj_states.append(crd_cur_tuple)

		#### Update best state/fitness for the entire run: ####
		if Fcur > Fglobal_best:
		    Fglobal_best = Fcur
		    crd_global_best_tuple = crd_cur_tuple

		if max_feval is not None and max_feval == fvals:
		    print("TERMINATING - REACHED MAX FEVALS")
		    return [fvals,Fglobal_best,crd_global_best_tuple,node_trials,Ftraj,traj_states]

		if l > 0 and l%50000 == 0:
		    print("l = {} ..".format(l))

		l += 1 # global move counter

	print("TERMINATING - REACHED MAXIMUM STEPS")
	return [fvals,Fglobal_best,crd_global_best_tuple,node_trials,Ftraj,traj_states]

