########################
### utilities_min.py ###
########################

from math import *
import numpy as np
import sys
import re

#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd

#import glob
#import os


# This function parses run_optimizers command line.
def parse_command_line_generic():

    args = {}

    argumentList = sys.argv[1:]

    if len(argumentList) == 0:
        print("No input arguments provided, please use '-h -amode=algorithm_name' or '--help -amode=algorithm_name' for help!")
        print("Allowed algorithm_name values: EvolutionaryAlgorithm/SimulatedAnnealing/TabuSearch/StochasticHillClimbing/SmartRunner")
        sys.exit("\nTerminating ..")
    else:

        for arg in argumentList:

            if arg == '-h' or arg == '--help':
                continue

            toks = arg.split("=")
            assert len(toks) == 2, "Error: malformed token(s) in parse_command_line_generic!"
            assert type(toks[0]) is str and toks[0][0] == '-', "Error: malformed token(s) in parse_command_line_generic!"
            toks[0] = toks[0][1:] # get rid of the leading '-'

            if re.search('[a-zA-Z]', toks[1]) and not re.search('^[([]', toks[1]): # string with non-numeric entries, NOT an array
                args[toks[0]] = toks[1]
            else:
                try:
                    tok_val = eval(toks[1])
                except:
                    raise Exception("Error: malformed list token: {} in parse_command_line_generic!".format(toks[1]))
                ####
                if toks[0] == "init":
                    args[toks[0]] = tuple(tok_val)
                else:
                    if type(tok_val) is list or type(tok_val) is tuple:
                        if len(tok_val) == 3 or len(tok_val) == 4: # start,stop,num_vals
                            # Check stop > start:
                            assert tok_val[1] >= tok_val[0], \
                                "Error: start < stop in parse_command_line_generic!"
                            # Check num_vals:
                            assert type(tok_val[2]) is int and tok_val[2] > 0, \
                                "Error: provide a non-negative integer for num_vals in parse_command_line_generic!"
                        ######
                        if len(tok_val) == 3: # start,stop,num_vals
                            ####
                            if tok_val[2] == 1:
                                if tok_val[1] == tok_val[0]:
                                    if type(tok_val[0]) is int and type(tok_val[1]) is int:
                                        tok_arr = [int(tok_val[0])]
                                    else:
                                        tok_arr = [float(tok_val[0])]
                                else:
                                    raise Exception("Unable to create an array with start={}, stop={}, num_vals={} in parse_command_line_generic!".\
                                        format(tok_val[0],tok_val[1],tok_val[2]))
                            else:
                                tok_arr = list(np.linspace(tok_val[0], tok_val[1], num=tok_val[2], dtype=np.float64))
                                step = (tok_arr[-1] - tok_arr[0])/(len(tok_arr)-1)
                                ####
                                if type(tok_val[0]) is int and type(tok_val[1]) is int and abs(step - int(step)) < 1e-10: # cast array to int
                                    for i in range(len(tok_arr)):
                                        tok_arr[i] = int(tok_arr[i])
                                else: # cast array to float
                                    for i in range(len(tok_arr)):
                                        tok_arr[i] = float(tok_arr[i])
                                
                        elif len(tok_val) == 4: # start,stop,num_vals,log_token
                            if type(tok_val[3]) == str and tok_val[3] == 'log10':
                                tok_arr = list(np.linspace(np.log10(tok_val[0]), np.log10(tok_val[1]), \
                                        num=tok_val[2], dtype=np.float64))
                                for i in range(len(tok_arr)):
                                    tok_arr[i] = float(10**tok_arr[i])
                            else:
                                raise Exception("Unknown log_token={} in parse_command_line_generic!".format(tok_val[3]))
                        else:
                            raise Exception("Expected 3 or 4 entries in {}!".format(toks[1]))
                        ######
                        args[toks[0]] = tok_arr
                    else: # int,float
                        args[toks[0]] = tok_val


        if '-h' in argumentList or '--help' in argumentList:

            if not 'amode' in args:
                print("Please provide -amode=algorithm_name with -h or --help for algorithm-specific help!")
                print("Allowed algorithm_name values: EvolutionaryAlgorithm/SimulatedAnnealing/TabuSearch/StochasticHillClimbing/SmartRunner")
                sys.exit("\nTerminating ..")
            else:
                amode = args['amode']

                if amode == "EvolutionaryAlgorithm":
                    print("python3 {} -amode=EvolutionaryAlgorithm [-mu_rate=mu_val OR -mu_rate=(mu_min,mu_max,mu_num[,\"log10\"]) OR -mu_rate=[mu_min,mu_max,mu_num[,\"log10\"]]] \\".format(sys.argv[0]))
                    print("[-x_rate=x_val OR -x_rate=(x_min,x_max,x_num[,\"log10\"]) OR -x_rate=[x_min,x_max,x_num[,\"log10\"]]] \\")
                    print("[-Rbar=R_val OR -Rbar=(R_min,R_max,R_num[,\"log10\"]) OR -Rbar=[R_min,R_max,R_num[,\"log10\"]]] \\")
                    print("-ltot=ltot_val -Npop=Npop_val -mode=landscape_type -moveset_mode=moveset_type -out=outfilename \\")
                    print("[-max_feval=max_fevals -nruns=nr_val -straj={0,1} -socc={0,1}] \\")
                    print("[-Jin=Jij_in.dat -Jout=Jij_out.dat -discrete={T,F}] \\")
                    print("[-NKin=NK_in.dat -NKout=NK_out.dat]")
                elif amode == "SimulatedAnnealing":
                    print("python3 {} -amode=SimulatedAnnealing [-Ti=Ti_val OR -Ti=(Ti_min,Ti_max,Ti_num[,\"log10\"]) OR -Ti=[Ti_min,Ti_max,Ti_num[,\"log10\"]]] \\".format(sys.argv[0]))
                    print("[-Tf=Tf_val OR -Tf=(Tf_min,Tf_max,Tf_num[,\"log10\"]) OR -Tf=[Tf_min,Tf_max,Tf_num[,\"log10\"]]] \\")
                    print("[-Rbar=R_val OR -Rbar=(R_min,R_max,R_num) OR -Rbar=[R_min,R_max,R_num]] \\")
                    print("-ltot=ltot_val -mode=landscape_type -moveset_mode=moveset_type -out=outfilename \\")
                    print("[-max_feval=max_fevals -schedule=schedule_type -nruns=nr_val -straj={0,1} -socc={0,1}] \\")
                    print("[[-init=(s1,s2,...) OR -init=[s1,s2,...]] -init_each={T,F}] \\")
                    print("[-Jin=Jij_in.dat -Jout=Jij_out.dat -discrete={T,F}] \\")
                    print("[-NKin=NK_in.dat -NKout=NK_out.dat]")
                elif amode == "TabuSearch":
                    print("python3 {} -amode=TabuSearch [-ltot=ltot_val OR -ltot=(ltot_min,ltot_max,ltot_num[,\"log10\"]) OR -ltot=[ltot_min,ltot_max,ltot_num[,\"log10\"]]] \\".format(sys.argv[0]))
                    print("[-tabu=tabu_val OR -tabu=(tabu_min,tabu_max,tabu_num[,\"log10\"]) OR -tabu=[tabu_min,tabu_max,tabu_num[,\"log10\"]]] \\")
                    print("[-Smax=Smax_val OR -Smax=(Smax_min,Smax_max,Smax_num) OR -Smax=[Smax_min,Smax_max,Smax_num]] \\")
                    print("-mode=landscape_type -moveset_mode=moveset_type -out=outfilename \\")
                    print("[-max_feval=max_fevals -nruns=nr_val -straj={0,1} -socc={0,1}] \\")
                    print("[[-init=(s1,s2,...) OR -init=[s1,s2,...]] -init_each={T,F}] \\")
                    print("[-Jin=Jij_in.dat -Jout=Jij_out.dat -discrete={T,F}] \\")
                    print("[-NKin=NK_in.dat -NKout=NK_out.dat]")
                elif amode == "StochasticHillClimbing":
                    print("python3 {} -amode=StochasticHillClimbing [-ltot=ltot_val OR -ltot=(ltot_min,ltot_max,ltot_num[,\"log10\"]) OR -ltot=[ltot_min,ltot_max,ltot_num[,\"log10\"]]] \\".format(sys.argv[0]))
                    print("[-T=T_val OR -T=(T_min,T_max,T_num[,\"log10\"]) OR -T=[T_min,T_max,T_num[,\"log10\"]]] \\")
                    print("[-Rbar=R_val OR -Rbar=(R_min,R_max,R_num[,\"log10\"]) OR -Rbar=[R_min,R_max,R_num[,\"log10\"]]] \\")
                    #print("[-Smax=Smax_val OR -Smax=(Smax_min,Smax_max,Smax_num) OR -Smax=[Smax_min,Smax_max,Smax_num]] \\")
                    print("-mode=landscape_type -moveset_mode=moveset_type -out=outfilename \\")
                    print("[-max_feval=max_fevals -nruns=nr_val -straj={0,1} -socc={0,1}] \\")
                    print("[[-init=(s1,s2,...) OR -init=[s1,s2,...]] -init_each={T,F}] \\")
                    print("[-Jin=Jij_in.dat -Jout=Jij_out.dat -discrete={T,F}] \\")
                    print("[-NKin=NK_in.dat -NKout=NK_out.dat]")
                elif amode == "SmartRunner":
                    print("python3 {} -amode=SmartRunner [-ltot=ltot_val OR -ltot=(ltot_min,ltot_max,ltot_num[,\"log10\"]) OR -ltot=[ltot_min,ltot_max,ltot_num[,\"log10\"]]] \\".format(sys.argv[0]))
                    print("[-DeltaF_rate_mean=DFr_val OR -DeltaF_rate_mean=(DFr_min,DFr_max,DFr_num[,\"log10\"]) OR -DeltaF_rate_mean=[DFr_min,DFr_max,DFr_num[,\"log10\"]]] \\")
                    print("[-optimism=opt_val OR -optimism=(opt_min,opt_max,opt_num[,\"log10\"]) OR -optimism=[opt_min,opt_max,opt_num[,\"log10\"]]] \\")
                    print("-mode=landscape_type -moveset_mode=moveset_type -out=outfilename \\")
                    print("[-max_feval=max_fevals -nruns=nr_val -l_max=l_max_val -straj={0,1} -socc={0,1} -sopt={1,2}] \\")
                    print("[[-init=(s1,s2,...) OR -init=[s1,s2,...]] -init_each={T,F}] \\")
                    print("[-Jin=Jij_in.dat -Jout=Jij_out.dat -discrete={T,F}] \\")
                    print("[-NKin=NK_in.dat -NKout=NK_out.dat]")
                else:
                    print("Unknown amode = {} in {}!".format(amode,sys.argv[0]))
                ####
                sys.exit("\nTerminating ..")

    return args


# This function reads data from a file output by run_optimizers.py
# into multiple Numpy matrices.
def read_data_generic(filename):

    f1 = open(filename, "r")

    header_cnt = 0
    line_cnt = 0

    data_lines = []

    for x in f1:
        toks = x.split()
        if (len(toks) == 0):
            continue # skip empty lines
        
        if toks[0] == '#' or toks[0][0] == '#':

            if header_cnt == 0:
                assert toks[1] == "amode", "Error: unexpected header line in read_data_generic!"
                amode = toks[3]
                amode = amode[:-1] # get rid of the trailing ','
                assert toks[4] == "mode", "Error: unexpected header line in read_data_generic!"
                mode = toks[6]
                mode = mode[:-1] # get rid of the trailing ','
                assert toks[7] == "moveset_mode", "Error: unexpected header line in read_data_generic!"
                moveset_mode = toks[9]
            elif header_cnt == 1:
                dim = []
                for x in toks[1].split('x'):
                    dim.append(int(x))
                assert len(dim) == 4, "Error: dimension mismatch in read_data_generic!"
                ####
                dtot = 1
                for x in dim: # ll*la*lF*nruns
                    dtot *= x
                ####
                D = int(toks[-1]) # tuple dim
                ####
                if amode == "EvolutionaryAlgorithm":
                    prm1_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm2_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm3_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                elif amode == "SimulatedAnnealing":
                    prm1_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm2_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm3_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                elif amode == "TabuSearch":
                    prm1_arr = np.zeros(dtot,dtype=np.int64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm2_arr = np.zeros(dtot,dtype=np.int64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm3_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                elif amode == "StochasticHillClimbing":
                    prm1_arr = np.zeros(dtot,dtype=np.int64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm2_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm3_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                elif amode == "SmartRunner":
                    prm1_arr = np.zeros(dtot,dtype=np.int64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm2_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                    prm3_arr = np.zeros(dtot,dtype=np.float64).reshape((dim[0],dim[1],dim[2],dim[3]))
                else:
                    raise Exception("Unknown amode: {} in read_data_generic!".format(amode))
                ####
                Fbest_arr = np.zeros(dtot).reshape((dim[0],dim[1],dim[2],dim[3]))
                feval_arr = np.zeros(dtot,dtype=np.int64).reshape((dim[0],dim[1],dim[2],dim[3]))
                if mode[:2] == "SK" or mode[:2] == "NK": # spin glasses
                    state_arr = np.zeros(dtot*D,dtype=np.int8).reshape((dim[0],dim[1],dim[2],dim[3],D))
                else:
                    state_arr = np.zeros(dtot*D).reshape((dim[0],dim[1],dim[2],dim[3],D))

            header_cnt += 1

            continue # skip commented lines

        else:
            assert len(toks) == 6, "Error: num_columns mismatch in read_data_generic!"

            if re.search('[a-zA-Z]', toks[-2]): # a string with non-numeric entries, supposed to be 'feval'
                #debug
                #print(toks[-2])
                continue # this is a header
            else:
                data_lines.append(x)
                line_cnt += 1
    #debug
    #print("line_cnt =",line_cnt)
    assert line_cnt == dtot, "Error: data dimension mismatch in read_data_generic!"

    f1.close()

    ######
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                for l in range(dim[3]):
                    toks = data_lines[cnt].split()
                    if amode == "EvolutionaryAlgorithm":
                        prm1_arr[i,j,k,l] = float(toks[0])
                        prm2_arr[i,j,k,l] = float(toks[1])
                        prm3_arr[i,j,k,l] = float(toks[2])
                    elif amode == "SimulatedAnnealing":
                        prm1_arr[i,j,k,l] = float(toks[0])
                        prm2_arr[i,j,k,l] = float(toks[1])
                        prm3_arr[i,j,k,l] = float(toks[2])
                    elif amode == "TabuSearch":
                        prm1_arr[i,j,k,l] = int(toks[0])
                        prm2_arr[i,j,k,l] = int(toks[1])
                        prm3_arr[i,j,k,l] = float(toks[2])
                    elif amode == "StochasticHillClimbing":
                        prm1_arr[i,j,k,l] = int(toks[0])
                        prm2_arr[i,j,k,l] = float(toks[1])
                        prm3_arr[i,j,k,l] = float(toks[2])
                    elif amode == "SmartRunner":
                        prm1_arr[i,j,k,l] = int(toks[0])
                        prm2_arr[i,j,k,l] = float(toks[1])
                        prm3_arr[i,j,k,l] = float(toks[2])
                    else:
                        raise Exception("Unknown amode: {} in read_data_generic!".format(amode))
                    ####
                    Fbest_arr[i,j,k,l] = float(toks[3])
                    feval_arr[i,j,k,l] = int(toks[4])
                    cur_tuple = eval(toks[5])
                    #debug
                    #print("F =",float(toks[3])," feval =",int(toks[4]), " tuple =",cur_tuple)
                    if cur_tuple is None: # this occasionally happens with amarel output
                        print("WARNING: empty tuple detected in {}: F = {}, feval = {}!".format(filename,toks[3],toks[4]))
                        cur_tuple = tuple(np.zeros(D))

                    assert len(cur_tuple) == D, "Error: tuple dim mismatch in read_data_generic!"
                    for n in range(D):
                        state_arr[i,j,k,l,n] = cur_tuple[n]
                    ####
                    cnt += 1

    dim.append(D)

    return [dim,prm1_arr,prm2_arr,prm3_arr,Fbest_arr,feval_arr,state_arr,amode,mode,moveset_mode]


# Converts integer seconds into HH:MM:SS format:
def convert(seconds):

    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%dH:%02dM:%02dS" % (hour, minutes, seconds)


# Reads fixed-format data from a fitness file:
def read_fitness_data(datafile):
    
    ltot = []
    F = []
    S = []

    f1 = open(datafile, "r")
    
    flag_header = 1
    for x in f1:
        toks = x.split()
        toks[-1] = toks[-1].strip('\n')
        #debug
        #print(toks," ",len(toks))
        if (len(toks) == 0):
            continue # skip empty lines
        elif (toks[0] == '' or toks[0] == '#' or toks[0][0] == '#'):
            continue # skip commented lines
        else:
            if flag_header == 1:
                # Header check:
                assert len(toks) == 3, "Error: dimension mismatch in read_fitness_data!"
                assert toks[0] == "ltot" and toks[1] == "F" and toks[2] == "S", "Error: unexpected header in read_fitness_data!"
                flag_header = 0
            else:
                assert len(toks) == 3, "Error: dimension mismatch in read_fitness_data!"
                ltot.append(int(toks[0]))
                F.append(float(toks[1]))
                S.append(eval(toks[2])) # this should produce a tuple

    f1.close()
    
    return [ltot,F,S]


# Reads fixed-format data from an occ file:
def read_occ_data(datafile):
    
    occ = []
    S = []

    f1 = open(datafile, "r")
    
    flag_header = 1
    for x in f1:
        toks = x.split()
        toks[-1] = toks[-1].strip('\n')
        #debug
        #print(toks," ",len(toks))
        if (len(toks) == 0):
            continue # skip empty lines
        elif (toks[0] == '' or toks[0] == '#' or toks[0][0] == '#'):
            continue # skip commented lines
        else:
            if flag_header == 1:
                # Header check:
                assert len(toks) == 2, "Error: dimension mismatch in read_occ_data!"
                assert toks[0] == "occ" and toks[1] == "S", "Error: unexpected header in read_occ_data!"
                flag_header = 0
            else:
                assert len(toks) == 2, "Error: dimension mismatch in read_occ_data!"
                occ.append(int(toks[0]))
                S.append(eval(toks[1])) # this should produce a tuple

    f1.close()
    
    return [occ,S]


# Processes multiple output files, concatenates data arrays.
def read_multiple_files(filenames):

    print("Reading data from {} ..".format(filenames[0]))

    [dim,prm1_arr,prm2_arr,prm3_arr,Fbest_arr,feval_arr,state_arr,amode,mode,moveset_mode] = \
            read_data_generic(filenames[0])

    # Keep the start and the end index of each original dataset in the final concatenated arrays:
    sind = []
    find = []
    sind.append(0)
    find.append(dim[3]-1)

    # Read in additional runs if provided:
    for j in range(1,len(filenames)):
        [dim_cur,prm1_arr_cur,prm2_arr_cur,prm3_arr_cur,Fbest_arr_cur,feval_arr_cur, \
         state_arr_cur,amode_cur,mode_cur,moveset_mode_cur] = read_data_generic(filenames[j])
        
        assert amode_cur == amode and mode_cur == mode and moveset_mode_cur == moveset_mode, \
                "Error: mode mismatch, unable to merge datafiles!"
        
        assert dim_cur[0] == dim[0] and dim_cur[1] == dim[1] and dim_cur[2] == dim[2], \
                "Error: dim mismatch, unable to merge datafiles!"
        
        # Concatenate data matrices:
        prm1_arr = np.concatenate((prm1_arr, prm1_arr_cur), axis=3)
        prm2_arr = np.concatenate((prm2_arr, prm2_arr_cur), axis=3)
        prm3_arr = np.concatenate((prm3_arr, prm3_arr_cur), axis=3)
        ######
        Fbest_arr = np.concatenate((Fbest_arr, Fbest_arr_cur), axis=3)
        feval_arr = np.concatenate((feval_arr, feval_arr_cur), axis=3)
        state_arr = np.concatenate((state_arr, state_arr_cur), axis=3)
        
        # Keep the start and the end index of each original dataset in the final concatenated arrays:
        sind.append(find[j-1]+1)
        find.append(sind[j]+dim_cur[3]-1)
        
        # Augment the dim array:
        dim[3] += dim_cur[3]
        
        print("Concatenated data from {} ..".format(filenames[j]))

    ######
    return [dim,sind,find,prm1_arr,prm2_arr,prm3_arr,Fbest_arr,feval_arr,state_arr,amode,mode,moveset_mode]


# "Smart" rounding which handles very small and very large numbers correctly.
def precision_round(number, digits=3):
    digits -= 1 # this is to make the rounding more intuitive: e.g. 0.123 -> 0.123 with digits=3, 0.123 -> 0.12 with digits=2
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - digits))


# This aux. function post-processes 'array' arguments:
def postprocess_val(tok_string,tok_type,args): # tok_type = int/float/str

    if tok_string in args:
        tok = args[tok_string]
        #debug
        #print(tok_type)
        #print("tok =",tok," type =",type(tok))
        if type(tok) is tok_type: # single value, convert to array
            tok_arr = []
            tok_arr.append(tok)
            tok = tok_arr
            #tok = eval('['+str(tok)+']')
        elif type(tok) is list:
            for x in tok:
                if not type(x) is tok_type:
                    raise Exception("Invalid type of {} value: {}".format(tok_string,type(x)))
        else:
            raise Exception("Invalid value of {}: {}".format(tok_string,tok))

    else:
        raise Exception("Please pass {} as an {} argument!".format(tok_string,sys.argv[0]))

    return tok


# This function outputs run data to a file in a flexible format.
def write_data(filename,comment_str,header_arr,data_arr):
    assert len(header_arr) == len(data_arr), "Error: input arrays must have equal lengths!"

    header_str = ''
    for j in range(len(header_arr)-1):
        header_str += "{:<4}         ".format(header_arr[j])
    header_str += "{:<4}\n".format(header_arr[-1])

    f2 = open(filename,"w")
    f2.write(comment_str)
    f2.write(header_str)
    
    for j in range(len(data_arr[0])):
        cur_str = ''
        for k in range(len(data_arr)-1):
            if type(data_arr[k][j]) is int:
                cur_str += "{:<6}    ".format(data_arr[k][j])
            elif type(data_arr[k][j]) is float:
                cur_str += "{:<.4e}    ".format(data_arr[k][j])
            else:
                raise Exception("Unknown data type: {} in write_data!".format(type(data_arr[k][j])))
        ####
        cur_str += "{}\n".format(data_arr[-1][j]) # this slot is for the best_state tuple
        f2.write(cur_str)
    
    f2.close()

