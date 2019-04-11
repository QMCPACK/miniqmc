import re
import subprocess
import os
import pickle

collect_timings = [r'Total',
                   r'Pseudopotential',
                   r'Spline Hessian Evaluation',
                   r'Update']
                   
collect_parameters = ['Number of orbitals',
                      'Splines per block',
                      'Number of tiles',
                      'Number of electrons',
                      'Rmax',
                      'Iterations',
                      'OpenMP threads',
                      'pack size =',
                      'crowds =']
                      
parameter_capture = r'.*\s([\d\.]+)'

all_runs_timings = []
all_runs_parameters = []
def parse_output(filename, all_run_timings, all_runs_parameters):
    with open(filename, 'r') as f:
        lines = f.readlines()
        this_runs_timings = {}
        this_runs_parameters = {}
        for line in lines:
            for param in collect_parameters:
                regex = param + parameter_capture;
                match = re.search(regex, line);
                if match:
                    this_runs_parameters[param] = match.group(1)
            for timing in collect_timings:
                regex = timing + r'\s+([0-9\.]+)'
                match = re.search(regex, line);
                if match:
                    this_runs_timings[timing] = match.group(1)
        print (this_runs_parameters)
        print (this_runs_timings)
        all_runs_timings.append(this_runs_timings) 
        all_runs_parameters.append(this_runs_parameters)


packs = [1,2,4,8,12,16,24,48]
batching = [1,2,4,8]
block_size = [256]
trials = 5

for pack_number in packs:
    for pack_size in batching:
        for bsize in block_size:
            for trial in range(trials):
                outfile = "scan_miniqmc_gpu_splines_p{}_w{}_a{}.out_{}".format(pack_size, pack_number, bsize, trial)
                print("trying",outfile)
                with open(outfile, 'w') as f:
                    process_complete = subprocess.run(["bin/miniqmc","-d","1","-M","-p", "{}".format(pack_size),
                                                       "-w", "{}".format(pack_number), "-a", "{}".format(bsize),
                                                       "-g", "2 1 1"], stdout=f)
                    if process_complete.returncode != 0:
                        print ("fail", outfile)

                        
# parse_output(outfile,all_runs_timings,all_runs_parameters)

# with open('pickled_miniqmc_scan', 'w') as f:
#     my_pickle = pickle.Pickler(f)
#     my_pickle.dump(all_runs_parameters)
#     my_pickle.dump(all_runs_timings)

# pkeys = [key for key in keys(all_runs_parameters[0])]
# pkey_string = pkeys.join(" ")
# tkeys = [key for key in keys(all_runs_timings[0])]
# tkey_string = tkeys.join(" ")
# header_string = pkey_string + " " + tkey_string
# print(header_string)
# for params, timings in zip(all_runs_param, all_runs_timings):
#     pvals = [val for key, val in params.items()]
#     tvals = [val for key, val in timings.items()]
#     pval_string = pvals.join(" ")
#     tval_string = tvals.join(" ")
#     val_string = pval_string + " " + tval_string
#     print(val_string)


