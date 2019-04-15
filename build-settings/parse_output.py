import json
import argparse
import re
import subprocess
import os
import pickle
import ast
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

parser = argparse.ArgumentParser()
parser.add_argument("run_tag", help="run tag")
args = parser.parse_args()

collect_timings = [r'Total',
                   r'Pseudopotential',
                   r'Spline Hessian Evaluation',
                   r'Update']
timing_capture = r'\s+([0-9\.]+)'

collect_parameters = ['primitive cells',
                      'Number of orbitals',
                      'splines per block',
                      'Number of tiles',
                      'Number of electrons',
                      'Rmax',
                      'Iterations',
                      'OpenMP threads',
                      'pack size =',
                      'crowds']
                      
parameter_capture = r'.*\s([\s\d\.]+)'

all_runs_timings = []
all_runs_parameters = []
def parse_output(filename, all_run_timings, all_runs_parameters):
    with open(filename, 'r') as f:
        lines = f.readlines()
        this_runs_timings = {}
        this_runs_parameters = {}
        for line in lines:
            try:
                for param in collect_parameters:
                    regex = param + parameter_capture
                    match = re.search(regex, line)
                    if match:
                        this_runs_parameters[param] = ast.literal_eval(match.group(1))
                for timing in collect_timings:
                    regex = timing + timing_capture
                    match = re.search(regex, line)
                    if match:
                        this_runs_timings[timing] = ast.literal_eval(match.group(1))
            except:
                print("parse failure on line:\n", line)

        all_runs_timings.append(this_runs_timings) 
        all_runs_parameters.append(this_runs_parameters)

packs = [1,2,4,8,12,16] #,24]#,32]
batching = [1,2,4,8] #,16,24,32]
block_size = [256]
trials = 3

for pack_number in packs:
    for pack_size in batching:
        for bsize in block_size:
            for trial in range(trials):
                outfile = "scan_miniqmc_{}_p{}_w{}_a{}.out_{}".format(args.run_tag, pack_size, pack_number, bsize, trial)
                print("reading",outfile)
                parse_output(outfile,all_runs_timings,all_runs_parameters)

# with open('pickled_miniqmc_scan', 'w') as f:
#     my_pickle = pickle.Pickler(f)
#     my_pickle.dump(all_runs_parameters)
#     my_pickle.dump(all_runs_timings)

pkeys = [key for key in all_runs_parameters[0].keys()]
pkey_string = "' '".join(pkeys)
tkeys = [key for key in all_runs_timings[0].keys()]
tkey_string = "' '".join(tkeys)
header_string = pkey_string + " " + tkey_string
print(header_string)
runs_datai = []
# for params, timings in zip(all_runs_parameters, all_runs_timings):
#     pvals = [val for key, val in params.items()]
#     tvals = [val for key, val in timings.items()]
#     pval_string = " ".join(str(pvals))
#     tval_string = " ".join(str(tvals))
#     val_string = pval_string + " " + tval_string
#     print(val_string)
#     runs_datai.append(pvals + tvals)


#now to merge identical runs and make an error bar.
merged_runs = {}
for params, timings in zip(all_runs_parameters, all_runs_timings):
    param_hash = hash(json.dumps(params))
    if param_hash in merged_runs:
        merged_runs[param_hash]['timings'].append(timings)
    else:
        merged_runs[param_hash] = {'params' : params,
                                   'timings' : [timings]}

for hash, mr in merged_runs.items():
    trials = len(mr['timings'])
    mr['avg_timings'] = {}
    mr['c95_timings'] = {}
    for key in mr['timings'][0]:
        series = [ timing[key] for timing in mr['timings'] ]
        mean = np.mean(series)
        mr['avg_timings'][key] = mean
        interval = st.t.interval(0.95, len(series)-1, loc=mean,
                                               scale=st.sem(series))
        mr['c95_timings'][key] = mean - interval[0]
    print (mr['avg_timings'])
    print (mr['c95_timings'])    


        
fig = plt.figure()
ax = fig.add_subplot(111)

all_merged_params = [ mr['params'] for hash, mr in merged_runs.items() ]
all_merged_avgs = [ mr['avg_timings'] for hash, mr in merged_runs.items() ]
all_merged_c95s = [ mr['c95_timings'] for hash, mr in merged_runs.items() ]

print(all_merged_c95s)

vs_pack_size = {}
for npacks in packs:
    for bsize in [256]:
        psize_data = [[params['pack size ='], (params['pack size ='] * params['crowds'] * params['Iterations']) / avg_timings['Total'], (params['pack size ='] * params['crowds'] * params['Iterations']), avg_timings['Total'],
                       c95_timings['Total']]
                      for params, avg_timings, c95_timings in zip(all_merged_params, all_merged_avgs,
                                                                  all_merged_c95s)
                      if 'crowds' in params and params['crowds'] == npacks and params['splines per block'] == bsize and 'Total' in timings]
        label = "{:} threads".format(npacks, bsize)
        x = [ x[0] for x in psize_data ]
        y = [ y[1] for y in psize_data ]
        yerr = [ p[2] / (p[3] + p[4]) - p[1] for p in psize_data ]
#        //yerr = [ [ err[1] - err[2][0], err[2][1] - err[1] ] for err in psize_data ]
#        yerr = list(map(list, zip(*yerr)))
        print (yerr)
        ax.errorbar(x, y, yerr=yerr, capsize=3, label=label, fmt='.-')

ax.set_title("{} Total Steps per second".format(args.run_tag))
ax.set_xlabel('batch size')
ax.set_ylabel('walker steps per second')
ax.legend()

plt.savefig("g{}_a{}_{}_batchsize.pdf".format(all_runs_parameters[0]['primitive cells'], all_runs_parameters[0]['splines per block'], args.run_tag))
#plt.show()



fig = plt.figure()
ax = fig.add_subplot(111)

vs_pack_size = {}
for npacks in packs:
    for bsize in [256]:
        psize_data = [[params['pack size ='], avg_timings['Total'] / params['Iterations'], c95_timings['Total'] / params['Iterations'] ]
                      for params, avg_timings, c95_timings in zip(all_merged_params, all_merged_avgs, all_merged_c95s)
                      if 'crowds' in params and params['crowds'] == npacks and params['splines per block'] == bsize and 'Total' in timings]
        label = "{:} threads".format(npacks)
        x = [ x[0] for x in psize_data ]
        y = [ y[1] for y in psize_data ]
        yerr = [ p[2] for p in psize_data ]
        ax.errorbar(x, y, yerr=yerr, capsize=3, label=label, fmt='.-')

ax.set_title("All {} walkers take one step (avg)".format(args.run_tag))
ax.set_xlabel('batch size')
ax.set_ylabel('step time')
ax.legend()
plt.savefig("g{}_a{}_{}_batchsize_step_speed.pdf".format(all_runs_parameters[0]['primitive cells'], all_runs_parameters[0]['splines per block'], args.run_tag))

#plt.show()

