import re
import subprocess
import os

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

parse_output('test_output2',all_runs_timings,all_runs_parameters)
