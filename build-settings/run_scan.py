import re
import subprocess
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_tag", type=str,  help="run tag")
parser.add_argument('-p', '--crowd_size', nargs="+", type=int, help="crowd sizes [1..N]", default= [1,2,4,8]) 
parser.add_argument('-c', '--crowds', nargs="+", type=int, help="numbers of crowds [1..N]", default = [1,4,8,16]) 
parser.add_argument('-b', '--block_size', nargs="+", type=int, help="block sizes [1..N]", default = [256])
parser.add_argument('-t', '--trials', type=int, help="number of each combination to run", default = 1)
parser.add_argument('-g', '--cells', nargs="+", type=str, help="cells to run string [1..N]", default = '2 1 1' )
parser.add_argument('-d', '--devices', nargs="+", type=int, help="device [1..N]", default = [1] )
args = parser.parse_args()

print (args)
# defaults

for p in itertools.product(args.devices,
                           args.cells,
                           args.crowds,
                           args.crowd_size,
                           args.block_size):
    device, cell, pack_number, pack_size, bsize = p
    for trial in range(args.trials):
        cell_str = re.sub('\s','_',cell)
        outfile = "scan_miniqmc_{}_d{}_g{}_p{}_w{}_a{}.out_{}".format(args.run_tag,device,cell_str,pack_size, pack_number, bsize, trial)
        command_str = str(["bin/miniqmc","-d",device,"-M","-p", "{}".format(pack_size),
                                               "-w", "{}".format(pack_number), "-a", "{}".format(bsize),
                                               "-g", "{}".format(cell)])
        print("trying",command_str)
        print(outfile)
        with open(outfile, 'w') as f:
            process_complete =  subprocess.run(["bin/miniqmc","-d",str(device),"-M","-p", "{}".format(pack_size),
                                               "-w", "{}".format(pack_number), "-a", "{}".format(bsize),
                                               "-g", "{}".format(cell)], stdout=f)
            if process_complete.returncode != 0:
                print ("fail", outfile)

