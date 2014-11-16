# -*- coding: utf-8 -*-

"""
All out rational-rules style gibbs on lexicons.
For MPI or local.
This is much slower than the vectorized versions.

MPI run:
$ mpiexec --hostfile ../../hosts.mpich2 -n 15 python Search_MCMC.py

"""
from LOTlib.MPI.MPI_map import MPI_map
from LOTlib import mh_sample
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Examples.Quantifier.Model import *

CHAINS = 3  #how many times do we run?
DATA_AMOUNTS = range(0,300, 100) #range(0,1500,100)
SAMPLES = 1 # 1000000
TOP_COUNT = 50
OUT_PATH = "/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Quantifier/data/mcmc-run.pkl"

QUIET = False
RUN_MPI = True # should we run on MPI? If not, just run as normal python


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the main sampling function to run

#one run with these parameters
def run(data_size):

    print "Running ", data_size

    # We store the top 100 from each run
    hypset = FiniteBestSet(TOP_COUNT, max=True)

    # initialize the data
    data = generate_data(data_size)

    # starting hypothesis -- here this generates at random
    learner = GriceanQuantifierLexicon(make_my_hypothesis, my_weight_function)

    # We will defautly generate from null the grammar if no value is specified
    for w in target.all_words(): learner.set_word(w)

    # populate the finite sample by running the sampler for this many steps
    for x in mh_sample(learner, data, SAMPLES, skip=0):
        hypset.push(x, x.posterior_score)

    return hypset

if __name__ == "__main__":

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # MPI interface

    # Map. SimpleMPI will use a normal MAP if we are not running in MPI
    allret = MPI_map(run, map(lambda x: [x], DATA_AMOUNTS * CHAINS)) # this many chains

    ## combine into a single hypothesis set and save
    outhyp = FiniteBestSet(max=True)
    for r in allret:
        print "# Merging ", len(r)
        outhyp.merge(r)

    import pickle
    pickle.dump(outhyp, open(OUT_PATH, 'w'))
