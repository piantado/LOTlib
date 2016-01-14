# -*- coding: utf-8 -*-

"""
This out after we search, taking the top hypotheses that were found and evaluating them on a bunch of new data

This outputs one set of lines for each word, for each amount of data. So when CHAINS=10,

MPI run:
$ mpiexec --hostfile ../../hosts.mpich2 -n 25 python Evaluate_MPI.py --dl=0 --chains=10

TODO: Make sure this is what you want-- when your random data leads you to a hypothesis, this computes how
      often that hypothesis agrees on NEW (or average) data -- how right the hypothesis is.


TODO: ADD OUTPUT OF THE TOP HYPOTHESES AT EAHC POINT IN TIME.

"""
from collections import defaultdict
from optparse import OptionParser
import numpy as np
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Miscellaneous import logsumexp
from LOTlib.Examples.Quantifier.Model import *


DATA_RANGE = range(0, 2050, 100) # Don't need an option for this right now
parser = OptionParser()

parser.add_option("--hypotheses", dest="LOAD_HYPOTHESES_PATH", type="string",
                  help="Input file (a pickle of FiniteBestSet)",
                  default="/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Quantifier/data/mcmc-run.pkl")
parser.add_option("--out", dest="OUT_PATH", type="string",
                  help="Output file (we append -stats and -hypotheses)",
                  default="/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Quantifier/data/eval")
parser.add_option("--chains", dest="CHAINS", type="int", default=1,
                  help="Number of chains to run (new data set for each chain)")
(options, args) = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the main sampling function to run

#one run with these parameters
def run(*args):
    #print "# Running data"

    global hypotheses

    data_size = args[0]

    p_representation = defaultdict(int) # how often do you get the right representation
    p_response = defaultdict(int) # how often do you get the right response?
    p_representation_literal = defaultdict(int) # how often do you get the right representation
    p_response_literal = defaultdict(int)  # how often do you get the right response?
    p_representation_presup = defaultdict(int) # how often do you get the right representation
    p_response_presup = defaultdict(int) # how often do you get the right response?

    #print "# Generating data"
    data = generate_data(data_size)

    # recompute these
    #print "# Computing posterior"
    #[ x.unclear_functions() for x in hypotheses ]
    [ x.compute_posterior(data) for x in hypotheses ]

    # normalize the posterior in fs
    #print "# Computing normalizer"
    Z = logsumexp([x.posterior_score for x in hypotheses])

    # and output the top hypotheses
    qq = FiniteBestSet(max=True, N=25)
    for h in hypotheses: qq.push(h, h.posterior_score) # get the tops
    for i, h in enumerate(qq.get_all(sorted=True)):
        for w in h.all_words():
            fprintn(8, data_size, i, w, h.posterior_score, q(h.value[w]), f=options.OUT_PATH+"-hypotheses."+str(get_rank())+".txt")

    # and compute the probability of being correct
    #print "# Computing correct probability"
    for h in hypotheses:
        hstr = str(h)
        #print data_size, len(data), exp(h.posterior_score), correct[ str(h)+":"+w ]
        for w in words:
            p = exp(h.posterior_score - Z)
            key = w + ":" + hstr

            p_representation[w] += p * (agree_pct[key] == 1.)
            p_representation_presup[w]  += p * (agree_pct_presup[key] == 1.) # if we always agree with the target, then we count as the right rep.
            p_representation_literal[w] += p * (agree_pct_literal[key] == 1.)

            # and just how often does the hypothesis agree?
            p_response[w] += p * agree_pct[key]
            p_response_presup[w]  += p * agree_pct_presup[key]
            p_response_literal[w] += p * agree_pct_literal[key]

    #print "# Outputting"


    for w in words:
        fprintn(10, str(get_rank()), q(w), data_size, p_representation[w], p_representation_presup[w], p_representation_literal[w], p_response[w], p_response_presup[w], p_response_literal[w], f=options.OUT_PATH+"-stats."+str(get_rank())+".txt")

    return 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MPI interface

if True: #not is_master_process(): # only load if you aren't zero (else we must wait for zero to load!!

    from LOTlib.Serialization import *
    fs = file2object(options.LOAD_HYPOTHESES_PATH)
    ## The finite set of samples
    #inh = open(options.LOAD_HYPOTHESES_PATH)
    #fs = pickle.load(inh)

    hypotheses = fs.get_all()

    print "#", get_rank(), ": Loaded pickle. ", len(hypotheses), " hypotheses."

    # get all the words
    words = hypotheses[0].all_words() # just get the words from the first hypothesis

    # now figure out how often each meaning is right for each word
    agree_pct = dict()  # how often does each part of meaning agree with each word?
    agree_pct_presup = dict()
    agree_pct_literal = dict()
    for h in hypotheses:
        for w in words:
            tresp = [ target.value[w](t) for t in TESTING_SET]
            hresp = [ h.value[w](t)      for t in TESTING_SET]

            key = w+":"+str(h)
            agree_pct[key]         = np.mean( collapse_undefs(tresp) == collapse_undefs(hresp) )
            agree_pct_presup[key]  = np.mean( extract_presup(tresp)  == extract_presup(hresp) )
            agree_pct_literal[key] = np.mean( extract_literal(tresp) == extract_literal(hresp) )

    print "#", get_rank(), ": Done caching"


if __name__ == "__main__":

    # run with null args, this many times
    allret = MPI_map(run, [ [x] for x in DATA_RANGE ] * options.CHAINS ) # pass an array of lists of arguments

    print "Complete."
