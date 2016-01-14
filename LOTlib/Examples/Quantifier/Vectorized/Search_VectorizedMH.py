# -*- coding: utf-8 -*-
"""
This basically out a new version and makes a much smaller hypothesis space, one for each word, which we can then run again on

Let's add options for many different kinds of proposals:
    - Distance metric based
    - Flat uniform
    - Sample from the prior
    - Build a large, sparse, connected graph of each guys' nearest neighbors
    - What if you hash the semantics, and "propose" by altering it a little and re-querying the hash?

"""
import sys
import os
from random import randint
from copy import copy
import numpy as np
from LOTlib.MetropolisHastings import MHStats

from ..Model import *

########################################################################
# Parse the input and command lines

from optparse import OptionParser
parser = OptionParser()

parser.add_option("--in", dest="IN_PATH", type="string", help="Input file (a pickle of FiniteBestSet)",
                  default="/home/piantado/Desktop/mit/Libraries/LOTlib/examples/Quantifier/data/all_trees_2012May2.pkl")
parser.add_option("--out", dest="OUT_PATH", type="string", help="Output file (a pickle of FiniteBestSet)",
                  default="/home/piantado/Desktop/mit/Libraries/LOTlib/examples/Quantifier/data/gibbs_trees.pkl")

#parser.add_option("--recompute-cache", dest="recompute-cache", action="store_true", default=False, help="Recompute the surprisal cache")

parser.add_option("--steps", dest="STEPS", type="int", default=10000, help="Number of Gibbs cycles to run")
parser.add_option("--skip", dest="SKIP", type="int", default=500, help="Samples to skip")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=50, help="Top number of hypotheses to store")
parser.add_option("--chains", dest="CHAINS", type="int", default=1, help="Number of chains to run (new data set for each chain)")

parser.add_option("--gibbs",   action="store_true",  dest="gibbs", default=False, help="Run Gibbs instead of MH")

parser.add_option("--recompute-cache",   action="store_true",  dest="recomputecache", default=False, help="Recompute the cached distance proposals (for MH)")
parser.add_option("--no-save-cache",   action="store_true",  dest="nosavecache", default=False, help="Don't saved the cached distance proposal")

parser.add_option("--data", dest="DATA", type="int",default=-1, help="Amount of data")
parser.add_option("--dmin", dest="DATA_MIN", type="int",default=25, help="Min data to run")
parser.add_option("--dmax", dest="DATA_MAX", type="int", default=500, help="Max data to run")
parser.add_option("--dstep", dest="DATA_STEP", type="int", default=25, help="Step size for varying data")

parser.add_option("--alpha", dest="ALPHA", type="float", default=0.95, help="Learner's assumed probability of true utterances")
parser.add_option("--palpha", dest="PALPHA", type="float", default=0.95, help="Parent's probability of true utterances")
parser.add_option("--onlyconservative",   action="store_true",  dest="ONLY_CONSERVATIVE", default=False, help="Only use conservative hypotheses")
parser.add_option("--onlycorrect",   action="store_true",  dest="ONLY_CORRECT", default=False, help="Only use the correct hypotheses")

parser.add_option("--dl", dest="DEBUG_LEVEL", type="int", default=10, help="Debug level -- higher will print more, 0 prints minimum")
parser.add_option("-q", "--quiet", action="store_true", dest="QUIET", default=False, help="Don't print status messages to stdout")
parser.add_option("-m", "--mpi",   action="store_true",  dest="RUN_MPI", default=False, help="For running on MPI (under mpiexec or similar)")

(options, args) = parser.parse_args()

if options.DATA == -1: options.DATA_AMOUNTS = range(options.DATA_MIN,options.DATA_MAX,options.DATA_STEP)
else:                  options.DATA_AMOUNTS = [ options.DATA ]

if not options.RUN_MPI: display_option_summary(options)
if options.RUN_MPI: from SimpleMPI.MPI_map import MPI_map, is_master_process # get our MPI_map function, which will execute run() on as many processors as we pass to mpiexec # import before we set DEBUG_LEVEL

# manage how much we print
if options.QUIET: options.DEBUG_LEVEL = 0
LOTlib.Miscellaneous.DEBUG_LEVEL = options.DEBUG_LEVEL

########################################################################
# Define Run and a function mapping vectorized lexicon to normal

# Main run function
def run(data_size):
    """
            A function to run on a given data size. We call this below either locally or for MPI
    """

    data = generate_data(data_size)

    # the prior for each tree
    prior = np.array([ x.compute_prior() for x in my_finite_trees])
    prior = prior - logsumexp(prior)

    ## the likelihood weights for each hypothesis
    weights = np.array([ my_weight_function(h) for h in my_finite_trees ])

    # response[h,di] gives the response of the h'th tree to data di
    response = np.array( [ mapto012(t.get_function_responses(data)) for t in my_finite_trees] )

    # reformat the data to make it vectory
    uttered_word_index = [word2index[d.word] for d in data]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Now actually run:
    hypset = FiniteBestSet(N=options.TOP_COUNT, max=True)

    learner = VectorizedLexicon_DistanceMetricProposal(target.all_words(), my_finite_trees, prior )
    databundle = [response, weights, uttered_word_index]

    mhstats = MHStats()

    if options.gibbs:
        generator = LOTlib.MetropolisHastings.gibbs_sample(learner, databundle, options.STEPS, dimensions=xrange(len(target.all_words())))
    else:
        generator = LOTlib.MetropolisHastings.mh_sample(learner, databundle, options.STEPS, skip=options.SKIP, stats=mhstats)
        #LOTlib.MetropolisHastings.tempered_transitions_sample(learner, databundle, options.STEPS, skip=options.SKIP, temperatures=[1.0, 1.5, 2.0])


    for g in generator:
        hypset.push(VectorizedLexicon_to_SimpleLexicon(g), g.posterior_score)
        mhstats.clear()

    return hypset


########################################################################
# Load the trees from either a file or a set of lexica
# and filter by conservative, etc.

my_finite_trees = load_finite_trees(options.IN_PATH)
print "# Done loading", len(my_finite_trees), "trees"

if options.ONLY_CONSERVATIVE: my_finite_trees = filter(is_conservative, my_finite_trees)
hyp2target = create_hyp2target(my_finite_trees) # A hash from each of these trees to what target word they match, if any
if options.ONLY_CORRECT: my_finite_trees = filter(lambda x: hyp2target[x] is not None,  my_finite_trees )
print "# Filtered to", len(my_finite_trees), "trees"

####################################################################################
## A bunch of things for computing distances and gradients, etc

# Hash each hypothesis to its index
hyp2index = {}
for i,h in enumerate(my_finite_trees):
    hyp2index[h] = i

if not options.gibbs: # if we aren't doing gibbs,
    DISTANCE_CACHE = "/home/piantado/Desktop/mit/Libraries/LOTlib/examples/Quantifier/cache/cache-"+os.path.basename(options.IN_PATH)
    if options.recomputecache or not os.path.exists(DISTANCE_CACHE):
        SCALE = 0.75 # tuned to give a reasonable acceptance ratio
        #TOP_N = len(my_finite_trees)
        TOP_N = 1000 # store only the top this many

        ## Compute distances the matrix way:
        data = generate_data(100)
        RM = numpy.zeros( (len(my_finite_trees), len(data)) ) # repsonse matrix
        for i, x in enumerate(my_finite_trees):
            RM[i,:] = numpy.array(mapto012(x.get_function_responses(data)))
        RM = numpy.matrix(RM)

        # Okay now convert to each of the kinds of agreement we want
        # NOTE: (RM==1)+0 Gives an integer matrix, 0/1 for whether RM==1
        # NOTE: we need to do these as += because otherwise we can get memory errors
        agree  = ((RM==1)+0) * ((RM==1)+0).transpose() # How often do we agree on 1s (yay matrix math)
        agree += ((RM==-1)+0) * ((RM==-1)+0).transpose() # how often do we agree on -1s
        agree += ((RM==0)+0) * ((RM==0)+0).transpose() # how often do we agree on undefs?
        print "# Done computing agreement matrix for proprosals"

        proposal_probability = numpy.exp( - SCALE * (len(data)-agree) ) # the more you disagree, the less you are proposed to

        # now we have to sort:
        mftlen = len(my_finite_trees)
        proposal_to =  numpy.zeros( (mftlen,TOP_N) )
        proposal_probs =  numpy.zeros( (mftlen,TOP_N) )
        proposal_Z =  numpy.zeros( (mftlen,1) )
        myrange = range(len(my_finite_trees))
        for i in xrange(len(my_finite_trees)):

            r = numpy.array( proposal_probability[ i, : ].tolist()[0] )

            r[i] = 0.0 # never propose to ourself

            # now sort
            r = -r # so we sort correctly, max first
            idx = r.argsort(kind='mergesort')[0:TOP_N] #Sort and take the first TOP_N
            r = -r
            proposal_Z[i] = numpy.sum(r[idx]) # necessary since we take a subset
            proposal_to[i, :]   = idx
            proposal_probs[i, :] = r[idx]

        # and save
        if not options.nosavecache:
            pickle.dump( [proposal_to, proposal_probs, proposal_Z], open( DISTANCE_CACHE, "wb" ) )
    else:
        proposal_to, proposal_probs, proposal_Z = pickle.load( open( DISTANCE_CACHE, "rb" ) )


def distance_based_proposer(x):
    y,lp = weighted_sample(proposal_to[x,:], probs=proposal_probs[x,:], Z=proposal_Z[x], return_probability=True, log=False)
    bp = lp + log(proposal_Z[x]) - log(proposal_Z[y]) # the distance d is the same, but the normalizer differs
    return y, lp - bp

####################################################################################
# Define a class that can do M-H on these fancy proposals

class VectorizedLexicon_DistanceMetricProposal(VectorizedLexicon):

    def __init__(self, *args, **kwargs): VectorizedLexicon.__init__(self, *args, **kwargs)
    def copy(self): return VectorizedLexicon_DistanceMetricProposal(self.target_words, self.finite_trees, self.priorlist, word_idx=np.copy(self.word_idx), ALPHA=self.ALPHA, PALPHA=self.PALPHA)

    def propose(self):
        new = self.copy()
        i = sample1( xrange(len(self.word_idx)) )
        #p, fb = distance_based_proposer( my_finite_trees[i] ) # TODO: MAKE OUR SEMANTIC DISTANCE METRIC ONE CHANGE
        #new.word_idx[i] = hyp2index[p]
        p, fb = distance_based_proposer( i )
        new.word_idx[i] = p
        return new, fb

def VectorizedLexicon_to_SimpleLexicon(vl):
    L = SimpleLexicon(grammar, args=['A', 'B', 'S']) ## REALLY THIS SHOULD BE GRICEAN
    for i, wi in enumerate(vl.word_idx):
        L.set_word(index2word[i], vl.finite_trees[wi])

    return L


####################################################################################
####################################################################################
## Main running
####################################################################################
####################################################################################

# choose the appropriate map function
if options.RUN_MPI:
    allret = MPI_map(run, map(lambda x: [x], options.DATA_AMOUNTS * options.CHAINS) ) # this many chains
    # NOTE: the child processes do not survive MPI_map
else:
    allret = map(run,  options.DATA_AMOUNTS * options.CHAINS)

## combine into a single hypothesis set and save --- only for rank  0 since all others are eaten by MPI_map
outhyp = FiniteBestSet(max=True)
for r in allret:
    outhyp.merge(r)
outhyp.options = options ## Save these so that the output data has our options as a field
outhyp.save(options.OUT_PATH)

# for debugging:
#learner.compute_likelihood()
#for g in LOTlib.MetropolisHastings.gibbs_sample(learner, data, options.STEPS, dimensions=xrange(len(target.all_words()))):
    #print g.posterior_score, g.prior, g.likelihood, g.word_idx, "\n", g
