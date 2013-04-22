# -*- coding: utf-8 -*-
"""
	This basically runs a new version and makes a much smaller hypothesis space, one for each word, which we can then run again on
"""
import sys
from random import randint
from copy import copy
import numpy as np

from Shared import *

####################################################################################
# Parse command line options

## Pares command line options
from optparse import OptionParser
parser = OptionParser()

parser.add_option("--in", dest="IN_PATH", type="string", help="Input file (a pickle of UniquePriorityQueue)", 
                  default="/home/piantado/Desktop/mit/Libraries/LOTlib/examples/Quantifier/data/all_trees_2012Mar11.pkl")
parser.add_option("--out", dest="OUT_PATH", type="string", help="Output file (a pickle of UniquePriorityQueue)", 
                  default="/home/piantado/Desktop/mit/Libraries/LOTlib/examples/Quantifier/data/gibbs_trees.pkl")
         
parser.add_option("--steps", dest="STEPS", type="int", default=100, help="Number of Gibbs cycles to run")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=50, help="Top number of hypotheses to store")
parser.add_option("--chains", dest="CHAINS", type="int", default=1, help="Number of chains to run (new data set for each chain)")
                  
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
if options.RUN_MPI: from LOTlib.MPI import MPI_map # get our MPI_map function, which will execute run() on as many processors as we pass to mpiexec # import before we set DEBUG_LEVEL

# manage how much we print
if options.QUIET: options.DEBUG_LEVEL = 0
LOTlib.Miscellaneous.DEBUG_LEVEL = options.DEBUG_LEVEL

# And echo the command line options for a record
dprintn(5, "# Running: ", options)

####################################################################################
# Map vectorized lexica to simple ones

def VectorizedLexicon_to_SimpleLexicon(vl):
		
	L = SimpleLexicon(G, ['A', 'B', 'S']) ## REALLY THIS SHOULD BE GRICEAN
	for i, wi in enumerate(vl.word_idx):
		L.set_word(index2word[i], vl.finite_trees[wi])
	
	return L

########################################################################
# Main run function
def run(data_size):
	"""
		A function to run on a given data size. We call this below either locally or for MPI
	"""
	data = generate_data(data_size)
	dprintn(8, "# Done generating data", data_size)
	
	# the prior for each tree
	prior = np.array([ x.log_probability() for x in my_finite_trees])
	prior = prior - logsumexp(prior)
	dprintn(8, "# Done computing priors", data_size)
	
	## the likelihood weights for each hypothesis
	weights = np.array([ my_gricean_weight(h, evaluate_expression(h, ['A', 'B', 'S']))  for h in my_finite_trees ])
	dprintn(8, "# Done computing weights", data_size)
	
	# response[h,di] gives the response of the h'th tree to data di
	response = np.array( [ mapto012(get_single_tree_responses(t, data, data=True)) for t in my_finite_trees] )
	dprintn(8, "# Done computing responses", data_size)
		
	# reformat the data to make it vectory
	uttered_word_index = [word2index[d[0].word] for d in data]
	
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Now actually run:
	hypset = UniquePriorityQueue(options.TOP_COUNT, max=True) 
	
	learner = VectorizedLexicon( target.all_words(), my_finite_trees, prior)
	databundle = [response, weights, uttered_word_index]
	
	for g in LOTlib.MetropolisHastings.gibbs_sample(learner, databundle, options.STEPS, dimensions=xrange(len(target.all_words()))):
		
		dprintn(10, data_size, g.lp, g.prior, g.likelihood, g.word_idx)
		dprintn(15, g) # and show the lexicon
		hypset.push(VectorizedLexicon_to_SimpleLexicon(g), g.lp)
	
	return hypset


########################################################################
## Main loop:
########################################################################

# Load the trees from either a file or a set of lexica
my_finite_trees = load_finite_trees(options.IN_PATH)
dprintn(5, "# Done loading", len(my_finite_trees), "trees")	

if options.ONLY_CONSERVATIVE: my_finite_trees = filter(is_conservative, my_finite_trees)
hyp2target = create_hyp2target(my_finite_trees) # A hash from each of these trees to what target word they match, if any
if options.ONLY_CORRECT: my_finite_trees = filter(lambda x: hyp2target[x] is not None,  my_finite_trees )
dprintn(5, "# Filtered to", len(my_finite_trees), "trees")

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
outhyp = UniquePriorityQueue(N=-1,max=True)
for r in allret: 
	dprintn(5, "# Merging ", len(r.Q))
	outhyp.merge(r)
outhyp.options = options ## Save these so that the output data has our options as a field
outhyp.save(options.OUT_PATH)

# for debugging:
#learner.compute_likelihood()
#for g in LOTlib.MetropolisHastings.gibbs_sample(learner, data, options.STEPS, dimensions=xrange(len(target.all_words()))):
	#print g.lp, g.prior, g.likelihood, g.word_idx, "\n", g
	