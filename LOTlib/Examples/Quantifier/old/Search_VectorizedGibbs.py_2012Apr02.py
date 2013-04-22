# -*- coding: utf-8 -*-
"""
	A super optimized vectorized form of the quantifier learner. Still in development.
	This runs on local or MPI, using options.RUN_MPI flag
	
	Run via:
	
	$time mpiexec --hostfile ../../hosts.mpich2 -n 20 python2.7-mpi Search_VectorizedGibbs.py -m --steps=100 --chains=10
	
	TODO:
	-- For quick math, we just have a smoothing term in the demoninator which should not matter, but check this in the future!
	-- Check what happens when you have more cpus than tasks!
	
"""
import sys
#sys.path.append("..")
from Shared import *

from random import randint
from copy import copy
import numpy as np

## Pares command line options
from optparse import OptionParser
parser = OptionParser()

parser.add_option("--in", dest="IN_PATH", type="string",
                  help="Input file (a pickle of UniquePriorityQueue)", 
                  default="/home/piantado/Desktop/mit/Libraries/LOTlib/examples/Quantifier/data/all_trees_2012Mar11.pkl")
parser.add_option("--out", dest="OUT_PATH", type="string",
                  help="Output file (a pickle of UniquePriorityQueue)", 
                  default="/home/piantado/Desktop/mit/Libraries/LOTlib/examples/Quantifier/data/gibbs_trees.pkl")
         
parser.add_option("--steps", dest="STEPS", type="int", default=100,
                  help="Number of Gibbs cycles to run")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=50,
                  help="Top number of hypotheses to store")
parser.add_option("--chains", dest="CHAINS", type="int", default=1,
                  help="Number of chains to run (new data set for each chain)")
                  
parser.add_option("--data", dest="DATA", type="int",default=-1,
                  help="Amount of data")
parser.add_option("--dmin", dest="DATA_MIN", type="int",default=25,
                  help="Min data to run")
parser.add_option("--dmax", dest="DATA_MAX", type="int", default=500,
                  help="Max data to run")
parser.add_option("--dstep", dest="DATA_STEP", type="int", default=25,
                  help="Step size for varying data")

parser.add_option("--alpha", dest="ALPHA", type="float", default=0.95,
                  help="Learner's assumed probability of true utterances")
parser.add_option("--palpha", dest="PALPHA", type="float", default=0.95,
                  help="Parent's probability of true utterances")
parser.add_option("--onlyconservative",   action="store_true",  dest="ONLY_CONSERVATIVE", default=False, help="Only use conservative hypotheses")
parser.add_option("--onlycorrect",   action="store_true",  dest="ONLY_CORRECT", default=False, help="Only use the correct hypotheses")

# standard options
parser.add_option("--dl", dest="DEBUG_LEVEL", type="int", default=10,
                  help="Debug level -- higher will print more, 0 prints minimum")
parser.add_option("-q", "--quiet", action="store_true", dest="QUIET", default=False, help="Don't print status messages to stdout")
parser.add_option("-m", "--mpi",   action="store_true",  dest="RUN_MPI", default=False, help="For running on MPI (under mpiexec or similar)")

(options, args) = parser.parse_args()

if options.DATA == -1:
	options.DATA_AMOUNTS = range(options.DATA_MIN,options.DATA_MAX,options.DATA_STEP)
else:
	options.DATA_AMOUNTS = [ options.DATA ]

if not options.RUN_MPI:
	display_option_summary(options)

if options.RUN_MPI: # import before we set DEBUG_LEVEL
	from LOTlib.MPI import MPI_map # get our MPI_map function, which will execute run() on as many processors as we pass to mpiexec
	
# manage how much we print
if options.QUIET: options.DEBUG_LEVEL = 0
LOTlib.Miscellaneous.DEBUG_LEVEL = options.DEBUG_LEVEL

# And echo the command line options for a record
dprintn(5, "# Running: ", options)

#####################################################################
## A fancy class for doing gibbs over this finite set of trees
class GibbsyGriceanVectorizedLexicon(Hypothesis):
	"""
		This is a Lexicon class that stores *only* indices into my_finite_trees, and is designed for doing gibbs,
		sampling over each possible word meaning. It requires running EnumerateTrees.py to create a finite set of 
		trees, and then loading them here. Then, all inferences etc. are vectorized for super speed.
		
		This requires a bunch of variables (the global ones) from run, so it cannot portably be extracted. But it can't
		be defined inside run or else it won't pickle correctly. Ah, python. 
	"""
	
	def __init__(self, word_idx=None):
		Hypothesis.__init__(self)
		
		if word_idx is None:
			self.word_idx = np.array( [randint(0,len(my_finite_trees)-1) for i in xrange(len(target.all_words())) ])
		else:
			self.word_idx = word_idx
	
	def __repr__(self): return str(self)
	def __eq__(self, other): return np.all(self.word_idx == other.word_idx)
	def __hash__(self): return hash(tuple(self.word_idx))
	def __cmp__(self, other): return cmp(str(self), str(other))
	
	def __str__(self): 
		global my_finite_trees
		s = ''
		aw = target.all_words()
		for i in xrange(len(self.word_idx)):
			s = s + aw[i] + "\t" + str(my_finite_trees[self.word_idx[i]]) + "\n"
		s = s + '\n'
		return s
	
	def copy(self):
		return GibbsyGriceanVectorizedLexicon(word_idx=np.copy(self.word_idx))
	
	def enumerative_proposer(self, wd):
		for k in xrange(len(my_finite_trees)):
			new = self.copy()
			new.word_idx[wd] = k
			yield new
			
	def compute_prior(self):
		self.prior = sum([prior[x] for x in self.word_idx])
		self.lp = self.prior + self.likelihood
		return self.prior
	
	def compute_likelihood(self, *args):
		"""
			Compute the likelihood on the data, super fast
			
			These use the global variables defined by run below.
			The alternative is to define this class inside of run, but that causes pickling errors
		"""
		
		r = response[self.word_idx] # gives vectors of responses to each data point
		w = weights[self.word_idx]
		#print r.shape
		if r.shape[1] == 0:  # if no data
			self.likelihood = 0.0
		else:
			
			true_weights = ((r > 0).transpose() * w).transpose().sum(axis=0)
			met_weights  = ((np.abs(r) > 0).transpose() * w).transpose().sum(axis=0)
			all_weights = w.sum(axis=0)
			rutt = r[uttered_word_index, zerothroughcols] # return value of the word actually uttered
			
			## now compute the likelihood:
			lp = np.sum( np.log( options.PALPHA*options.ALPHA*weights[self.word_idx[uttered_word_index]]*(rutt>0) / (1e-5 + true_weights) + \
					options.PALPHA*( (true_weights>0)*(1.0-options.ALPHA) + (true_weights==0)*1.0) * weights[self.word_idx[uttered_word_index]] * (np.abs(rutt) > 0) / (1e-5 + met_weights) + \
					( (met_weights>0)*(1.0-options.PALPHA) + (met_weights==0)*1.0 ) * weights[self.word_idx[uttered_word_index]] / (1e-5 + all_weights)))
				
			self.likelihood = lp
		self.lp = self.likelihood+self.prior
		return self.likelihood 
		
	def to_lexicon(self):
		
		L = GriceanSimpleLexicon(G, ['A', 'B', 'S'])
		for i, wi in enumerate(self.word_idx):
			L.set_word(index2word[i], my_finite_trees[wi])
			
			
		return L
		
########################################################################
# Main run function
def run(data_size):
	"""
		A function to run on a given data size. We call this below either locally or for MPI
	"""
	
	## These variables must be defined globally so they are in scope for GibbsyGriceanVectorizedLexicon
	global response
	global weights
	global uttered_word_index
	global prior
	global data
	global word2index
	global index2word
	global zerothroughcols
	global my_finite_trees
	
	#if SIG_INTERRUPTED: return 
	
	#dprintn(8, "# Running ", data_size, "on rank ", MPI.COMM_WORLD.Get_rank() )
	
	data = generate_data(data_size)
	dprintn(8, "# Done generating data", data_size)
	#print data

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
	
	# map each word to an index -- NOTE: The form in the class below MUST match this order
	word2index = dict()
	index2word = target.all_words()
	for i, w in enumerate(index2word):
		word2index[w] = i
		index2word[i] = w
		
	# reformat the data to make it vectory
	uttered_word_index = [word2index[d[0].word] for d in data]
	
	# vector 0,1,2, ... number of columsn
	zerothroughcols = np.array(range(data_size))
	
	#######################################################3
	## Now actually run:
	hypset = UniquePriorityQueue(options.TOP_COUNT, max=True) 
	
	learner = GibbsyGriceanVectorizedLexicon()

	for g in LOTlib.MetropolisHastings.gibbs_sample(learner, data, options.STEPS, dimensions=xrange(len(target.all_words()))):
		
		dprintn(10, data_size, g.lp, g.prior, g.likelihood, g.word_idx)
		dprintn(15, g) # and show the lexicon
		hypset.push(g.to_lexicon(), g.lp)
	
	
	# And clear for return pickling
	#[x.value.clear_functions() for x in hypset.Q]
	
	return hypset


########################################################################
## Main loop:
########################################################################

# load the list of lexicons
inh = open(options.IN_PATH)
fs = pickle.load(inh)
my_finite_trees = fs.get_all()
dprintn(5, "# Done loading", len(my_finite_trees), "trees")
if options.ONLY_CONSERVATIVE: my_finite_trees = filter(is_conservative, my_finite_trees)
hyp2target = create_hyp2target(my_finite_trees) # A hash from each of these trees to what target word they match, if any
if options.ONLY_CORRECT: my_finite_trees = filter(lambda x: hyp2target[x] is not None,  my_finite_trees )
dprintn(5, "# Filtered to", len(my_finite_trees), "trees")
#print my_finite_trees


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
	