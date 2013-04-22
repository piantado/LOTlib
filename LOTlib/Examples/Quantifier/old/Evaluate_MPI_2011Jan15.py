# -*- coding: utf-8 -*-
"""
	This runs after we search, taking the top hypotheses that were found and evaluating them on a bunch of new data
	
	This outputs one set of lines for each word, for each amount of data. So when CHAINS=10, 
	
	MPI run:
	$ mpiexec --hostfile ../../hosts.mpich2 -n 25 python2.7-mpi Evaluate_MPI.py
"""

from Shared import *
from mpi4py import MPI
from LOTlib.MPI import * # get our MPI_map function, which will execute run() on as many processors as we pass to mpiexec
from scipy.maxentropy import logsumexp

DATA_RANGE = range(0, 1000, 50)
CHAINS=100
LOAD_HYPOTHESES_PATH = "/home/piantado/Desktop/mit/Libraries/LOTlib/examples/Quantifier/data/gibbs_trees.pkl"
OUT_PATH = "/home/piantado/Desktop/mit/Libraries/LOTlib/examples/Quantifier/data/eval/"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the main sampling function to run

#one run with these parameters
def run(*args):
	dprintn(8, "# Generating data")
	
	global hypotheses
	
	data_size = args[0]
	
	here_correct = dict() # how often is each word right?
	for w in words: here_correct[w] = 0.0
	
	dprintn(8, "# Generating data")
	data = generate_data(data_size)
	
	# recompute these
	dprintn(8, "# Computing posterior")
	[ x.compute_posterior(data) for x in hypotheses ]
	
	# normalize the posterior in fs
	dprintn(8, "# Computing normalizer")
	Z = logsumexp([x.lp for x in hypotheses])
	
	# and compute the probability of being correct
	dprintn(8, "# Computing correct probability")
	for h in hypotheses:
		#print data_size, len(data), exp(h.lp), correct[ str(h)+":"+w ]
		for w in words:
			# the posterior times the prob of agreement with the right one, weighted by number of iterations
			here_correct[w] += exp(h.lp-Z) * correct[ str(h)+":"+w ] 
	
	dprintn(8, "# Outputting")
	o = open(OUT_PATH+str(rank), 'a')
	for w in words:
		print >>o, rank, data_size, here_correct[w], q(w)
	o.close()
	
	return 0
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MPI interface

# The finite set of samples
inh = open(LOAD_HYPOTHESES_PATH)
fs = pickle.load(inh)

hypotheses = fs.get_all()
print rank, ": Loaded pickle. ", len(hypotheses), " hypotheses."

# get all the words
words = target.all_words()

# now figure out how often each meaning is right for each word
correct = dict()
#correct_presup = dict()
for h in hypotheses:
	h.unclear_functions()
	#print h
	for w in words:
		# the proportion of time they agree
		p = float(sum(map(lambda s: ifelse( h.dfunc[w](*s) == target.dfunc[w](*s), 1.0, 0.0), TESTING_SET) )) / float(TESTING_SET_SIZE)
		correct[ str(h)+":"+w ] = p # ifelse(p==1.0,1.0,0)
		
		#print w, "\t",  p, "\t", h.dexpr[w], "\t", target.dexpr[w], "\t"
	# for debugging: print p, "\t", h.dexpr['neither']

print rank, ": Done caching"

# run with null args, this many times
allret = MPI_map(run, [ [x] for x in DATA_RANGE ] * CHAINS ) # pass an array of lists of arguments

print "Complete."

