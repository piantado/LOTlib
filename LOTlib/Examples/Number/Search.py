# -*- coding: utf-8 -*-

"""
This runs the number model on the laptop or an MPI cluster. 
This is the primary implementation intended for replication of Piantadosi, Tenebaum & Goodman
To install on my system, I had to build mpich2, mpi4py and set up ubunut with the following: 
	https://help.ubuntu.com/community/MpichCluster

To run on MPI:
$time mpiexec -hostfile /home/piantado/Desktop/mit/Libraries/LOTlib/hosts.mpich2 -n 36 python Search.py --steps=10000 --top=50 --chains=25 --large=1000 --dmin=0 --dmax=300 --dstep=10 --mpi --out=/home/mpiu/tmp.pkl
"""

from Shared import *

## Parse command line options:
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--out", dest="OUT_PATH", type="string",
                  help="Output file (a pickle of FiniteBestSet)", 
                  default="/home/piantado/Desktop/mit/Libraries/LOTlib/LOTlib/Examples/Number/mpi-run.pkl")
         
parser.add_option("--steps", dest="STEPS", type="int", default=200000,
                  help="Number of Gibbs cycles to run")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=1000,
                  help="Top number of hypotheses to store")
parser.add_option("--chains", dest="CHAINS", type="int", default=1,
                  help="Number of chains to run (new data set for each chain)")
parser.add_option("--large", dest="LARGE_DATA_SIZE", type="int", default=-1,
                  help="If > 0, recomputes the likelihood on a sample of data this size")
                  
parser.add_option("--data", dest="DATA", type="int",default=-1,
                  help="Amount of data")
parser.add_option("--dmin", dest="DATA_MIN", type="int",default=25,
                  help="Min data to run")
parser.add_option("--dmax", dest="DATA_MAX", type="int", default=500,
                  help="Max data to run")
parser.add_option("--dstep", dest="DATA_STEP", type="int", default=25,
                  help="Step size for varying data")
                  
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

# manage how much we print
if options.QUIET: options.DEBUG_LEVEL = 0
LOTlib.Miscellaneous.DEBUG_LEVEL = options.DEBUG_LEVEL

# And echo the command line options for a record
#dprintn(5, "# Running: ", options)

if options.RUN_MPI:
	from SimpleMPI.MPI_map import MPI_map

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# the main sampling function to run

def run(data_size):
	"""
		This runs on the DATA_RANGE amounts of data and returns *all* hypothese in the top options.TOP_COUNT
	"""
	#print "Running ", MPI.COMM_WORLD.Get_rank()
	hyps = FiniteBestSet(max=True, N=options.TOP_COUNT) 
	
	# initialize the data
	data = generate_data(data_size)
	
	# starting hypothesis -- here this generates at random
	initial_hyp = NumberExpression(G)
	
	## populate the finite sample by running the sampler for this many steps
	for h in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, options.STEPS, skip=0):
		hyps.push(h, h.lp)
		dprintn(15, data_size, h.lp, h) # and show the lexicon
		
	return hyps
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Main running

# choose the appropriate map function
if options.RUN_MPI:
	allret = MPI_map(run, map(lambda x: [x], options.DATA_AMOUNTS * options.CHAINS)) 
	# Only the master process escapes MPI_map
else:
	allret = map(run,  options.DATA_AMOUNTS * options.CHAINS)

# Handle all of the output
allfs = FiniteBestSet(max=True)
allfs.merge(allret)
allfs.save(options.OUT_PATH) # save this in a file

## If we want to print the summary with the "large" data size (average posterior score computed empirically)
if options.LARGE_DATA_SIZE > 0:
	
	#now evaluate on different amounts of data too:
	huge_data = generate_data(options.LARGE_DATA_SIZE)
	
	H = allfs.get_all()
	[h.compute_posterior(huge_data) for h in H]
	
	# show the *average* ll for each hypothesis
	for h in H:
		if h.prior > float("-inf"):
			print h.prior, h.likelihood/float(options.LARGE_DATA_SIZE), q(get_knower_pattern(h)),  q(h) # a quoted x
