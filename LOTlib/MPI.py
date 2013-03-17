# -*- coding: utf-8 -*-
"""

	The MPI interface for LOTlib
	
	So, slave processes are caught either by calling capture_slaves(), or the first time you call MPI_map. In either case, they wait around indefinitely until told to exit
	
"""

from mpi4py import MPI
from LOTlib.Miscellaneous import *
import random
import LOTlib
from LOTlib.ProgressBar import *
from LOTlib.ParallelBufferedIO import *

# Tags for message passing
SYNCHRONIZE_TAG = 0x2
EXIT_TAG = 0x1
RUN_TAG = 0x0 
MASTER_PROCESS = 0 # what process is the master node?

SLAVE_RETURN = None # what do slaves return when they complete?

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
out = None


# Make sure we always finalize when we exit, or else all hell breaks loose
import atexit # for running things at exit
def myexit():
	global out
	if out is not None: out.close() # close this -- meaning the subprocess is told to stop 
	MPI.Finalize()
atexit.register(myexit)

def is_master_process():
	return rank == MASTER_PROCESS



def synchronize_variable(f):
	"""
		Evaluate f on the head node to yield a variable value, and then send that to everything. 
		So in our code, we can say 
		
		y = synchornize_variable( lambda : random.random())
	"""
	if rank == MASTER_PROCESS:
		ret = f() # evaluate
		for i in xrange(size): # send this to each process
			if i != MASTER_PROCESS: comm.send( ret, dest=i, tag=SYNCHRONIZE_TAG)
		return ret
	else:
		while True:
			# test for the exit code
			if comm.Iprobe(source=MASTER_PROCESS, tag=SYNCHRONIZE_TAG):
				return comm.recv(source=MASTER_PROCESS, tag=SYNCHRONIZE_TAG)


def worker_process(outfile=None):
	"""
		This implements a worker process who is sitting to receive
	"""
	global out # make this global so that myexit can call it
	assert rank != MASTER_PROCESS # better not have master processes in here!
	
	if outfile is not None:
		out = ParallelBufferedIO(outfile)
	
	while True:
		# test for the exit code
		if comm.Iprobe(source=MASTER_PROCESS, tag=EXIT_TAG):
			comm.recv(source=MASTER_PROCESS, tag=EXIT_TAG)
			
			print >>sys.stderr, "# Process exiting ", rank
			# We must exit
			MPI.Finalize()
			sys.exit(0)
		
		# test for a function to evaluate
		if comm.Iprobe(source=MASTER_PROCESS, tag=RUN_TAG):
			f, i, a = comm.recv(source=MASTER_PROCESS, tag=RUN_TAG) # get our next job
			dprintn(100, rank, " received ", i, a)
			r = f(*a)
			if outfile is not None: out.write(*r) # non-blocking write to a file if we want it
			comm.send([i, r], dest=MASTER_PROCESS, tag=RUN_TAG) # send a message that we've finished	
	
	if outfile is not None:
		out.close()
	
def capture_slaves(outfile=None):
	"""
		You can call this on an MPI process, to start the slave processign requests.
		It is called defaultly the first time you call MPI_map, but it can be called earlier in the program if you have some data processing that not all slaves should do
		
		outfile - an output file for the slave to print to. This is thread-safe
	"""
	if rank != MASTER_PROCESS: 
		worker_process(outfile=outfile)

def MPI_done():
	# Tell all to exit from this map (Not exit overall)
	for i in range(1,size): 
		print >>sys.stderr, "# Master calling exit on ", i
		comm.send(None, dest=i, tag=EXIT_TAG) 
		

# Let's make a new kind, where we spawn up to the length in order to process, each time we see an MPI_map
#http://mpi4py.scipy.org/docs/usrman/tutorial.html
def MPI_map(f, args, random_order=True, outfile=None, mpi_done=False):
	"""
		Execute, in parallel, a function on each argument, and return the list [x1, f(x1)], [x2, f(x2)].
		
		f -- the function. Must be defined using "def" (not lambda) or else the slaves can't see it
		args -- a list of arguments to apply f to
		random_order - should we evaluate in a randomized order (in order to keep progress bar honest)
		outfile - should reuslts be printed nicely in paralle ot outfile?
		mpi_done - if True, we tell all subprocesses to die. This is handy if you only have one MPI_map, and
		therefore don't need the processes again
	"""
	
	if size == 1: 
		print >>sys.stderr, "# *** NOTE: 'MPI_map' running as 'map' since size=1!"
		return map(lambda x: f( *listifnot(x)), args)
		
	arglen = len(args)
	started_count = 0 # how many things do we need to get back?
	completed_count = 0
	ind = range(arglen)
	ret = [None]*arglen # the return values
	running = [False]*size  # which processes are running?
	
	if random_order: random.shuffle(ind)
	
	# calling this is a sink that all slave processes fall into, waiting
	capture_slaves(outfile)
	
	# Now only the master process survives:
	try:
		while completed_count < arglen:
			for i in range(1,min(size,arglen),1): # run at most the number of arguments in parallel
				if (not running[i]) and (started_count < arglen):
					dprintn(100, "# Main sending ", args[ind[started_count]], " to ", i)
					comm.send( [f, ind[started_count],  listifnot(args[ind[started_count]]) ], dest=i, tag=RUN_TAG)
					started_count += 1
					running[i] = True
				if comm.Iprobe(source=i, tag=RUN_TAG): # test for a message
					completed_count += 1 # we've gotten back one more
					draw_progress(float(completed_count)/float(arglen))
					
					ri, r = comm.recv(source=i, tag=RUN_TAG) # get the message
					ret[ri] = r # save it
					running[i] = False # so we send it a new job
				
		# if we don't need the slave processes anymore
		if mpi_done: MPI_done()			
	except: 
		MPI_done() # shut down everyone (on, e.g., interrupt, etc)
		raise
	
	print >>sys.stderr, "\n" # since we drew the progress bar!
	return ret
		
if __name__=="__main__":
	#LOTlib.Miscellaneous.DEBUG_LEVEL = 10

	def f(i):
		return [i, i ** 15000]

	r = MPI_map(f, map(lambda x: [x], range(500)), outfile="/tmp/mpitest.txt") # this many chains

	#print r
	sys.exit(0)
	