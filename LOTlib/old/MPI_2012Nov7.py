# -*- coding: utf-8 -*-
"""

	TODO: Change this so we can call MPI_map more than once!
	
	We should move to a client/server model where we can call MPI_map and get free workers as they appear. 
	Then, we can use them anywhere int eh prorgam. Everything except the first MPI process is held as a worker. 
	
"""

from mpi4py import MPI
from LOTlib.Miscellaneous import *
import random
from time import time
import numpy as np

from LOTlib.ProgressBar import *

#MPI.Init_thread() 

def MPI_map(f, arglist, timeit=False, random_order=True):
	"""
		Execute, in parallel, a function on each argument, and return the list [x1, f(x1)], [x2, f(x2)].
		
		f -- the function
		arglist -- a list of arguments to apply f to
		time - if True, we report back each time something is completed, how long it took and what the mean is. 
		
		TODO: Implement a "root" argument to specify any processor as the root; 
		TODO: Implement random_order
		
	"""
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	
	# the arguments. This handles null argument lists a little better
	arglistlen = len(arglist)
	
	if rank == 0:
		
		ret = [None] * arglistlen
		
		to_receive = arglistlen  # how many do we have to receive back??
		
		# A little unnecessary, but this keeps track of what's running currently
		running = [False] * size 
		
		## for timing
		start_time = np.array([ 0 ] * arglistlen)
		stop_time  = np.array([ 0 ] * arglistlen)
		time_dist = []
		
		# what arglist element do we send next?
		send_next = 0
		
		# loop until we've heard everything back
		while to_receive > 0:
			
			# TODO: UPDATE THIS TO BE LIKE MPI4PY_MAP WHICH USES MPI.ANY_PROCESS ETC 
			for i in range(1,min(size, arglistlen),1): # run at most the number of arguments in parallel
				if (not running[i]):
					if (send_next < arglistlen):
						dprintn(100, "Main sending ", arglist[send_next], " to ", i)
						#comm.send([idx[send_next], arglist[idx[send_next]]], dest=i, tag=11)
						comm.send([send_next, arglist[send_next]], dest=i, tag=11)
						send_next = send_next + 1
						running[i] = True
						if timeit: start_time[idx[send_next]] = time()
					#else: ## THIS DOES NOT WORK!! WTF??
						# tell them to stop
						 #comm.send(None, dest=i, tag=7) # for return
				if comm.Iprobe(source=i, tag=11): # test for a message
					draw_progress(float(send_next)/float(arglistlen))
					
					ri, r = comm.recv(source=i, tag=11) # get the message
					ret[ri] = r # save it
					running[i] = False # so we send it a new job
					to_receive += -1 # we've gotten back one more
					#if timeit:
						#thetime = time()
						#stop_time[ri] = thetime
						#m = np.mean(stop_time[stop_time > 0]-start_time[stop_time > 0]) # among stoppers, what is the mean? NOTE: This is not quite right sicne stoppers have stopped most quickly!
						
						## Figure out how long this thing is going to take
						#required_time = arglistlen * m # the mean per times total
						#already_run_time = sum(thetime - start_time[stop_time > 0]) + sum(stop_time[stop_time == 0] - start_time[stop_time == 0]) 
						#print "# MPI MAP TIME: ", m, "( N =", len(stop_time[stop_time>0]), "); Max=", np.max(stop_time-start_time)
						##print "# MPI MAP COMPLETE IN APPROX: ", (required_time-already_run_time)/(size*60), "minutes" ## TODO: Compute the estimated time till completion using a smarter method!
		
		# tell everyone to shut the hell down
		for i in range(1,size): comm.send(None, dest=i, tag=7) # for return
		
		return ret
		
	else: 
		while True:
			# test for the exit code
			if comm.Iprobe(source=0, tag=7):
				comm.recv(source=0, tag=7)
				MPI.Finalize() # Fucking shit, if we don't do this everything is terrible, because you get weird exists at random points after this
				sys.exit(0)
			
			# test for a function to evaluate
			if comm.Iprobe(source=0, tag=11):
				i, a = comm.recv(source=0, tag=11) # get our next job
				dprintn(100, rank, " received ", i, a)
				r = f(*a)
				comm.send([i, r], dest=0, tag=11) # send a message that we've finished
	
# Example:


#def f(i):
	#return i ** 1500 # str(i**2)+":"+str(rank)

###if MPI.COMM_WORLD.Get_rank() == 0: 
#r = MPI_map(f, map(lambda x: [x], range(50))) # this many chains

#if rank == 0:
	#print r