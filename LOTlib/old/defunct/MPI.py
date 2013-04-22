# -*- coding: utf-8 -*-

from mpi4py import MPI
import time

def MPI_map(f, arglist, returnarg=True):
	"""
		Execute, in parallel, a function on each argument, and return the list [x1, f(x1)], [x2, f(x2)], etc. but in an 
		arbitrary order. This queues its arguments and shuffles them out as processes finish
		
		f -- the function
		arglist -- a list of arguments to apply f to
		returnarg -- if True (default) we return [a, f(a)], else just f(a)
		
		TOOD: Implement a "root" argument
		
	"""
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	
	#print size,rank
	
	# the arguments. This handles null argument lists a little better
	arglistlen = len(arglist)
	if arglistlen <= 0: return
	
	if rank == 0:
		
		ret = []
		
		to_receive = arglistlen  # how many do we have to receive back??
		
		# A little unnecessary, but this keeps track of what's running currently
		running = [False] * arglistlen 
		
		# what arglist element do we send next?
		send_next = 0
		
		# loop until we've heard everything back
		while to_receive > 0:
			#print to_receive, size
			#print size
			for i in range(1,min(size, arglistlen),1): # run at most the number of arguments in parallel
				print "Checking ", i, "Running = ", running[i]
				if running[i] and comm.Iprobe(source=i, tag=11): # test for a message
				
					r = comm.recv(source=i, tag=11) # get the message
					ret.append( r ) # save it
					running[i] = False # so we send it a new job
					to_receive = to_receive - 1 # we've gotten back one more
					#continue # give it a break, so don't start immediately here again
					#print "Main received ", r, " from ", i
				
				time.sleep(1)
				#time.sleep(1)
				#print "Running:", running[i], send_next 
				if not running[i]:
					
					if send_next < arglistlen: # more work to be done
					
						print "Main sending ", arglist[send_next], " to ", i
						comm.isend(arglist[send_next], dest=i, tag=11)
						print "sent"
						send_next = send_next + 1
						running[i] = True
					else: 
						comm.isend(None, dest=i, tag=11)
						running[i] = True # don't check this again
				
						
			#data = [(i+1)**2 for i in range(25) ]
			
		return ret
	else: 
		while True:
			time.sleep(1)
			#print rank, " waiting for message "
			if comm.Iprobe(source=0, tag=11): # test for a message
				print "*****", rank, " probe passed"
				a = comm.recv(source=0, tag=11) # get our next job
				print "*****", rank, " received ", a
				if a == None: 
					return # we are done
				else:
					print "Running:", a, " on ", rank
					fa = f(*a) # apply f to it
					
					if returnarg: r = [a, fa] # what we return
					else:         r = fa
					
					comm.isend(r, dest=0, tag=11) # send a non-blocking message that we've finished

	
# Example:

#def myf(x, y):
	#return x+y
	
#x = MPI_map(myf, [  [1,2], [3,4], [9,4], [1,4], [5,6], [7,8], [1,3], [4,7]  ])

#if MPI.COMM_WORLD.Get_rank() == 0: 
	#print x