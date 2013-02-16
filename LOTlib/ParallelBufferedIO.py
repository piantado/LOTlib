from multiprocessing import Process, Queue, current_process
from lockfile import FileLock, LockFailed
import os
import time

DELAY_TIME = 0.1 # How long do I sleep between trying to access the file?

class ParallelBufferedIO:
	"""
		Provides asynchronous output to a file. You can call "write" and a subprocess will handle output buffering so that the 
		main processes does not hang while waiting for a lock. 
		
		NOTE: You MUST close this or else all hell breaks loose with subprocesses
	"""
	
	def __init__(self, path):
		"""
			- path - the output path of this file. Must append to.
		"""
		self.path = path
		self.Q = Queue()
		self.printing_process = Process(target=self.subprocess_loop, args=[])
		self.printing_process.start()
		
	def write(self,*args):
		self.Q.put(args)
		
	def close(self):
		self.Q.put(None)
		
	def subprocess_loop(self):
		"""
			An internal loop my subprocess maintains for outputting
		"""
		
		# convert to a full path and make a lock
		path = os.path.realpath(self.path)
		lock = FileLock(self.path)
		
		while True:
			time.sleep(DELAY_TIME)
			if (not self.Q.empty()):
				lock.acquire() # get the lock (or wait till we do)
				with open(self.path, 'a') as o:
					while not self.Q.empty(): # dump the entire queue
						x = self.Q.get()
						if x is None: # this is our signal we are done with input 
							lock.release()
							return
						else: # this
							for xi in x: print >>o, xi,
							print >>o, "\n",
				lock.release()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__=="__main__":
	
	## Can run this with: mpiexec -n 10 python ParallelBufferedIO.py
	
	bo = ParallelBufferedIO("/tmp/pbio.txt")
	bo.write("Hi there", "pal!")
	bo.close()
	
	quit()
	
	