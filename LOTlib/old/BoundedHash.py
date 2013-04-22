
class BoundedHash:
		"""
			This provides the back end for memoizing, storing the most recently used
			TODO: update it so that it can also store the top(?)
		"""
		def __init__(self, N=1000):
			self.ret_hash = dict()
			self.last_count = dict()
			self.counter = 0
			self.dict_size = 0
			self.N = N
			
		def set(self, k, v):
			
			if k not in self.ret_hash:
				self.dict_size += 1
			
			# set it
			self.ret_hash[k] = v
			
			# and clean up if we are too large
			if self.dict_size > 2 * self.N:
				
				v = self.last_count.values()
				median = numpy.median( v )
				
				self.dict_size = len(v) / 2
				for k in self.last_count.keys():
					if self.last_count[k] < median: # if you were used less recently
						del self.last_count[k]
						del self.ret_hash[k]
		
		def get(self, k, default=None):
			self.last_count[k] = self.counter
			self.counter += 1
			return self.ret_hash.get(k, default)
		
		def contains(self, k):	
			return (k in self.ret_hash)
			


class BoundedMemoize:
	
	def __init__(self, N=10000):
		self.hsh = BoundedHash(N)
		
	def __call__(self, f):
		
		# the function we create
		def newf(*args):
			
			if self.hsh.contains(args):
				return self.hsh.get(args)
			else:
				r = f(*args)
				self.hsh.set(args, r)
				return r
			
		# and return this new fancy function
		return newf
# a dictionary class that can be bounded in size -- for memoization that doesn't gobble too much memory
# this stores between N and 2N hash items, trimming in half when it exceeds 2N by who was used most recently
#class BoundedMemoize:
	
	#def __init__(self, N=10000):
		#self.ret_hash = dict()
		#self.last_count = dict()
		#self.counter = 0
		#self.dict_size = 0
		#self.N = N
	
	#def __call__(self, f):
		
		## the function we create
		#def newf(*args):
			
			#self.last_count[args] = self.counter
			#self.counter += 1
			
			#if args in self.ret_hash:
				##print "\tHIT", args
				##print  self.ret_hash[args], args
				#return self.ret_hash[args]
			#else:
				##print "MISS", args
				#r = f(*args)
				##print "DONE"
				#self.ret_hash[args] = r
				#self.dict_size += 1
				
				## and clean up if we are too large
				#if self.dict_size > 2 * self.N:
					
					#v = self.last_count.values()
					#median = numpy.median( v )
					
					#self.dict_size = len(v) / 2
					#for k in self.last_count.keys():
						#if self.last_count[k] < median: # if you were used less recently
							#del self.last_count[k]
							#del self.ret_hash[k]
				#return r
				
		## and return this new fancy function
		#return newf