#-*- coding: utf-8 -*-


class BoundedDictionary:
	"""
		This provides the back end for memoizing, storing the most recently used, 
		and cleaning up by half when we exceed N>2
	"""
	def __init__(self, N=1000):
		self.__dict__.update(locals())
		
		self.ret_hash = dict() # store the return values
		self.last_count = dict() # when were we last used?
		self.counter = 0 # what use are we on?
		self.dict_size = 0 # how many entries?
		
	def __len__(self): return self.dict_size
	def str(self):     return str(self.ret_hash)
	def repr(self):    return repr(self.ret_hash)
	
	
	def __setitem__(self, k, v):
		
		if k not in self.ret_hash:
			self.dict_size += 1
		
		# set it
		self.ret_hash[k] = v
		self.last_count[k] = self.counter
		
		# and clean up if we are too large
		if self.dict_size > 2 * self.N:
			
			v = sorted(self.last_count.values())
			median = v[len(v)/2]
			
			keys = self.last_count.keys()
			for k in keys:
				if self.last_count[k] <= median: # if you were used less recently
					del self.last_count[k]
					del self.ret_hash[k]
					self.dict_size -= 1
	
	def __getitem__(self, k):
		self.last_count[k] = self.counter
		self.counter += 1
		return self.ret_hash[k] # MUST use this form so we raise an exception if k not in ret_hash (for use by Memoize)
	
	def __contains__(self, k):	
		return (k in self.ret_hash)
		



	
