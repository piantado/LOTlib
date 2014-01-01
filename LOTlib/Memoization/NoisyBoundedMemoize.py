import numpy # for median

from collections import defaultdict

class NoisyBoundedMemoize(object):
	"""
		Stores at most N entries, approximately keeping the most frequently used ones
		
		When we use a new value, we put it binomially from the beginning.
		When we use it again, we move it forward, again binomially
		This keeps the most used things least likely to be replaced
		
	"""
	
	def __init__(self, F, N=1000, p=0.1):
		self.__dict__.update(locals())
		
		self.array = [None] * N
		
		self.key2value = dict()
		self.key2position = defaultdict(int) # maps a key to a position in the array
		
		# keep our stats!
		self.misses, self.hits = 0,0 
		
	def __call__(self, key):
				
		if key in self.key2value:
			self.hits +=1 
			
			# if it's in key2value, get its position, move it forward
			# binomially (+1), and then update key2position and array
			i = self.key2position[key]
			
			if i < self.N-2: j = i + numpy.random.binomial(self.N-i-2, self.p)+1
			else:            j = i
			
			
			# Swap with the lower neighbor if we're not already tops
			self.array[i], self.array[j] = self.array[j], self.array[i]
			self.key2position[self.array[j]] = j
			self.key2position[self.array[i]] = i				
			
			return self.key2value[key]
		
		else:
			self.misses += 1
			
			# we have space to keep adding -- sample a binomial from the end
			pos = numpy.random.binomial(self.N-1, self.p)
			value = self.F(key)
			
			# delete who was there
			if self.array[pos] is not None:
				del self.key2position[self.array[pos]]
				del self.key2value[self.array[pos]]
			
			#print "Adding ", key, " in ", pos
			self.array[pos] = key
			self.key2value[key] = value
			self.key2position[key] = pos
			
			
			return value
			

	def __setitem__(self, key):
		assert False, "Cannot set item in NoisyBoundedMemoize"


if __name__ == "__main__":

	b = NoisyBoundedMemoize(lambda x: x**100 % 1533456, N=15)

	for i in xrange(100000):
		x = numpy.random.geometric(0.3)
		#print "ACCESSING ", x
		#print b[x]
		y = b(x)
	print b.array



