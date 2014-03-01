
class Memoize(dict):
	"""
		Provide a memoization class
		This inherits from dictionary, so if we subclass as something else, we will get other dictionary
		properties. This acts like a dictionary (inhering from dict) and a function
	"""
	
	def __init__(self, f):
		self.__dict__.update(locals())
	
	def __call__(self, *args):
		try: 
			return self[args]
		except KeyError:
			r = self.f(*args)
			self[args] = r
			return r



from BoundedDictionary import BoundedDictionary
class BoundedMemoize(BoundedDictionary, Memoize):
	def __init__(self, f, N=1000):
		Memoize.__init__(self, f)
		BoundedDictionary.__init__(self, N)



if __name__ == '__main__':
	
	bd = BoundedMemoize(lambda x,y: 1, N=1)
	
	print bd('a', 'b')
	print bd('x', 'y')
	print bd('f', 'g')
	
	print bd.ret_hash