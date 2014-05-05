"""
	This class defines different types of data for models. 
	It also provides "Obj"s for bundling together features
	
	For functions, we have a data object consisting of the [output, args]

"""
from copy import deepcopy

from LOTlib.Miscellaneous import weighted_sample

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FunctionData:
	"""
		This is a nicely wrapped kind of data--if we give it to a FunctionHypothesis, it knows
		to extract the "input" (NOT the output) and run those on FunctionHypothesis.value(*input)
		So when you have functional hypotheses, this is a convenient form of data
	"""
	
	def __init__(self, input, output, **kwargs):
		assert isinstance(input, list) or isinstance(input, tuple) ## since we apply to this
		self.input = input
		self.output = output
		self.__dict__.update(kwargs)
		
	def __repr__(self): 
		return '<' + ','.join(map(str, self.input)) + " -> " + str(self.output) + '>'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class UtteranceData:
	"""
		A wrapper for utterances. An utterance data point is a word, in a context, with some set of possible words we could have said
	"""
	def __init__(self, word, context, all_words):
		self.__dict__.update(locals())
		
	def __repr__(self):
		return str(self.word)+': '+str(self.context) + " / " + str(self.all_words)
		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Obj:
	""" Represent bundles of features"""
	
	def __init__(self, **f):
		for k, v in f.iteritems():
			setattr(self, k, v)
	
	def __str__(self):
		outstr = '<OBJECT: '
		for k, v in self.__dict__.iteritems():
			outstr = outstr + str(k) + '=' + str(v) + ' '
		outstr = outstr + '>'
		return outstr
		
	def __repr__(self): # used for being printed in lists
		return str(self)
	
	# We may or may not want these, depending on whether we keep Objs in sets...
	#def __eq__(self, other): return str(self) == str(other) 
	#def __hash__(self): return hash(str(self))

def make_all_objects(**f):
	"""
	This takes a list of lists and crosses them into all objects
	e.g. make_all_objects( size=[1,2,3], color=['red', 'green', 'blue'] )
	"""

	keys = f.keys()
	out_objs = []
	
	for vi in f[keys[0]]: 
		out_objs.append(Obj( **{keys[0]: vi} ))
	
	for i in range(1, len(keys)): # for every other key
		newout = []
		for o in out_objs:
			for vi in f[keys[i]]:
				ok = deepcopy(o)
				setattr(ok, keys[i], vi)
				newout.append(ok)
		out_objs = newout
	
	return out_objs


# make a set of size N appropriate to using "set" functions on -- this means it must contain copies, not duplicate references
def sample_sets_of_objects(N, objs):
	s = weighted_sample(objs, N=N, returnlist=True) # the set of objects
	return map(deepcopy, s) # the set must NOT be just the pointers sampled, since then set() operations will collapse them!
	
