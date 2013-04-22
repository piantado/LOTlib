# -*- coding: utf-8 -*-
from copy import deepcopy

from LOTlib.Miscellaneous import *

class Obj:
	
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

# this takes a list of lists and crosses them into all objects
# e.g. make_all_objects( size=[1,2,3], color=['red', 'green', 'blue'] )
def make_all_objects(**f):
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
	

## Testing:
#for s in make_all_objs(size=[1,2,3], color=['red', 'green', 'blue'], texture=['white', 'hot']):
	#print s

		
		