# -*- coding: utf-8 -*-
"""
	This makes a file depend on 
	
	Here, dependencies can be either objects or files. If they are objects, we check to be sure
	that they have not changed (via pickle), and if they are files we use sha5

	Recompute an object if the dependencies (files or objects) change; the object is also parameterized, so that it 
	can depend on some things (like input, etc)

"""

import exceptions

class ObjectLoadFailure(exceptions.Exception):
	def __init__(self, params):
		self.params=params
	def __str__(self):
		return params

try:   x = load_persistent_object("abc", age=33, years=43, dependencies=[a,b,c])
except LoadxFailure:
	
	# Load the file here
	
	save(
	


	

if please_set(x, check_cached("xfile", age=22, years=33)):
	


def param2filename(root, **kwargs):
	out = root
	for w in sorted( kwargs.keys() ):
		out=out+"__"+w+"_"+str(kwargs[w])
	return out
		
def save_paramterized_object(outfile, obj, dependencies, params):
	
	
	
print param2filename("outfile", a="sdf", b=435, c=3222)


		
	