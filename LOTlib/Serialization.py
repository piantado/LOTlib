"""
	Just a wrapper for whatever serialization library works best currently. 
	
	Right now it seems to be "cloud" that works best...
"""

import pickle
import cloud

def serialize2file(obj,f):
	with open(f, 'wb') as openfile:
		cloud.serialization.cloudpickle.dump(obj, openfile)
		
def file2object(f):
	with open(f, 'r') as in_file:
		return pickle.load(in_file)
