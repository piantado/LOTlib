# -*- coding: utf-8 -*-
import numpy as np

class HashableArray():
	"""
		A class for image hypotheses that initializes to square zeros
		and has a much faster hashing function. This basically just
		wraps a numpy array
	"""
	
	def __init__(self, ar=None, shape=(5,5), shallow=True, hsh=None):
		"""
			If shallow, we copy the array (e.g. if it is going to be modified)
			
			TODO: If hsh==None, then we don't return a strung, we just return hsh. This way, we can construct these
			really efficiently, giving a finite set their own hsh values. So hsh is an identifier that really speeds up hashing
		"""
		if ar != None:
			if shallow: self.ar = ar
			else:       self.ar = deepcopy(ar)
		else: self.ar = np.zeros( shape, dtype=int )
		
		# make this accessible to everyone
		self.shape = self.ar.shape	
		
		
	def __getitem__(self, *args):
		return self.ar.__getitem__(*args)
	def __setitem__(self, *args):
		return self.ar.__setitem__(*args)
	
	## TODO: Optimize this--a lot depends on how well we hash these
	def __hash__(self):
		#return self.ar.__repr__().__hash__()
		#return self.ar__str__().__hash__()
		#return hashlib.sha1(self.ar).hexdigest().__hash__()
		#if self.hsh == None:
		return self.ar.dumps().__hash__()
		#else return self.hsh

	def __eq__(self, y):
		#if hsh == None:
		return np.equal(self.ar, y.ar).all()
		#else:
			#return self.hsh == y.hsh
	
	def __str__(self):
		return self.ar.__str__()
