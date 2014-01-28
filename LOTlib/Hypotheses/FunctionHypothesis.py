from Hypothesis import Hypothesis

from LOTlib.Miscellaneous import *
from LOTlib.DataAndObjects import FunctionData,UtteranceData
from copy import copy, deepcopy

class FunctionHypothesis(Hypothesis):
	"""
		A special type of hypothesis whose value is a function. 
		The function is automatically eval-ed when we set_value, and is automatically hidden and unhidden when we pickle
		This can also be called like a function, as in fh(data)!
	"""
	
	def __init__(self, value=None, f=None, args=['x']):
		"""
			value - the value of this hypothesis
			f - defaultly None, in which case this uses self.value2function
			args - the argumetns to the function
		"""
		self.args = args # must come first since below calls value2function
		Hypothesis.__init__(self,value) # this initializes prior and likleihood variables, so keep it here!
		self.set_value(value,f)
		
	def __copy__(self):
		""" Create a copy, only deeply of of value """
		return FunctionHypothesis(value=copy(self.value), f=self.fvalue, args=self.args)
		
	def __call__(self, *vals):
		""" 
			Make this callable just like a function. Yay python! 
		"""
		try:
			#print self
			return self.fvalue(*vals)
		except TypeError:
			print "TypeError in function call: "+str(self)
			raise TypeError
		except NameError:
			print "NameError in function call: " + str(self)
			raise NameError
	
	def value2function(self, value):
		""" How we convert a value into a function. Default is LOTlib.Miscellaneous.evaluate_expression """
		
		# Risky here to catch all exceptions, but we'll do it and warn on failure
		try:
			return evaluate_expression(value, args=self.args)
		except:
			print "# Warning: failed to execute evaluate_expression on " + v
			return lambdaNone
	
	def reset_function(self):
		""" re-construct the function from the value -- useful after pickling """
		self.set_value(self.value)
		
	
	def set_value(self, value, f=None):
		"""
		The the value. You optionally can send f, and not write (this is for some speed considerations) but you better be sure f is correct
		since an error will not be caught!
		"""
		
		Hypothesis.set_value(self,value)
		
		if f is not None:     self.fvalue = f
		elif value is None:   self.fvalue = None
		else:                 self.fvalue = self.value2function(value)
	
	def get_function_responses(self, data):
		""" 
		Evaluate this function on some data
		Returns a list of my responses to data, handling exceptions (setting to None)
		"""
		
		#return map(lambda di: self(*di.args), data)
	
		out = []
		for di in data:
			#print ">>", di, di.__class__.__name__, type(di), isinstance(di, FunctionData)
			r = None
			try:
				if   isinstance(di, FunctionData):  r = self(*di.args)
				elif isinstance(di, UtteranceData): r = self(*di.context)
				else:                               r = self(*di) # otherwise just pass along
			except RecursionDepthException: pass # If there is a recursion depth exception, just ignore (so r=None)
			
			out.append(r) # so we get "None" when things mess up
		return out
		
	# ~~~~~~~~~
	# Make this thing pickleable
	def __getstate__(self):
		""" We copy the current dict so that when we pickle, we destroy the function"""
		dd = copy(self.__dict__)
		dd['fvalue'] = None # clear the function out
		return dd
	def __setstate__(self, state):
		self.__dict__.update(state)
		self.set_value(self.value) # just re-set the value so that we re-compute the function
