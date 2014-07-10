"""
	A special type of hypothesis whose value is a function. 
	The function is automatically eval-ed when we set_value, and is automatically hidden and unhidden when we pickle
	This can also be called like a function, as in fh(data)!
"""



from Hypothesis import Hypothesis

from LOTlib.Evaluation.Eval import evaluate_expression
from LOTlib.Evaluation.EvaluationException import EvaluationException
from LOTlib.Miscellaneous import lambdaNone
from LOTlib.DataAndObjects import FunctionData
from copy import copy

class FunctionHypothesis(Hypothesis):
	"""
		A special type of hypothesis whose value is a function. 
		The function is automatically eval-ed when we set_value, and is automatically hidden and unhidden when we pickle
		This can also be called like a function, as in fh(data)!
	"""
	
	def __init__(self, value=None, f=None, args=['x'], **kwargs):
		"""
			*value* - the value of this hypothesis

			*f* - defaultly None, in which case this uses self.value2function

			*args* - the arguments to the function
		"""
		self.args = args # must come first since below calls value2function
		Hypothesis.__init__(self, value, **kwargs) # this initializes prior and likleihood variables, so keep it here!
		self.set_value(value,f)
		
	def __copy__(self):
		""" Create a copy, only deeply of f value """
		return FunctionHypothesis(value=copy(self.value), f=self.fvalue, args=self.args)
		
	def __call__(self, *vals):
		""" 
			Make this callable just like a function (as in: myFunction(data)). Yay python!
		"""
		assert not any([isinstance(x, FunctionData) for x in vals]), "*** Probably you mean to pass FunctionData.input instead of FunctionData?"
		assert callable(self.fvalue)
		#print "CALL", str(self), vals
		
		try:
			return self.fvalue(*vals)
		except EvaluationException:
			return None
		except TypeError:
			print "TypeError in function call: "+str(self)+"  ;  "+str(vals)
			raise TypeError
		except NameError:
			print "NameError in function call: " + str(self)
			raise NameError
	
	def value2function(self, value):
		""" How we convert a value into a function. Default is LOTlib.Miscellaneous.evaluate_expression """
		
		# Risky here to catch all exceptions, but we'll do it and warn on failure
		try:
			return evaluate_expression(value, args=self.args)
		except Exception as e:
			print "# Warning: failed to execute evaluate_expression on " + str(value)
			print "# ", e
			return lambdaNone
	
	def reset_function(self):
		""" re-construct the function from the value -- useful after pickling """
		self.set_value(self.value)
		
	
	def set_value(self, value, f=None):
		"""
			Sets the value for the hypothesis. 
			Another option: send f, and not write (this is for some speed considerations) but you better be sure f is correct
			since an error will not be caught!
		"""
		
		Hypothesis.set_value(self,value)
		
		if f is not None:     self.fvalue = f
		elif value is None:   self.fvalue = None
		else:                 self.fvalue = self.value2function(value)
	
	def force_function(self, f):
		self.value = "<FORCED_FUNCTION>"
		self.fvalue=f
	
	def compute_single_likelihood(self, datum):
		"""
			A function that must be implemented by subclasses to compute the likelihood of a single datum/response pair.
			This should NOT implement the temperature (that is handled by compute_likelihood)
		"""
		raise NotImplementedError
	
	def compute_likelihood(self, data):
		"""
			Computes the likelihood. Pretty self-explanatory :)
		"""
		self.likelihood = sum(map( self.compute_single_likelihood, data)) / self.likelihood_temperature
		
		self.posterior_score = self.prior + self.likelihood
		return self.likelihood


	# ~~~~~~~~~
	# Make this thing pickleable
	
	def __getstate__(self):
		""" We copy the current dict so that when we pickle, we destroy the function"""
		dd = copy(self.__dict__)
		dd['fvalue'] = None # clear the function out
		return dd

	def __setstate__(self, state):
		"""
			sets the state of the hypothesis (when we unpickle)
		"""
		self.__dict__.update(state)
		self.set_value(self.value) # just re-set the value so that we re-compute the function
