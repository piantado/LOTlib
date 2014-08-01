"""
	Routines for evaling pure lambda calculus
	
	TODO: Currently lacking eta-reduction
"""

from copy import copy
from EvaluationException import EvaluationException
from LOTlib.Miscellaneous import lambda_str
MAX_RECURSE_DEPTH = 25
MAX_NODES = 50 # how many is the max, in all stages of eval?

def lambda_reduce(fn):
	# Just a wrapper to copy
	return lambda_reduce_(copy(fn))

def lambda_reduce_(fn, depth=1):
	"""
		Reduce a pure lambda calculus term, applying functions to args on the right via substitution
		
		NOTE: Check that the bound variable naming here isn't messed up
	"""
	#assert isFunctionNode(fn)
	
	if depth > MAX_RECURSE_DEPTH:		
		raise EvaluationException("MAX_RECURSE_DEPTH surpassed!")
	elif fn.count_subnodes() > MAX_NODES:
		raise EvaluationException("Max number of nodes surpassed!")	
	elif fn.name == 'apply_':
		assert len(fn.args)==2
		
		f,x = fn.args
		if f.name != 'lambda':
			#print "\tREDUCING ", copy(f)
			# first try to reduce the inner arg until it's a lambda
			f = lambda_reduce_(f, depth+1) # TODO: Copy necessary?
		
		assert f.name == 'lambda' and len(f.args)==1
		
		below = f.args[0]
		
		# We'll give x a new name
		for xi in x: xi.bv_prefix = 'z'
		x.fix_bound_variables()
		
		#print "\tSUBSTITUTING", x, "-->", f
		for b in [n for n in below]: # must extract all before replacement
			if b.name == f.bv_name: # if we are the right bound variable
				b.setto(copy(x))
		#print "\tTO YIELD", below
		
		# and rename back to y
		for bi in below: bi.bv_prefix = 'y'
		below.fix_bound_variables() 
		
		return lambda_reduce_(below, depth+1)
	elif fn.name == '': # NT->NT expansions
		assert len(fn.args)==1
		return lambda_reduce_(fn.args[0], depth) # same depth
	else:
		return fn


def compose_and_reduce(*args):
	"""
		((args[0] args[1]) args[2]) .. 
		
		This copies each arg, so you don't have to
	"""
	
	## TODO: UGH THIS CIRCULAR IMPORT
	from LOTlib.FunctionNode import FunctionNode
	assert len(args) > 1
	
	fn = FunctionNode('EXPR', 'apply_', [copy(args[0]), copy(args[1])])
	for i in xrange(2,len(args)):
		fn = FunctionNode('EXPR', 'apply_', [fn, copy(args[i])])
	
	try:
		return lambda_reduce(fn)  
	except RuntimeError:
		return None
	

if __name__=="__main__":
	
	## Make a simple grammar for lambda calculus
	from LOTlib.Grammar import Grammar
	
	G = Grammar()
	
	# Here, rules creating smaller lambdas are higher prob; created simpler lambdas are also higher prob
	G.add_rule('START', '', ['EXPR'], 1.0)
	G.add_rule('EXPR', 'lambda', ['EXPR'], 2.0, bv_type='EXPR', bv_args=None, bv_p=2.0) 
	G.add_rule('EXPR', 'apply_', ['EXPR', 'EXPR'], 1.0) 

	# And print some expressions and reduce	
	for _ in xrange(1000):
		t = G.generate()
		
		try:
			print lambda_str(t)
			print lambda_str(lambda_reduce(t))
		except EvaluationException as e:
			print "***", e, lambda_str(t)
		print "\n"
		
		