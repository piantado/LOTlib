"""

	Some search algorithms
"""
import itertools

def grid_search(f, *lims):
	"""
		Simple/stupid grid search over all options to find the best value
		f - objective function to be MAXIMIZED
		lims - arrays of possible values for each argument to f (in order)
	"""
		
	best_value = float("-inf")
	best_params = None
	
	print lims
	for x in itertools.product(*lims):
		#print x
		v = f(*x)
		if v > best_value:
			best_value = v
			best_params = x
			
	return best_params
	


	
# Testing debug:
if __name__ == '__main__':
	
	x = [-1,0,1]
	y = [-2,0,2]
	
	print grid_search( lambda a,b: a*a*b-a, x, y)
	