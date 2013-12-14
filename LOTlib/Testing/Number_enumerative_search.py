# -*- coding: utf-8 -*-

"""

	Here, we start with a sample of hypotheses, and consider all moves from them, organized in a priority queue. 
	This allows us to search the space much more efficiently, perhaps
	
	Hmm Still not great performance. Really, the amount of time spent on one hypothesis should depend on its probability relative to its competitors. I wonder what would happen if we sampled from the queue (renomalizing), and explored around the sampled guy, adding to the queue again. Then we preferentially sample high probability regions, 
"""

from Number_Shared import *
import LOTlib.EnumerativeSearch

# # # # # # # # # # # # # # # # # # # # # # # # #
# Generate some data

nt_moves = dict() # hash from nonterminals to a *list* of possible moves
for nt in G.nonterminals():
	
	fs = UniquePriorityQueue(N=100, max=True)
	for i in xrange(10000):
		g = G.generate(nt)
		fs.push(g, g.log_probability())
	
	nt_moves[nt] = fs.get_all()
#print "Done generating moves."

def next_states(ne):	
	""" 
		All possible trees we can get from t		
		We loop over subnodes, replacing
	"""
	t = ne.value # extract the trees
	
	#first yeild all the other things of type t
	for tt in nt_moves[t.returntype]:
		yield NumberExpression(tt.copy())
		
	# now try replacing all other nodes
	for tt in t:
		
		# yield everything of the same type
		if tt.returntype == t.returntype:
			yield NumberExpression(tt.copy())
		
		for i in xrange(len(tt.args)):
			
			## here is the proposal to trim out a node of the same type
			#if tt.args[i].returntype == tt.returntype:
				#old_tt = deepcopy(tt)
				#tt.setto(tt.args[i])
				#yield deepcopy(t) # remove each internal node
				#tt.setto(old_tt)
			
			old_a = tt.args[i]
			
			for new_a in nt_moves[old_a.returntype]:
				tt.args[i] = new_a
				yield NumberExpression(t.copy()) # we go down and copy t and the new node
			
			tt.args[i] = old_a



def Number_enumerative_search(data, PRIOR_SCORE_TEMPERATURE=1.0, PRIOR_LIKELIHOOD_TEMPERATURE=1.0, N=10000, breakout=1000, yield_immediate=False):
	"""
		This runs the enumerative search, wrapping the relevant variables
		PRIOR_SCORE_TEMPERATURE -
		PRIOR_LIKELIHOOD_TEMPERATURE - if this is +inf, then we don't use prior temperature
	"""
	## Now we can enumerate!
	start = [ NumberExpression(G.generate('WORD')) for i in xrange(50000) ]
	#print "# Done generating start states."
	
	def score(nh):
		pt = nh.compute_posterior(data)
		return pt[0]/PRIOR_SCORE_TEMPERATURE+pt[1]/PRIOR_LIKELIHOOD_TEMPERATURE
		
	for x,s in LOTlib.EnumerativeSearch.enumerative_search( start , next_states, score, N=100000, breakout=1000, yield_immediate=yield_immediate):
		yield x
		
#############################################################################
#############################################################################

if __name__ == '__main__':
	
	data = generate_data(75)
	print "*** Target likelihood: ", target.compute_likelihood(data)

	print "Starting enumerate __main__:"
	
	for nh in Number_enumerative_search(data, yield_immediate=False):
		print q(get_knower_pattern(nh)), nh.prior, nh.likelihood, nh
	