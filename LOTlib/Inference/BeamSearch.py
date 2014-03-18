	
"""
	Experimental "beam" search over hypotheses. This stores a set which of hypotheses, and to sample
	it samples from that set and then proposes from that. It adds high probability hypotheses to the set
	in order to "adapt" the proposal. However, we may want a form that only adds things that were not
	high probability from anything else in the set, or removes them if they are proposed to too often
	from others in the set (to sparsify things)

"""
from LOTlib.Miscellaneous import *
from LOTlib.FiniteBestSet import *
import numpy
import math
import LOTlib.Miscellaneous

def beam_search(start, data, top=100, samples=1000, reps=100, temperature=1.0, taboo=False):
	"""
		Expands a beam of  top hypotheses. This assumes that each gets a .posterior_score
		- top - the top number of hypotheses to store
		- reps - how many times do we do this?
		- samples - how many to draw from each sample
		- taboo - should we store states we've seen and not evaluate them again? 
			  this costs memory but saves bunches on time
		           
	"""
	
	fs = FiniteBestSet(max=True, N=top)
	
	# Initial generation
	lp = sum(start.compute_posterior(data))
	fs.push(start, lp)
	
	if taboo: 
		taboo_set = set()
			
	for rep in xrange(reps):
		to_add = FiniteBestSet(max=True, N=top) # we can add at most this many
		
		xes = fs.get_all()
		lps = numpy.array([x.posterior_score/temperature for x in xes])
		Z = logsumexp(lps)
		if Z > -Infinity: counts = numpy.exp(lps-Z)*samples # distribute samples according to probability
		else:             counts = [samples / len(lps) ] * len(lps) # distribute evenly
			
		for x, xcnt in zip(xes, counts):
			if math.isnan(xcnt): continue
			for k in xrange(int(xcnt)): # how many times do we propose to this x?
				pp, _ = x.propose()
				
				if (not taboo) or (pp not in taboo_set):
					to_add.push(pp, sum(pp.compute_posterior(data))) # should be faster than checking if we're "in" since its logarithmic
					if taboo: taboo_set.add(pp)  # keep track of this guy
				
				if LOTlib.SIG_INTERRUPTED: break
			if LOTlib.SIG_INTERRUPTED: break
		
		# and put these in
		fs.merge(to_add)
		yield fs
		
		if LOTlib.SIG_INTERRUPTED: break
		
		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
	
	
	from LOTlib.Examples.Number.Shared import *
	
	DATA_SIZE = 200
	
	data = generate_data(DATA_SIZE)

	# A starting hypothesis (later ones are created by .propose, called in LOTlib.MetropolisHastings
	initial_hyp = NumberExpression(G)

	i = 0
	for fs in beam_search(initial_hyp, data, temperature=100.0):
		for x in fs.get_all(sorted=True):
			print i, x.posterior_score, x
			pass
		i += 1
	
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	