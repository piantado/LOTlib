	
"""
	Like BeamSearch.py but we remove things that are proposed to too often from others in the set

"""
from LOTlib.Miscellaneous import *
from LOTlib.FiniteBestSet import *
import numpy


def adaptive_beam_search(initial_hyp, data, threshold=100.0, samples=1000, reps=100, temperature=100.0):
	"""
		Expands a beam of  top hypotheses. This assumes that each gets a .posterior_score
		- top - the top number of hypotheses to store
		- reps - how many times do we do this?
		- samples -- how many to draw from each sample
	"""
	
	beam = list() # instead of a FiniteBestSet
	
	# Initial generation
	lp = sum(initial_hyp.compute_posterior(data))
	beam.append(initial_hyp)
	
	for rep in xrange(reps):
		
		lps = numpy.array([x.posterior_score/temperature for x in beam])
		Z = logsumexp(lps)
		counts = numpy.exp(lps-Z)*samples # distribute samples according to probability
		toadd = []
		
		for x, xcnt in zip(beam, counts):
			if LOTlib.SIG_INTERRUPTED: break
			for k in xrange(int(xcnt)): # how many times do we propose to this x?
				if LOTlib.SIG_INTERRUPTED: break
				pp, _ = x.propose()
				
				if pp not in beam and pp not in toadd:
					if LOTlib.SIG_INTERRUPTED: break
					pp_posterior = sum(pp.compute_posterior(data))/temperature
					
					# see how likely we are to get to this:
					from_bi = map(lambda bi: G.lp_regenerate_propose_to(bi, pp), beam)
					
					from_all_bs = logsumexp( from_bi + lps-Z )
					
					# if we are much better than our proposal prob, add us
					
					#pp_posteiror-Z is how much more prob mass then the current beam, 
					# plus the probability of getting to it from the beam
					#print pp_posterior, Z, from_all_bs
					if (pp_posterior-Z) - from_all_bs > threshold:
						#print "# ADDING", pp_posterior, x.posterior_score, from_all_bs, "\n", pp, "\n", x
						toadd.append(pp)
		# and put these in
		beam.extend(toadd)
	
		yield beam
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
	
	
	from LOTlib.Examples.Number.Shared import *
	
	DATA_SIZE = 200
	
	data = generate_data(DATA_SIZE)

	# A starting hypothesis (later ones are created by .propose, called in LOTlib.MetropolisHastings
	initial_hyp = NumberExpression(G)
	
	i = 0
	for beam in adaptive_beam_search(initial_hyp, data):
		for x in sorted(beam, key=lambda z: z.posterior_score):
			print i, x.posterior_score, x
		i += 1
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	