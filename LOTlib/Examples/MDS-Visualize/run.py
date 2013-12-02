# MDS visualization of theory space

from LOTlib import *
from Shared import *
import numpy
from collections import defaultdict
import itertools

DATA_SIZE = 1000

data = generate_data(DATA_SIZE) ## Data for computing the average ll

# Just load these instead of running:
#time mpiexec -n 10 python Search.py --steps=100000 --chains=10 --dmin=0 --dmax=400 --dstep=25 --mpi --out=tmp.pkl --top=50
allhyp = pickle.load(open("tmp.pkl"))

# Our similarity measure
def structural_similarity(a,b):
	return a.value.proposal_probability_to(b.value) + b.value.proposal_probability_to(a.value)

###############################################################3
## Now compute simliarity between

hyps = allhyp.get_all(sorted=True)

kl2hyp = defaultdict(set)

for h in hyps:
	kl2hyp[get_knower_pattern(h)].add(h)

# so we have a fixed order
kls = sorted(kl2hyp.keys())

o = open("hypotheses.txt", 'w')
print >>o, "lp", "prior", "likelihood", "knower.level"
for a in kls:
	aZ = logsumexp([ x.prior for x in kl2hyp[a] ])
	
	for x in kl2hyp[a]: break # let x be one element of a
	
	print >>o, x.lp, aZ, x.compute_likelihood(data)/len(data), q(get_knower_pattern(x))
o.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Similarity via the proposals
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

o = open("similarity.txt", 'w')
for a in kls:
	if SIG_INTERRUPTED: break
	print >>o, a,
	
	aZ = logsumexp([ x.prior for x in kl2hyp[a] ])
	
	for b in kls:
		if SIG_INTERRUPTED: break
	
		# do the product of all of type a to all of type b, adding up lps
		# weight by the prior (normalized to this set via aZ), which says that if we choose a random element form this set
		# acording to its prior (and thus, posterior since the ll are the same), what is the transition probability?
		p = logsumexp( [ (x.prior-aZ) + x.value.proposal_probability_to(y.value) for x,y in itertools.product( kl2hyp[a], kl2hyp[b] ) ] )
			
		print >>o, p, 
	print >>o, "\n",
o.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Similarity via the proposal
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#o = open("structural-similarity.txt", 'w')
#for i in range(len(hyps)):
	#if SIG_INTERRUPTED: break
	
	## Sum dist version -- make it symmetric!
	## TODO: This is half as fast as it should be!
	
	#print >>o, i, "\t", '\t'.join([ str(structural_similarity(hyps[i], hyps[j])) for j in range(len(hyps))])
#o.close()

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Similarity via log likelihood
## (again, inefficient)
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#def cos_sim(a,b):
	#a = numpy.array(a)
	#b = numpy.array(b)
	
	#return numpy.dot(a,b)/( numpy.sqrt(numpy.dot(a,a)) * numpy.sqrt(numpy.dot(b,b)))

## Store and compute the likelihood
#for h in hyps: h.compute_likelihood(data) 

#o = open("data-similarity.txt", 'w')
#for i in range(len(hyps)):
	#if SIG_INTERRUPTED: break
	
	## Sum dist version -- make it symmetric!
	## TODO: This is half as fast as it should be!
	#print >>o, i, "\t", '\t'.join([ str( cos_sim(hyps[i].stored_likelihood, hyps[j].stored_likelihood)  ) for j in range(len(hyps))])
#o.close()
