"""
        A simple case of combinatory categorial grammar for a toy domain.

        This just uses brute force parsing.


        TODO: Learn that MAN is JOHN or BILL

"""

import re

from LOTlib import lot_iter
from LOTlib.Miscellaneous import qq
from LOTlib.FiniteBestSet import FiniteBestSet
from CCGLexicon import CCGLexicon
from Grammar import make_hypothesis



from Data import all_words, data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SAMPLES = 100000

def run(llt=1.0):

    h0 = CCGLexicon(make_hypothesis, words=all_words, alpha=0.9, palpha=0.9, likelihood_temperature=llt)

    fbs = FiniteBestSet(N=10)
    from LOTlib.Inference.MetropolisHastings import mh_sample
    for h in lot_iter(mh_sample(h0, data, SAMPLES)):
        fbs.add(h, h.posterior_score)

    return fbs


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### MPI map
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from SimpleMPI.MPI_map import MPI_map, is_master_process

allret = MPI_map(run, map(lambda x: [x], [0.01, 0.1, 1.0] * 100 ))

if is_master_process():

    allfbs = FiniteBestSet(max=True)
    allfbs.merge(allret)

    H = allfbs.get_all()

    for h in H:
        h.likelihood_temperature = 0.01 # on what set of data we want?
        h.compute_posterior(data)

    # show the *average* ll for each hypothesis
    for h in sorted(H, key=lambda h: h.posterior_score):
        print h.posterior_score, h.prior, h.likelihood, h.likelihood_temperature
        print h

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Play around with some different inference schemes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#h0 = CCGLexicon(make_hypothesis, words=all_words, alpha=0.9, palpha=0.9, likelihood_temperature=0.01)
#for i, h in lot_iter(enumerate(mh_sample(h0, data, 400000000, skip=0, debug=False))):
    #print h.posterior_score, h.prior, h.likelihood, qq(re.sub(r"\n", ";", str(h)))

from LOTlib.Inference.IncreaseTemperatureMH import increase_temperature_mh_sample

h0 = CCGLexicon(make_hypothesis, words=all_words, alpha=0.9, palpha=0.9, likelihood_temperature=0.01)
for i, h in lot_iter(enumerate(increase_temperature_mh_sample(h0, data, 400000000, skip=0, increase_amount=1.50))):
    print h.posterior_score, h.prior, h.likelihood, qq(re.sub(r"\n", ";", str(h)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Run on a single computer, printing out
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#fbs = FiniteBestSet(N=100)
#h0 = CCGLexicon(make_hypothesis, words=all_words, alpha=0.9, palpha=0.9, likelihood_temperature=0.051)
#for i, h in lot_iter(enumerate(mh_sample(h0, data, 400000000, skip=0, debug=False))):
    #fbs.add(h, h.posterior_score)

    #if i%100==0:
        #print h.posterior_score, h.prior, h.likelihood #, re.sub(r"\n", ";", str(h))
        #print h

#for h in fbs.get_all(sorted=True):
    #print h.posterior_score, h.prior, h.likelihood #, re.sub(r"\n", ";", str(h))
    #print h


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Just generate and parse
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#for _ in xrange(1000):

    #cp = h.can_parse(data[3].utterance)
    #if cp:
        #s, t, f = cp
        #print L
        #print s, t, f
        #print f(data[3].context)
        #print "\n\n"
