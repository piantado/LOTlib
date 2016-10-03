import pickle

try:
    import numpy
except ImportError:
    import numpypy as numpy

def build_conceptlist(c,l):
    return "CONCEPT_%s__LIST_%s.txt"%(c,l)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up some logging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from LOTlib.Miscellaneous import setup_directory, qq

LOG = "log"
setup_directory(LOG) # make a directory for ourselves

# Make sure we print the entire matrix
numpy.set_printoptions(threshold=numpy.inf)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the hypotheses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# map each concept to a hypothesis
with open('hypotheses/hypotheses-1.pkl', 'r') as f:
# with open('hypotheses/hypotheses-1.pkl', 'r') as f:
    concept2hypotheses = pickle.load(f)

hypotheses = set()
for hset in concept2hypotheses.values():
    hypotheses.update(hset)

print "# Loaded %s hypotheses" % len(hypotheses)
with open(LOG+"/hypotheses.txt", 'w') as f:
    for i, h in enumerate(hypotheses):
        print >>f, i, qq(h)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the human data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# We will map tuples of concept-list, set, response to counts.
import pandas
import math
from collections import Counter
human_data = pandas.read_csv('HumanData/TurkData-Accuracy.txt', sep='\t', low_memory=False, index_col=False)
human_yes, human_no = Counter(), Counter()
for r in xrange(human_data.shape[0]): # for each row
    cl = build_conceptlist(human_data['concept'][r], human_data['list'][r])
    rsp = human_data['response'][r]
    rn =  human_data['response.number'][r]
    sn = human_data['set.number'][r]
    key = tuple([cl, sn, rn ])
    if rsp == 'F':   human_no[key] += 1
    elif rsp == 'T': human_yes[key] += 1
    elif math.isnan(rsp): continue # a few missing data points
    else:
        assert False, "Error in row %s %s" %(r, rsp)
print "# Loaded human data"

from Model.Data import concept2data
print "# Loaded concept2data"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the rule count matrices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Inference.GrammarInference.Precompute import create_counts

from Model.Grammar import grammar

trees = [h.value for h in hypotheses]

nt2counts, sig2idx, prior_offset = create_counts(grammar, trees, log=LOG)

print "# Computed counts for each hypothesis & nonterminal"

# print counts

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build up the info about the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

L        = [] # each hypothesis's cumulative likelihood to each data point
NYes     = []
NNo      = []
output   = []

for concept in concept2data.keys()[:50]: ## TODO: DO THIS FOR ALL
    print "#\t Doing concept %s" % concept
    data = concept2data[concept]

    # compute the likelihood for all the data here
    for h in hypotheses:
        h.stored_likelihood = [h.compute_single_likelihood(d) for d in data]

    for di in xrange(25):
        print "Doing ", concept, " ", di

        ll  = [ sum(h.stored_likelihood[:di]) for h in hypotheses ] # each likelihood
        out = [ map(lambda x: 1*h(data[di].input, x), data[di].input) for h in hypotheses] # each response

        # Each out is a set of responses, that we have to unwrap
        for oi in xrange(len(data[di].input)):
            k = tuple([concept, di+1, oi+1])

            # assert k in human_yes and k in human_no, "*** Hmmm key %s is not in the data" % k
            if k in human_yes or k in human_no:
                output.extend( [x[oi] for x in out] ) # each hypothesis's prediction for this item
                L.extend( ll ) # add all the likelihoods on
                NYes.append( human_yes[k]  )
                NNo.append( human_no[k] )
            else:
                print "*** Warning, %s not in human_yes or human_no"%str(k)


print "# Created L, NYes, NNo, and Output of size %s" % len(L)
assert len(L) == len(output)
assert len(NYes)==len(NNo)


from struct import pack

# stack together counts
kys = nt2counts.keys()
ntlen = [len(nt2counts[k][0]) for k in kys] # store how long each
counts = numpy.concatenate([nt2counts[k] for k in kys], axis=1)

print "const int NRULES = %s;" % sum(ntlen)
print "const int NHYP = %s;" % len(hypotheses)
print "const int NDATA = %s;" % len(NYes)
print "const int NNT = %s;" % len(ntlen)
print "const int NTLEN[NNT] = {%s};" % ','.join(map(str,ntlen))

import h5py
with h5py.File('data.h5', 'w') as hf:
    # first write the 'specs'
    hf.create_dataset('specs',    data=[len(hypotheses), sum(ntlen), len(NYes), len(ntlen)], dtype=int)
    # write all the data
    hf.create_dataset('counts',    data=numpy.ravel(counts, order='C'), dtype=int)
    hf.create_dataset('output',    data=output, dtype=float)
    hf.create_dataset('human_yes', data=numpy.ravel(NYes, order='C'), dtype=int)
    hf.create_dataset('human_no',  data=numpy.ravel(NNo, order='C'), dtype=int)
    hf.create_dataset('likelihood',  data=numpy.ravel(L, order='C'), dtype=float)
    hf.create_dataset('ntlen', data=ntlen, dtype=int)
