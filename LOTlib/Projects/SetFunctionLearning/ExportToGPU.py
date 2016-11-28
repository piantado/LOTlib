import pickle
from LOTlib import break_ctrlc
import LOTlib # if we want to set LOTlib.SIG_INTERRUPTED=False for resuming ipython
import numpy

def build_conceptlist(c,l):
    return "CONCEPT_%s__LIST_%s.txt"%(c,l)

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

from LOTlib.GrammarInference.Precompute import create_counts

from Model.Grammar import grammar

trees = [h.value for h in hypotheses]

nt2counts, sig2idx, prior_offset = create_counts(grammar, trees, log=None)

print "# Computed counts for each hypothesis & nonterminal"

# print counts

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build up the info about the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

concepts = concept2data.keys()

NYes     = []
NNo      = []

# Make NYes and Nno
for c in concepts:
    data = concept2data[c]

    for di, d in enumerate(data[:25]): # as far as we go into the trial
        for ri in xrange(len(d.input)): # each individual response
            k = tuple([c, di+1, ri+1])
            # assert k in human_yes and k in human_no, "*** Hmmm key %s is not in the data" % k
            if k in human_yes or k in human_no:
                NYes.append( human_yes[k])
                NNo.append( human_no[k])
            else:
                print "*** Warning, %s not in human" % k
                NYes.append( human_yes[k])
                NNo.append( human_no[k])


# Now load up every response from each hypothesis
L        = [] # each hypothesis's cumulative likelihood to each data point
output   = []

NHyp = 0
for h in break_ctrlc(hypotheses):
    NHyp += 1 # keep track in case we break
    print "# Processing ", NHyp, h

    for c in concepts:
        data = concept2data[c]

        predll = h.predictive_likelihood(data)

        ll = [h.compute_predictive_likelihood(d) for d in data]

        # add on the outputs
        for di, d in enumerate(data[:25]): # as far as we go into the trial
            for ri in xrange(len(d.input)): # each individual response
                if tuple([c, di+1, ri+1]) in human_yes or k in human_no: # in case we removed any above
                    output.append( 1.0 * h(d.input, d.input[ri]) )
                    L.append(predll[di])

    assert len(L) == len(output)


print "# Created L, NYes, NNo, and Output of size %s" % len(L)
assert len(L) == len(output)
assert len(NYes)==len(NNo)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Export
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import h5py

# stack together counts
kys = nt2counts.keys()
ntlen = [len(nt2counts[k][0]) for k in kys] # store how long each
counts = numpy.concatenate([nt2counts[k] for k in kys], axis=1)

print "const int NRULES = %s;" % sum(ntlen)
print "const int NHYP = %s;" % NHyp
print "const int NDATA = %s;" % len(NYes)
print "const int NNT = %s;" % len(ntlen)
print "const int NTLEN[NNT] = {%s};" % ','.join(map(str,ntlen))

# print out the key for understanding the columns
sigs = sig2idx.keys()
for k in kys:
    s = [sig for sig in sigs if sig[0] == k]
    s = sorted(s, key=sig2idx.get)
    for si in s:
        print sig2idx[si], si

# export as hdf5
with h5py.File('data.h5', 'w') as hf:
    # first write the 'specs'
    hf.create_dataset('specs',    data=[NHyp, sum(ntlen), len(NYes), len(ntlen)], dtype=int)
    # write all the data
    hf.create_dataset('counts',    data=numpy.ravel(counts[:NHyp], order='C'), dtype=int) # must be sure if we stop early, we don't include too many counts
    hf.create_dataset('output',    data=output, dtype=float)
    hf.create_dataset('human_yes', data=numpy.ravel(NYes, order='C'), dtype=int)
    hf.create_dataset('human_no',  data=numpy.ravel(NNo, order='C'), dtype=int)
    hf.create_dataset('likelihood',  data=numpy.ravel(L, order='C'), dtype=float)
    hf.create_dataset('ntlen', data=ntlen, dtype=int)
