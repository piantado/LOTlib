"""
    Grammar inference

    H.baserate = 0.4 #
    H.compute_posterior(data) -- call for everything
    H.Z() - get normalizer

"""
import pickle

from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis

from Model.Grammar import grammar

# map each concept to a hypothesis
with open('hypotheses.pkl', 'r') as f:
    concept2hypotheses = pickle.load(f)
print "# Loaded hypotheses: ", map(len, concept2hypotheses.values())

def build_conceptlist(c,l):
    return "CONCEPT_%s__LIST_%s.txt"%(c,l)


## Build up a data structure for representing the human data
## We will map tuples of concept-list, set, response to counts.
import pandas
import math
from collections import Counter
human_data = pandas.read_csv('HumanData/TurkData-Accuracy.txt', sep='\t', low_memory=False, index_col=False)
human_yes = Counter()
human_no  = Counter()
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

from LOTlib.Miscellaneous import logsumexp, attrmem
from math import exp, log
from copy import deepcopy
class MyGrammarHypothesis(GrammarHypothesis):

    def __init__(self, value=None):
        self.concept2hypotheses = concept2hypotheses
        GrammarHypothesis.__init__(self, grammar, [], value=value)


    def __copy__(self, value=None):
        cpy = type(self)(value)
        cpy.concept2hypotheses = deepcopy(self.concept2hypotheses)
        return cpy

    def update(self):
        """Update `self.grammar` & priors for `self.hypotheses` relative to `self.value`.

        We'll need to do this whenever we calculate things like predictive, because we need to use
          the `posterior_score` of each domain hypothesis get weights for our predictions.

        """
        # Set probability for each rule corresponding to value index
        for i in range(1, self.n):
            self.rules[i].p = self.value[i]

        # Recompute prior for each hypothesis, given new grammar probs
        for cl in self.concept2hypotheses.keys():
            for h in self.concept2hypotheses[cl]:
                h.compute_prior()
                h.update_posterior()

    @attrmem('likelihood')
    def compute_likelihood(self, data):

        ll = 0.0
        for cl in self.concept2hypotheses.keys(): # for each concept and list

            if cl not in concept2data:
                print "# Warning, %s not in concept2data."%cl
                continue

            d = concept2data[cl]

            for si in xrange(len(d)): # for each prefix of the data

                hypotheses = self.concept2hypotheses[cl]
                assert len(hypotheses) > 0

                # update the posteriors for this amount of data
                for h in hypotheses:
                    h.compute_posterior(d[:si]) # up to but not including this set

                # get their normalizer
                Z = logsumexp([h.posterior_score for h in hypotheses])

                nxtd = d[si]
                pred = [0.0] * len(nxtd.input) # how we respond to each
                # compute the predictive
                for h in hypotheses:
                    p = exp(h.posterior_score-Z)
                    for i, ri in enumerate(h.evaluate_on_set(nxtd.input)):
                        pred[i] += p*((ri==True)*h.alpha + (1.0-h.alpha)*h.baserate)
                for ri, pi in enumerate(pred):
                    key = tuple([cl, si, ri])
                    # assert key in human_yes and key in human_no, "No key " + key # Does not have to be there because there can be zero counts
                    # print pi
                    ll += human_yes[key]*log(pi) + human_no[key]*log(1.-pi)
        return ll

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib import break_ctrlc
    # h = MyGrammarHypothesis()
    # h.compute_likelihood([])

    from LOTlib.MCMCSummary.VectorSummary import VectorSummary
    summary = VectorSummary(skip=1, cap=99999)
    summary.csv_initfiles("out")


    for h in break_ctrlc(MHSampler(MyGrammarHypothesis(), [], trace=True)):
        summary.add(h)
        summary.csv_appendfiles("out", None)

        print h


