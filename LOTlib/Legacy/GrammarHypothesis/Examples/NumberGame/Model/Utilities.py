
import pickle
from LOTlib import break_ctrlc
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Miscellaneous import logsumexp



def normalizing_constant(hypotheses):
    """Estimate normalizing constant Z by logsumexp(posterior scores for all hypotheses)."""
    return logsumexp([h.posterior_score for h in hypotheses])

def save_hypotheses(sampler, filename='numbergame_hypotheses.p'):
    hypotheses = set()
    for h in break_ctrlc(sampler):
        hypotheses.add(h)

    f = open(filename, "wb")
    pickle.dump(hypotheses, f)
    return hypotheses


def load_hypotheses(filename='numbergame_hypotheses.p'):
    f = open(filename, "rb")
    hypotheses = pickle.load(f)
    return hypotheses


def top_ng_hypotheses(hypotheses, n=10):
    sorted_hypotheses = sorted(hypotheses, key=lambda x: x.posterior_score)
    print '%'*120
    for h in sorted_hypotheses[-n:]:
        print str(h)
        print h.prior, h.likelihood, h.posterior_score
    print '%'*120
    # # FBS version
    # hypotheses = FiniteBestSet(generator=prior_sampler, N=N, key="posterior_score")
    # for h in hypotheses:
    #     print str(h), h.posterior_score'''
    return sorted_hypotheses[-n:]
