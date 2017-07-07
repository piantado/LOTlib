from Levenshtein import editops
from LOTlib.Miscellaneous import logplusexp, logsumexp
from math import log

def edit_likelihood(x,y, alphabet_size=2, alpha=0.99):
    """
    This computes the likelihood of going from x->y by choosing levenshtein ops uniformly at random,
    and sampling from the alphabet on insertions and replacements.
    """
    ops = editops(x,y)
    lp = log(alpha)*(len(y)-len(ops)) # all the unchanged
    for o, _, _ in ops:
        if   o == 'equal':   assert False # should never get here
        elif o == 'replace': lp += log(1.0-alpha) - log(3.0) - log(alphabet_size)
        elif o == 'insert':  lp += log(1.0-alpha) - log(3.0) - log(alphabet_size)
        elif o == 'delete':  lp += log(1.0-alpha) - log(3.0)
        else: assert False
    return lp


def swappy_likelihood(x,y, alphabet_size=2, alpha=0.99):
    """
    We assume that y comes from x by randomly swapping back and forth on whether you are copying
    elements of x or typing randomly
    """

    pass 


def log_geom_pdf(n,p):
    """ Scipy's is soooo slow"""
    return (n-1)*log(1.0-p) + log(p)

def prefix_likelihood(x,y, alphabet_size=2, alpha=0.99):
    """
    Compute likelihood under a model you draw  l ~ geometric(1-alpha) and copy the first alpha positions
    and  then generate a length l2 ~ geometric(0.5) and fill in with random characters from the alphabet
    """

    if len(x) == 0 or len(y) == 0:
        lp = log_geom_pdf(len(y),0.5) - (len(y))*log(alphabet_size)
    else:
        lp = 0.0
        for i in xrange(min(len(x),len(y))):
            if x[i]==y[i]:
                # if these are still equal, they could have come from either, so together the probability of all
                # of these
                pthis = log_geom_pdf(i,1.0-alpha) + log_geom_pdf(len(y)-i,0.5) - (len(y)-i)*log(alphabet_size)

                lp = logplusexp(lp, pthis)
            else:
                # once they are unequal, we had to have generated the rest from noise
                # lp stores the sum of all of the ways of generating up to i

                lp = logplusexp(lp, log_geom_pdf(len(y)-i,0.5) - (len(y)-i)*log(alphabet_size))
                break

    return lp


class LevenshteinPseudoLikelihood(object):

    """
        Data is a dictionary from strings to counts; use the min edit distance if not in the output of the function.
        Requires self.alphabet_size to say how many possible tokens there are
    """

    def compute_single_likelihood(self, datum):
        assert isinstance(datum.output, dict)

        hp = self(*datum.input)  # output dictionary, output->probabilities
        assert isinstance(hp, dict)

        s = 0.0
        for k, dc in datum.output.items():
            if k in hp:
                s += dc * hp[k]
            elif len(hp.keys()) > 0:
                # probability fo each string under this editing model
                s += dc * logsumexp([ v + edit_likelihood(x, k, alphabet_size=self.alphabet_size, alpha=datum.alpha) for x, v in hp.items() ]) # the highest probability string; or we could logsumexp
            else:
                s += dc * edit_likelihood('', k, alphabet_size=self.alphabet_size, alpha=datum.alpha)

            # This is the mixing {a,b}* noise model
            # lp = log(1.0-datum.alpha) - log(self.alphabet_size+1)*(len(k)+1) #the +1s here count the character marking the end of the string
            # if k in hp:
            #     lp = logplusexp(lp, log(datum.alpha) + hp[k]) # if non-noise possible
            # s += dc*lp
        return s

class MonkeyNoiseLikelihood(object):
    """
    Data is a dictionary from strings to counts.
    Assume that out of dictionary strings are generated by a random typing process
    Requires self.alphabet_size to say how many possible tokens there are
    Data here requires an alpha for noise
    """

    def compute_single_likelihood(self, datum):
        assert isinstance(datum.output, dict)

        hp = self(*datum.input)  # output dictionary, output->probabilities
        assert isinstance(hp, dict)

        s = 0.0
        for k, dc in datum.output.items():

            lp = -log(self.alphabet_size+1)*(len(k)+1) + log(1.0-datum.alpha) # probability of generating under random typing; +1 is for an EOS marker
            if k in hp:
                lp = logplusexp(lp, hp[k] + log(datum.alpha)) # if we could have been generated
            s += dc*lp

        return s


""" Some worse old ways to do this """
from StochasticLikelihood import StochasticLikelihood
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from Levenshtein import distance

class LevenshteinPseudoLikelihood(Hypothesis):
    """
    A (pseudo)likelihood function that is e^(-string edit distance)
    """

    def compute_single_likelihood(self, datum, distance_factor=1.0):
        return -distance_factor*distance(datum.output, self(*datum.input))


class StochasticLevenshteinPseudoLikelihood(StochasticLikelihood):
    """
    A levenshtein distance metric on likelihoods, where the output of a program is corrupted by
    levenshtein noise. This allows for a smoother space of hypotheses over strings.

    Since compute_likelihood passes **kwargs to compute_single_likelihood, we can pass distance_factor
    to compute_likelihood to get it here.
    """

    def compute_single_likelihood(self, datum, llcounts, distance_factor=100.0):
        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

        lo = sum(llcounts.values()) # normalizing constant

        # We are going to compute a pseudo-likelihood, counting close strings as being close
        return sum([datum.output[k]*logsumexp([log(llcounts[r])-log(lo) - distance_factor*distance(r, k) for r in llcounts.keys()]) for k in datum.output.keys()])