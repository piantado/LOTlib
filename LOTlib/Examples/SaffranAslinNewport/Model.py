"""

A simple example to show use of flip(). Here, we observe strings and have to come up with a stochastic generating
function (one that happens to have memorized "words").

Run with python Demo.py --model=SaffranAslinNewport --alsoprint='lambda h: str([(k,v) for k,v in sorted(h().items(), reverse=True, key=operator.itemgetter(1))])'

"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

terminals = ['ba', 'ro', 'gi',
             'pa', 'to', 'la',
             'bi', 'gu', 'so']

from LOTlib.Grammar import Grammar

grammar = Grammar(start='STRING')

grammar.add_rule('STRING', '(%s if %s else %s)', ['STRING', 'BOOL', 'STRING'], 1.)

grammar.add_rule('STRING', 'strcons_(%s, %s)', ['STRING', 'STRING'], 1.)
grammar.add_rule('STRING', 'recurse_(C)', None, 1.0)

grammar.add_rule('STRING', '', ['ATOM'], 5.0)

grammar.add_rule('BOOL', 'C.flip(p=%s)', ['PROB'], 1.) # flip within a context

for i in xrange(5,10):
    grammar.add_rule('PROB', '0.%s' % i, None, 1.)

# add all the terminals
for t in terminals:
    grammar.add_rule('ATOM', "\'%s\'"%t, None, 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis, RecursionDepthException
from LOTlib.Hypotheses.Likelihoods.MultinomialLikelihood import MultinomialLikelihoodLogLongestSubstring
from LOTlib.Flip import compute_outcomes, TooManyContextsException
from LOTlib.Primitives.Strings import StringLengthException

class SANHypothesis(MultinomialLikelihoodLogLongestSubstring, RecursiveLOTHypothesis):
    """
    This inherits MultinomialLikelihoodLogPrefixDistance which computes the likelihood assuming that __call__ returns
    a dictinoary from strings to probabilities. The particular likelihood is one that permits you to get strings
    approximately right as long as you get their prefix right, but MultinomialLikelihood has a few other options.
    """

    def __init__(self, *args, **kwargs):
        # Since we use flip, this must be an argument of a 'context' C that will manage the randomness. Note that
        # C is the last argument, so Cfirst=False below
        RecursiveLOTHypothesis.__init__(self, grammar=grammar, display="lambda recurse_, C: %s", *args, **kwargs)


    def __call__(self):
        try:
            # We must use compute_outcomes to enumerate a bunch of possible outcomes and their probabilities, eventually
            # returning a dictinoary from output strings to their log probs. Here, the function we call is RecursiveLOTHypothesis
            # with self as a default argment. We ignore RecursionDepthException and StringLEnghtExcpetions. We must give
            # Cfirst=False because the first arg is *required* to be recurse_ for RecursiveLOTHypothesis
            return compute_outcomes(RecursiveLOTHypothesis.__call__, self, catchandpass=(RecursionDepthException, StringLengthException), Cfirst=False)
        except TooManyContextsException:
            # if we happen to get too many contexts on the stack, return nothing.
            return {'': 0.0}

def make_hypothesis(*args, **kwargs):
    return SANHypothesis(*args, **kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData
from collections import Counter
from LOTlib.Miscellaneous import sample_one

words = ['barogi', 'patola', 'biguso']

def make_data(N=30):
    """
    The data here consist of saffran-aslin-newport type strings. They have a geometric length distribution more like what
    you might find in natural data, with more frequent shorter strings. This is modeled in the hypothesis with a flip to
    whether or not you recurse to generate a longer string.
    """

    data = []
    cnt = Counter()
    for _ in xrange(N):
        cnt[''.join(sample_one(words) for _ in xrange(5))] += 1

    return [FunctionData(input=[], output=cnt)]

if __name__ == "__main__":
    print make_data()

    for _ in xrange(100):
        h = SANHypothesis()
        print h, h()

