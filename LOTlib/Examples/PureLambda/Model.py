# -*- coding: utf-8 -*-

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now that's a simple grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar

grammar = Grammar()

grammar.add_rule('START', '', ['EXPR'], 1.0)

grammar.add_rule('EXPR', 'lambda', ['EXPR'], 2.0, bv_type='EXPR', bv_args=None, bv_p=5.0)
grammar.add_rule('EXPR', 'apply_', ['EXPR', 'EXPR'], 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a simple grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from math import log
from LOTlib.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LambdaEvaluation import lambda_reduce, lambdastring, compose_and_reduce, EvaluationException
from LOTlib.Miscellaneous import qq, Infinity, attrmem

class PureLambdaLexicon(SimpleLexicon):

    def __init__(self, alpha=0.99, **kwargs):
        self.alpha = alpha

        SimpleLexicon.__init__(self, **kwargs)

    def __str__(self):
        return ('\n'.join([u"%-15s: %s" % (qq(w), lambdastring(v.value)) for w, v in sorted(self.value.iteritems())]) + '\0').encode('utf-8')

    def __call__(self, *args):
        """
            We "call" by taking some args, mapping each to words, and currying:

            (((args[0] args[1]) args[2]) ...)

            To do this, we builld the right FunctionNode and then lambda_reduce it.
            Sometimes this returns None--when our evaluation doesn't halt (fast enough)

            NOTE: IF we change the grammar, EXPR must be the returntype!
        """
        assert len(args) > 1

        try:
            return compose_and_reduce(*[self.value[k].value for k in args])
        except EvaluationException:
            return None

    @attrmem('prior')
    def compute_prior(self):
        """
            Assign 0-probability to hypotheses with repeat values.
            BUT we only discover repeat values by reducing.
        """
        seen = set()
        for k,v in self.value.items():
            try:
                reduced = str(lambda_reduce(v.value))
            except EvaluationException:
                return -Infinity

            if reduced in seen:
                return -Infinity
            else:
                seen.add(reduced)

        return SimpleLexicon.compute_prior(self)

    def compute_single_likelihood(self, di):
        """
            Compute the likelihood, where di.input is a [f arg1 arg2 arg3 ...], curried
        """

        freduced = self(*di.input) # this is where the magic happens

        # Okay with 1.0-ALPHA we'll sample uniformly from words
        # and with probability ALPHA we'll sample from all the words which have
        # identical associated lambdas
        p = (1.0-self.alpha) / len(self.all_words())

        freduced_str = lambdastring(freduced)

        # TODO: maybe we also reduce self.value[w]?
        matches = filter(lambda w: lambdastring(self.value[w].value) == freduced_str, self.all_words())
        # print ">>>", di.input, freduced_str, matches

        if di.output in matches:
            p += self.alpha / len(matches) # else outlier likelihood

        return log(p)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a simple grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import re
from LOTlib.DataAndObjects import FunctionData

def load_words_and_data(path):
    """
    Takes a data path and return [words, data]
    """

    # Load the data
    data = []
    with open(path, 'r') as f:
        for l in f:
            if re.match(r"\s*#", l): continue # skip comments
            if not re.match(r"[^\s]", l): continue # skip whitespace

            lhs, output = re.split(r"\s*=\s*", l.strip())
            args = re.split(r"\s+", lhs)

            data.append( FunctionData(input=args, output=output) )
            #print "Loading data", args, "->", output

    # Figure out all the words! (here, tokens)
    words = set()
    for di in data:
        words.add(di.output)
        [words.add(x) for x in di.input]
    words = list(words)

    return [words, data]


def print_lexicon_and_data(L, data):
    """
    Friendlier printing of the lexicon and associated data inputs and outputs
    """

    print L.posterior_score, L.prior, L.likelihood
    print L

    for di in data:
        outstr = lambdastring(L(*di.input))
        print "\t", di.input, "->", di.output, "\t==>", [ w for w in L.value.keys() if lambdastring(L.value[w].value)==outstr], "\t", outstr

    print "\n"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up standard exports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "domains/boolean-logic.txt")

def make_hypothesis(data=DEFAULT_DATA, **kwargs):
    words, data = load_words_and_data(data)
    L0 = PureLambdaLexicon(**kwargs)
    for w in words:
        L0.set_word(w, LOTHypothesis(grammar, args=[], maxnodes=15))
    return L0

def make_data(n=1, data=DEFAULT_DATA):
    words, data = load_words_and_data(data)
    return data*n

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main running code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from optparse import OptionParser
    from LOTlib import break_ctrlc
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

    parser = OptionParser()
    parser.add_option("--in", dest="IN", type="string", help="Input data file", default=DEFAULT_DATA)
    options, _ = parser.parse_args()

    words, data = load_words_and_data(options.IN)
    L0 = PureLambdaLexicon(likelihood_temperature=1.0)
    for w in words:
        L0.set_word(w, LOTHypothesis(grammar, args=[], maxnodes=15))

    for L in break_ctrlc(MHSampler(L0, data)):
        # print_lexicon_and_data(L, data) # If you want to see all the output for each data point, use this

        print L.posterior_score, L.prior, L.likelihood
        print L, "\n"
