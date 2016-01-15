
WORDS = ['even', 'odd']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q

grammar = Grammar()
grammar.add_rule('START', '', ['BOOL'], 1.)

grammar.add_rule('BOOL', '(%s == %s)', ['NUMBER', 'NUMBER'], 1.)
grammar.add_rule('BOOL', '(not %s)', ['BOOL'], 1.)

grammar.add_rule('BOOL', '(%s and %s)', ['BOOL', 'BOOL'], 1.)
grammar.add_rule('BOOL', '(%s or %s)',  ['BOOL', 'BOOL'], 1.)  # use the short_circuit form

grammar.add_rule('NUMBER', 'x', None, 1.)
grammar.add_rule('NUMBER', '1', None, 1.)
grammar.add_rule('NUMBER', '0', None, 1.)
grammar.add_rule('NUMBER', 'plus_', ['NUMBER', 'NUMBER'], 1.)
grammar.add_rule('NUMBER', 'minus_', ['NUMBER', 'NUMBER'], 1.)

for w in WORDS:
    grammar.add_rule('BOOL', 'lexicon', [q(w), 'NUMBER'], 1.)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData

def make_data(n=1, alpha=0.99):
    data = []
    for x in xrange(1, 10):
        data.append( FunctionData(input=['even', x], output=(x % 2 == 0), alpha=alpha) )
        data.append( FunctionData(input=['odd',  x], output=(x % 2 == 1), alpha=alpha) )
    return data*n

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.Lexicon.RecursiveLexicon import RecursiveLexicon
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

class EvenOddLexicon(BinaryLikelihood, RecursiveLexicon):
    pass

def make_hypothesis(**kwargs):

    h = EvenOddLexicon(**kwargs)

    for w in WORDS:
        h.set_word(w, LOTHypothesis(grammar, display='lambda lexicon, x: %s'))

    return h

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    standard_sample(make_hypothesis, make_data, save_top=False)





