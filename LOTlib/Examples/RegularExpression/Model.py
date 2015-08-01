
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData

def make_data(size=1, alpha=0.99):
    return [FunctionData(input=['aaaa'], output=True, alpha=alpha),
            FunctionData(input=['aaab'], output=False, alpha=alpha),
            FunctionData(input=['aabb'], output=False, alpha=alpha),
            FunctionData(input=['aaba'], output=False, alpha=alpha),
            FunctionData(input=['aca'], output=True, alpha=alpha),
            FunctionData(input=['aaca'], output=True, alpha=alpha),
            FunctionData(input=['a'], output=True, alpha=alpha)] * size

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar

grammar = Grammar()

grammar.add_rule('START', '', ['EXPR'], 1.0)

grammar.add_rule('EXPR', '(%s*)', ['EXPR'], 1.0)
grammar.add_rule('EXPR', '(%s?)', ['EXPR'], 1.0)
grammar.add_rule('EXPR', '(%s+)', ['EXPR'], 1.0)
grammar.add_rule('EXPR', '(%s|%s)', ['EXPR', 'EXPR'], 1.0)
grammar.add_rule('EXPR', '%s%s', ['TERMINAL', 'EXPR'], 5.0)
grammar.add_rule('EXPR', '%s', ['TERMINAL'], 5.0)

for v in 'abc.':
    grammar.add_rule('TERMINAL', v, None, 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.FunctionNode import isFunctionNode
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
from LOTlib.Evaluation.EvaluationException import EvaluationException
import re

class RegexHypothesis(BinaryLikelihood, LOTHypothesis):
    """Define a special hypothesis for regular expressions.

    This requires overwriting compile_function to use our custom interpretation model on trees -- not just
    simple eval.
    """

    def __init__(self, **kwargs):
        LOTHypothesis.__init__(self, grammar, **kwargs)

    def compile_function(self):
        c = re.compile(str(self.value))
        return (lambda s: (c.match(s) is not None))

    def __str__(self):
        return str(self.value)

    def __call__(self, *args):
        try:
            return LOTHypothesis.__call__(self, *args)
        except EvaluationException:
            return None

# def to_regex(fn):
#     """Map a tree to a regular expression.
#
#     Custom mapping from a function node to a regular expression string (like, e.g. "(ab)*(c|d)" )
#     """
#     assert isFunctionNode(fn)
#
#     if fn.name == 'star_':         return '(%s)*'% to_regex(fn.args[0])
#     elif fn.name == 'plus_':       return '(%s)+'% to_regex(fn.args[0])
#     elif fn.name == 'question_':   return '(%s)?'% to_regex(fn.args[0])
#     elif fn.name == 'or_':         return '(%s|%s)'% tuple(map(to_regex, fn.args))
#     elif fn.name == 'str_append_': return '%s%s'% (fn.args[0], to_regex(fn.args[1]))
#     elif fn.name == 'terminal_':   return '%s'%fn.args[0]
#     elif fn.name == '':            return to_regex(fn.args[0])
#     else:
#         assert False, fn


def make_hypothesis(**kwargs):
    """Define a new kind of LOTHypothesis, that gives regex strings.

    These have a special interpretation function that compiles differently than straight python eval.
    """
    return RegexHypothesis(**kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib import break_ctrlc
    from LOTlib.Miscellaneous import qq

    h0 = make_hypothesis()
    data = make_data()

    for h in break_ctrlc(MHSampler(h0, data, steps=10000)):
        print h.posterior_score, h.prior, h.likelihood, qq(h)