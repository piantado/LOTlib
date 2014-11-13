"""
Just create some combinators and reduce them.

"""
from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

from LOTlib.Miscellaneous import q
from LOTlib.Evaluation.Primitives.Functional import cons_ # for evaling

G = Grammar()

G.add_rule('START', 'cons_', ['START', 'START'], 2.0)

G.add_rule('START', 'I', None, 1.0)
G.add_rule('START', 'S', None, 1.0)
G.add_rule('START', 'K', None, 1.0)

from LOTlib.Evaluation.CombinatoryLogic import combinator_reduce
from LOTlib.Evaluation.EvaluationException import EvaluationException

for _ in range(10000):

    t = G.generate()

    lst = t.liststring()

    print lst, "\t->\t",
    try:
        print combinator_reduce(lst)
    except EvaluationException as e:
        print "*Probable-NON-HALT*"
