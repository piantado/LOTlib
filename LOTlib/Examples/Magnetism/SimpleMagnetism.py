"""
        A very simple case of predicate invention, inspired by

        T. D. Ullman, N. D. Goodman and J. B. Tenenbaum (2012), Theory learning as stochastic search in the language of thought. Cognitive Development.

        Here, we invent simple predicates whose value is determined by a set membership (BASE-SET), and express logical
        concepts over those predicates. Data is set up to be like magnetism, with positives (pi) and negatives (ni) that interact
        with each other but not within groups.

        This is simple because there's only two types of things, and you observe all interactions. See ComplexMagnetism.py for a more complex case...
"""

from LOTlib.Miscellaneous import qq
from LOTlib.Grammar import Grammar
from LOTlib.DataAndObjects import FunctionData
from LOTlib.FunctionNode import cleanFunctionNodeString

grammar = Grammar()

grammar.add_rule('START', '', ['Pabstraction'], 1.0) # a predicate abstraction

# lambdaUsePredicate is where you can use the predicate defined in lambdaDefinePredicate
grammar.add_rule('Pabstraction',  'apply_', ['lambdaUsePredicate', 'lambdaDefinePredicate'], 1.0, )
grammar.add_rule('lambdaUsePredicate', 'lambda', ['INNER-BOOL'],    5.0, bv_type='INNER-BOOL', bv_args=['OBJECT'], bv_prefix='F')
grammar.add_rule('lambdaUsePredicate', 'lambda', ['Pabstraction'], 1.0,  bv_type='INNER-BOOL', bv_args=['OBJECT'], bv_prefix='F')


# Define a predicate that will just check if something is in a BASE-SET
grammar.add_rule('lambdaDefinePredicate', 'lambda', ['lambdaDefinePredicateINNER'], 1.0,  bv_type='OBJECT', bv_args=None, bv_prefix='z')
# the function on objects, that allows them to be put into classes (analogous to a logical model here)
grammar.add_rule('lambdaDefinePredicateINNER', 'is_in_', ['OBJECT', 'BASE-SET'], 1.0)

# Here's how things will look with applylambda:
#grammar.add_rule('START', 'applylambda', ['INNER-BOOL'], 10.0,  bv_type='INNER-BOOL', bv_args=['OBJECT'], bv_prefix='F')
#grammar.add_rule('START', 'applylambda', ['START'],      1.0,   bv_type='INNER-BOOL', bv_args=['OBJECT'], bv_prefix='F')


grammar.add_rule('INNER-BOOL', 'is_in_', ['OBJECT', 'BASE-SET'], 10.0)


# After we've defined F, these are used to construct the concept
grammar.add_rule('INNER-BOOL', 'and_', ['INNER-BOOL', 'INNER-BOOL'], 1.0)
grammar.add_rule('INNER-BOOL', 'or_', ['INNER-BOOL', 'INNER-BOOL'], 1.0)
grammar.add_rule('INNER-BOOL', 'not_', ['INNER-BOOL'], 1.0)

grammar.add_rule('OBJECT', 'x', None, 1.0)
grammar.add_rule('OBJECT', 'y', None, 1.0)
grammar.add_rule('OBJECT', '', ['BASE-OBJECT'], 1.0) # maybe or maybe not?

# BASE-SET is here a set of BASE-OBJECTS (non-args)
grammar.add_rule('BASE-SET', 'set_add_', ['BASE-OBJECT', 'BASE-SET'], 1.0)
grammar.add_rule('BASE-SET', 'set_', [], 1.0)

grammar.add_rule('BASE-OBJECT', qq('p1'), None, 1.0)
grammar.add_rule('BASE-OBJECT', qq('p2'), None, 1.0)
grammar.add_rule('BASE-OBJECT', qq('n1'), None, 1.0)
grammar.add_rule('BASE-OBJECT', qq('n2'), None, 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up data -- true output means attraction (p=positive; n=negative)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data = [ FunctionData(input=[ "p1", "n1" ], output=True),
                 FunctionData(input=[ "p1", "n2" ], output=True),
                 FunctionData(input=[ "p1", "p1" ], output=False),
                 FunctionData(input=[ "p1", "p2" ], output=False),

                 FunctionData(input=[ "p2", "n1" ], output=True),
                 FunctionData(input=[ "p2", "n2" ], output=True),
                 FunctionData(input=[ "p2", "p1" ], output=False),
                 FunctionData(input=[ "p2", "p2" ], output=False),

                 FunctionData(input=[ "n1", "n1" ], output=False),
                 FunctionData(input=[ "n1", "n2" ], output=False),
                 FunctionData(input=[ "n1", "p1" ], output=True),
                 FunctionData(input=[ "n1", "p2" ], output=True),

                 FunctionData(input=[ "n2", "n1" ], output=False),
                 FunctionData(input=[ "n2", "n2" ], output=False),
                 FunctionData(input=[ "n2", "p1" ], output=True),
                 FunctionData(input=[ "n2", "p2" ], output=True)]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Standard exports
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

def make_h0(**kwargs):
    return LOTHypothesis(grammar, args=['x', 'y'], ALPHA=0.999, **kwargs) # alpha here trades off with the amount of data. Currently assuming no noise, but that's not necessary


if __name__ == "__main__":

    from LOTlib import lot_iter

    from LOTlib.Inference.Proposals.RegenerationProposal import *
    #mp = MixtureProposal([RegenerationProposal(grammar), InsertDeleteProposal(grammar)] )
    mp = RegenerationProposal(grammar)

    h0 = LOTHypothesis(grammar, args=['x', 'y'], ALPHA=0.999, proposal_function=mp) # alpha here trades off with the amount of data. Currently assuming no noise, but that's not necessary


    from LOTlib.Inference.MetropolisHastings import mh_sample
    for h in lot_iter(mh_sample(h0, data, 4000000, skip=100)):
        print h.posterior_score, h.likelihood, h.prior,  cleanFunctionNodeString(h)
        print map( lambda d: h(*d.input), data)
