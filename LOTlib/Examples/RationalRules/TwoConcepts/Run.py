"""
Rational rules over two concepts at the same time.

Another way to do this would be to use a Lexicon and write a custom likelihood method
"""

from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Examples.RationalRules.Model.Utilities import grammar
from Data import data


def run_mh():
    """Run the MH."""
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # somewhat weirdly, we'll make an upper node above "START" for the two concepts
    # and require it to check if concept (an argument below) is 'A'
    grammar.add_rule('TWO_CONCEPT_START', 'if_', ['(concept==\'A\')', 'START', 'START'], 1.0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create an initial hypothesis
    # This is where we set a number of relevant variables -- whether to use RR, alpha, etc.
    # Here we give args as "concept" (used in TWO_CONCEPT_START above) and "x"
    h0 = RationalRulesLOTHypothesis(grammar=G, rrAlpha=1.0, ALPHA=0.9, start='TWO_CONCEPT_START', args=['concept', 'x'])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run the vanilla sampler. Without steps, it will run infinitely
    # this prints out posterior (posterior_score), prior, likelihood,
    for h in mh_sample(h0, data, 10000, skip=100):
        print h.posterior_score, h.prior, h.likelihood, q(h)
