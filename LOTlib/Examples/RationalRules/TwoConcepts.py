"""
Rational rules over two concepts at the same time.

Here, we define an if at the top that switches between two concepts.

"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData, Obj

# Here data comes from different concepts, A and B, and the input is [concept, object]

def make_data(size=10, alpha=0.99):
    # Replicate all data N times
    return [FunctionData(['A', Obj(shape='square', color='red')], True, alpha=alpha),
            FunctionData(['A', Obj(shape='square', color='blue')], False, alpha=alpha),
            FunctionData(['A', Obj(shape='triangle', color='blue')], False, alpha=alpha),
            FunctionData(['A', Obj(shape='triangle', color='red')], False, alpha=alpha),

            FunctionData(['B', Obj(shape='square', color='red')], False, alpha=alpha),
            FunctionData(['B', Obj(shape='square', color='blue')], True, alpha=alpha),
            FunctionData(['B', Obj(shape='triangle', color='blue')], True, alpha=alpha),
            FunctionData(['B', Obj(shape='triangle', color='red')], True, alpha=alpha)] * size

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Model import grammar

# somewhat weirdly, we'll make an upper node above "START" for the two concepts
# and require it to check if concept (an argument below) is 'A'
grammar.add_rule('TWO_CONCEPT_START', 'if_', ['(concept==\'A\')', 'START', 'START'], 1.0)

grammar.start = 'TWO_CONCEPT_START'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Model import make_hypothesis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    from LOTlib import break_ctrlc
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib.Miscellaneous import q


    # Create an initial hypothesis
    # This is where we set a number of relevant variables -- whether to use RR, alpha, etc.
    # Here we give args as "concept" (used in TWO_CONCEPT_START above) and "x"
    h0 = make_hypothesis(grammar=grammar, args=['concept', 'x'])

    data = make_data()

    # Run the vanilla sampler. Without steps, it will run infinitely
    # this prints out posterior (posterior_score), prior, likelihood,
    for h in break_ctrlc(MHSampler(h0, data, 10000, skip=100)):
        print h.posterior_score, h.prior, h.likelihood, q(h)
