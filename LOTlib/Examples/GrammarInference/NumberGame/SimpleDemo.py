"""
A simple demo of inference with GrammarHypothesis, VectorSummary, NumberGameHypothesis, & MHSampler.

"""
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Inference.PriorSample import prior_sample
from LOTlib.Examples.NumberGame.NewVersion.Model import *
from Model import *


def run(grammar=simple_test_grammar, data=toy_3n, domain=20, alpha=0.99, enum_d=5, grammar_n=10000, cap=100):
    """
    Enumerate some NumberGameHypotheses, then use these to sample some GrammarHypotheses over `data`.

    Arguments:
        grammar(LOTlib.Grammar): This is our grammar.
        data(list): List of FunctionNodes to use as input/output data.
        alpha(float): Noise parameter for NumberGameHypothesis.
        domain(int): Domain parameter for NumberGameHypothesis.
        grammar_n(int): Number of GrammarHypotheses to sample.
        cap(int): VectorSummary will collect this many GrammarHypothesis samples.

    Confirmed working:
        * run(grammar=simple_test_grammar, data=toy_3n)     [12/15]
        * run(grammar=simple_grammar_2, data=toy_3n)        [12/16]

    Note:
        These currently have to be run within ipython notebook for plotting to work.

        Just open a notebook and execute this:

        >> from SimpleDemo import *
        >> run()

    """
    # --------------------------------------------------------------------------------------------------------
    # Enumerate some NumberGameHypotheses.

    hypotheses = []
    for fn in grammar.enumerate(d=enum_d):
        h = NumberGameHypothesis(grammar=grammar, domain=domain, alpha=alpha)
        h.set_value(fn)
        hypotheses.append(h)

    print '='*100, '\nNumberGameHypotheses:'
    for h in hypotheses:
        print h, h(), h.domain, h.alpha

    # --------------------------------------------------------------------------------------------------------
    # Sample some GrammarHypotheses

    grammar_h0 = GrammarHypothesis(grammar, hypotheses, proposal_step=.1, proposal_n=1)
    print '='*100, '\nGrammarRules:'
    for r in grammar_h0.rules:
        print r

    mh_grammar_sampler = MHSampler(grammar_h0, data, grammar_n, trace=False)
    mh_grammar_summary = sample_grammar_hypotheses(mh_grammar_sampler, skip=grammar_n/cap, cap=cap)
    mh_grammar_summary.print_top_samples()
    mh_grammar_summary.graph_samples()


if __name__ == "__main__":
    run()


#
# '''sample NumberGameHypotheses'''
# h0 = make_h0(grammar=simple_test_grammar, domain=domain, alpha=alpha)
# mh_sampler = MHSampler(h0, data[0].input, n)
# hypotheses = set([h for h in lot_iter(mh_sampler)])
#
#
# '''print distribution over power rule:  [prior, likelihood, posterior]'''
# # vals, posteriors = grammar_h0.rule_distribution(data, 'ipowf_', np.arange(0.1, 5., 0.1))
# # print_dist(vals, posteriors)
# #visualize_dist(vals, posteriors, 'union_')
#
#

