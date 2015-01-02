"""
A simple demo of inference with GrammarHypothesis, VectorSummary, NumberGameHypothesis, & MHSampler.

"""
import pickle
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Examples.NumberGame.JoshModel.Model import *
from LOTlib.Examples.NumberGame.NewVersion.Model import *
from Model import *


def run(grammar=simple_test_grammar, josh=False, data=toy_3n, domain=20,
        alpha=0.99, enum_d=5, grammar_n=10000, cap=100,
        print_stuff='grammar_h', plot_type=None, pickle_data=False):
    """
    Enumerate some NumberGameHypotheses, then use these to sample some GrammarHypotheses over `data`.

    Arguments:
        grammar(LOTlib.Grammar): This is our grammar.
        data(list): List of FunctionNodes to use as input/output data.
        alpha(float): Noise parameter for NumberGameHypothesis.
        domain(int): Domain parameter for NumberGameHypothesis.
        grammar_n(int): Number of GrammarHypotheses to sample.
        cap(int): VectorSummary will collect this many GrammarHypothesis samples.
        plot(str): String input indicating type of plot to graph: [violin | line | MLE | MAP].

    Confirmed working:
        * run(grammar=simple_test_grammar, data=toy_3n)     [12/15]
        * run(grammar=simple_grammar_2, data=toy_3n)        [12/16]

    Note:
        These currently have to be run within ipython notebook for plotting to work.

        Just open a notebook and execute this:

        >> from SimpleDemo import *
        >> run()

    """

    if josh is 'mix':
        ParameterHypothesis = MixtureGrammarHypothesis
        DomainHypothesis = JoshConceptsHypothesis
    elif josh is 'indep':
        ParameterHypothesis = ShallowGrammarHypothesis
        DomainHypothesis = JoshConceptsHypothesis
    elif josh is 'lot':
        ParameterHypothesis = GrammarHypothesis
        DomainHypothesis = JoshConceptsHypothesis
    else:
        ParameterHypothesis = GrammarHypothesis
        DomainHypothesis = NumberGameHypothesis

    # --------------------------------------------------------------------------------------------------------
    # Enumerate some NumberGameHypotheses.

    hypotheses = []
    for fn in grammar.enumerate(d=enum_d):
        h = DomainHypothesis(grammar=grammar, domain=domain, alpha=alpha)
        h.set_value(fn)
        hypotheses.append(h)

    # --------------------------------------------------------------------------------------------------------
    # Print all NumberGameHypotheses that were generated

    if print_stuff is True or 'ngh' in print_stuff:
        print '='*100, '\nNumberGameHypotheses:'
        for h in hypotheses:
            print h, h(), h.domain, h.alpha
    print 'Number of NumberGameHypotheses: ', len(hypotheses)

    # --------------------------------------------------------------------------------------------------------
    # Print all GrammarRules in our Grammar, with corresponding value index

    if print_stuff is True or 'rules' in print_stuff:
        print '='*100, '\nGrammarRules:'
        rules = [r for sublist in grammar.rules.values() for r in sublist]
        for i, r in enumerate(rules):
            print i, '\t|  ', r

    # --------------------------------------------------------------------------------------------------------
    # Sample some GrammarHypotheses / load MCMCSummary from pickle

    if pickle_data == 'load':
        f = open('MCMC_summary_data.p', "rb")
        mh_grammar_summary = pickle.load(f)
    else:
        grammar_h0 = ParameterHypothesis(grammar, hypotheses, propose_step=.1, propose_n=1)
        mh_grammar_sampler = MHSampler(grammar_h0, data, grammar_n, trace=False)
        mh_grammar_summary = sample_grammar_hypotheses(mh_grammar_sampler, skip=grammar_n/cap, cap=cap)

    # --------------------------------------------------------------------------------------------------------
    # Plot stuff

    if plot_type is not None:
        mh_grammar_summary.plot(plot_type)
        # return mh_grammar_summary

    if print_stuff is True or 'grammar_h' in print_stuff:
        mh_grammar_summary.print_top_samples()

    # --------------------------------------------------------------------------------------------------------
    # Save pickled MCMCSummary

    if pickle_data == 'save':
        mh_grammar_summary.pickle_summary()




if __name__ == "__main__":
    run(grammar=complex_grammar, data=toy_2pownp1,
        domain=20, alpha=0.99, enum_d=6, grammar_n=10000, cap=1000,
        plot_type=None, pickle_data='save')


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

