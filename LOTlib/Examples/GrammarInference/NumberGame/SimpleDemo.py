"""
A simple demo of inference with GrammarHypothesis, VectorSummary, NumberGameHypothesis, & MHSampler.

"""
import os
import time
import csv
import pickle
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Hypotheses.GrammarHypothesisVectorized import GrammarHypothesisVectorized
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Examples.NumberGame.JoshModel.Model import *
from LOTlib.Examples.NumberGame.NewVersion.Model import *
from Model import *


def run(grammar=simple_test_grammar, josh='', data=toy_3n, domain=20,
        alpha=0.99, enum_d=5, grammar_n=10000, skip=10, cap=100,
        print_stuff='grammar_h', plot_type='', plot_widget=False,
        pickle_data='', filename='', csv_save=''):
    """
    Enumerate some NumberGameHypotheses, then use these to sample some GrammarHypotheses over `data`.

    Arguments
    ---------
    grammar(LOTlib.Grammar):
        This is our grammar.
    josh(str):
        Are we using stuff from `LOTlib.Examples.NumberGame.JoshModel`? ['mix' | 'indep' | 'lot' | None]
    data(list):
        List of FunctionNodes to use as input/output data.
    domain(int):
        Domain parameter for NumberGameHypothesis.
    alpha(float):
        Noise parameter for NumberGameHypothesis.
    enum_d(int):
        How deep do we go when enumerating ngh's?
    grammar_n(int):
        Number of GrammarHypotheses to sample.
    cap(int):
        VectorSummary will collect this many GrammarHypothesis samples.
    print_stuff(str/list):
        What do we print? ['all' | 'ngh' | 'rules | 'grammar_h' | list(str)]
    plot_type(str):
        Indicate which type of plot to draw: ['violin' | 'line' | 'MLE' | 'MAP' | ''].
    pickle_data(str):
        This tells us if we load of save the summary: ['load' | 'save' | ''].
    filename(str):
        If we're pickling, this is the file name to load/save.

    Confirmed working
    -----------------
    * run(grammar=simple_test_grammar, data=toy_3n)     [12/15]
    * run(grammar=simple_grammar_2, data=toy_3n)        [12/16]

    Note
    ----
    These currently have to be run within ipython notebook for plotting to work.
    Just open a notebook and execute the following::
        >> from SimpleDemo import *
        >> run()

    """
    if josh is 'mix':
        ParameterHypothesis = MixtureGrammarHypothesis
        DomainHypothesis = JoshConceptsHypothesis
    elif josh is 'lot':
        ParameterHypothesis = GrammarHypothesisVectorized
        DomainHypothesis = JoshConceptsHypothesis
    else:
        ParameterHypothesis = GrammarHypothesisVectorized
        DomainHypothesis = NumberGameHypothesis

    # --------------------------------------------------------------------------------------------------------
    # Sample/enumerate some NumberGameHypotheses.

    if enum_d is 'mcmc':
        h0 = DomainHypothesis(grammar=simple_test_grammar, domain=domain, alpha=alpha)
        mh_sampler = MHSampler(h0, data[0].input, 50000)
        hypotheses = set([h for h in lot_iter(mh_sampler)])
        hypotheses = sorted(hypotheses, key=(lambda h: -h.posterior_score))
        hypotheses = hypotheses[0:10000]        # Only keep the top 10,000 ngame hypotheses
    else:
        hypotheses = []
        for fn in grammar.enumerate(d=enum_d):
            h = DomainHypothesis(grammar=grammar, domain=domain, alpha=alpha)
            h.set_value(fn)
            hypotheses.append(h)

    # --------------------------------------------------------------------------------------------------------
    # Print all NumberGameHypotheses that were generated

    if print_stuff is 'all' or 'ngh' in print_stuff:
        print '='*100, '\nNumberGameHypotheses:'
        for h in hypotheses:
            print h, h(), h.domain, h.alpha
        print 'Number of NumberGameHypotheses: ', len(hypotheses)

    # --------------------------------------------------------------------------------------------------------
    # Print all GrammarRules in our Grammar, with corresponding value index

    if print_stuff is 'all' or 'rules' in print_stuff:
        print '='*100, '\nGrammarRules:'
        rules = [r for sublist in grammar.rules.values() for r in sublist]
        for i, r in enumerate(rules):
            print i, '\t|  ', r

    # --------------------------------------------------------------------------------------------------------
    # Sample some GrammarHypotheses / load MCMCSummary from pickle

    if pickle_data == 'load':
        f = open(filename, "rb")
        mh_grammar_summary = pickle.load(f)
    else:
        grammar_h0 = ParameterHypothesis(grammar, hypotheses, propose_step=.1, propose_n=1)
        mh_grammar_sampler = MHSampler(grammar_h0, data, grammar_n, trace=False)
        mh_grammar_summary = VectorSummary(skip=skip, cap=cap)

        if csv_save:
            f = open(csv_save, 'wb')
            f.close()
        if 'samples' in print_stuff:
            print '^*'*60, '\nGenerating GrammarHypothesis Samples\n', '^*'*60

        for i, h in enumerate(mh_grammar_summary(mh_grammar_sampler)):
            # Save every N/1000 samples
            if csv_save:
                if i % (mh_grammar_sampler.steps/300) is 0:
                    with open(csv_save, 'rb') as r:
                        reader = csv.reader(r)
                        old_rows = [row for row in reader]
                    with open(csv_save, 'wb') as w:
                        writer = csv.writer(w)
                        writer.writerows(old_rows)
                        writer.writerows([[i, r.nt, r.name, str(r.to), r.p] for r in h.rules])
            # Print every N/20 samples
            if 'samples' in print_stuff:
                if i % (mh_grammar_sampler.steps/20) == 0:
                    print ['%.3f' % v for v in h.value], '\n', i, '-'*100
                    print h.prior, h.likelihood, h.posterior_score

    # --------------------------------------------------------------------------------------------------------
    # Plot stuff

    if plot_type:
        mh_grammar_summary.plot(plot_type)
        if plot_widget:
            return mh_grammar_summary

    if print_stuff is 'all' or 'grammar_h' in print_stuff:
        mh_grammar_summary.print_top_samples()

    # --------------------------------------------------------------------------------------------------------
    # Save pickled MCMCSummary

    if pickle_data == 'save':
        mh_grammar_summary.pickle_summary(filename=filename)


# ============================================================================================================

if __name__ == "__main__":
    path = os.getcwd()

    # --------------------------------------------------------------------------------------------------------
    # Mixture model
    # --------------------------------------------------------------------------------------------------------

    # import cProfile
    # cProfile.run("""run(grammar=mix_grammar, josh='mix', data=josh_data, domain=100,
    #                     alpha=0.9, enum_d=5, grammar_n=50, skip=10, cap=100,
    #                     print_stuff=[], plot_type=[], pickle_data=False)""",
    #              filename=path+'/out/profile/mix_model_50.profile')

    # run(grammar=mix_grammar, josh='mix', data=josh_data, domain=100,
    #     alpha=0.9, enum_d=7, grammar_n=1000, skip=10, cap=100,
    #     print_stuff='', plot_type=[], pickle_data='save',
    #     filename=path+'/out/p/mix_model_1000.p',
    #     csv_save=path+'/out/csv/mix_model_1000.csv')

    # --------------------------------------------------------------------------------------------------------
    # Individual rule probabilities model
    # --------------------------------------------------------------------------------------------------------

    # import cProfile
    # cProfile.run("""run(grammar=individual_grammar, josh='lot', data=josh_data, domain=100,
    #                     alpha=0.9, enum_d=5, grammar_n=100, skip=10, cap=100,
    #                     print_stuff=[], plot_type=[], pickle_data=False)""",
    #              filename='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference'
    #                       '/NumberGame/out/profile/individual_100.profile')

    # run(grammar=individual_grammar, josh='lot', data=josh_data, domain=100,
    #     alpha=0.9, enum_d=7, grammar_n=300, skip=3, cap=100,
    #     print_stuff='', plot_type='', pickle_data='save',
    #     filename=path+'/out/p/individual_1000.p',
    #     csv_save=path+'/out/csv/individual_1000.csv')

    # --------------------------------------------------------------------------------------------------------
    # LOT grammar
    # --------------------------------------------------------------------------------------------------------

    run(grammar=lot_grammar, josh='', data=josh_data, domain=100,
        alpha=0.9, enum_d='mcmc', grammar_n=1000, skip=1000, cap=100,
        print_stuff='samples', plot_type='', pickle_data='save',
        filename=path+'/out/p/lot_1000.p',
        csv_save=path+'/out/csv/lot_1000.csv')

    # --------------------------------------------------------------------------------------------------------
    # TESTING  |  Original number game
    # --------------------------------------------------------------------------------------------------------

    # run(grammar=complex_grammar, data=toy_2n, domain=20,
    #     alpha=0.99, enum_d=6, grammar_n=1000, skip=10, cap=100,
    #     print_stuff='rules', plot_type=[], pickle_data='save',
    #     filename='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference/NumberGame'
    #              '/out/newest_complex_2n_1000.p')

    # import cProfile
    # cProfile.run("""run(grammar=complex_grammar, data=toy_npow2p1, domain=20,
    #                     alpha=0.9, enum_d=6, grammar_n=10000, skip=10, cap=100,
    #                     print_stuff=[], plot_type=[], pickle_data=False)""",
    #              filename='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference'
    #                       '/NumberGame/out/1_14/vector_complex_npow2p1_10000.profile')

    # --------------------------------------------------------------------------------------------------------


#
#
# '''print distribution over power rule:  [prior, likelihood, posterior]'''
# # vals, posteriors = grammar_h0.rule_distribution(data, 'ipowf_', np.arange(0.1, 5., 0.1))
# # print_dist(vals, posdfteriors)
# #visualize_dist(vals, posteriors, 'union_')


