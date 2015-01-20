"""
A simple demo of inference with GrammarHypothesis, VectorSummary, NumberGameHypothesis, & MHSampler.

"""
import os
import time
import re
import csv
from math import exp
import pickle
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Hypotheses.GrammarHypothesisVectorized import GrammarHypothesisVectorized
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.Examples.NumberGame.JoshModel.Model import *
from LOTlib.Examples.NumberGame.NewVersion.Model import *
from Model import *


def run(grammar=simple_test_grammar, josh='', data=toy_3n,
        domain=100, alpha=0.99, ngh='enum6', ngh_file='',
        grammar_n=10000, skip=10, cap=100,
        print_stuff='grammar_h', plot_type='', plot_widget=False,
        gh_pickle='', gh_file='', csv_save=''):
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
    ngh(str):
        Do we enum/sample/load/save NumberGameHypotheses?  ['enum6' | 'mcmc1000' | 'load' | 'save']
    ngh_file(str):
        Where is the file we save/load our ngh's to/from?
    grammar_n(int):
        Number of GrammarHypotheses to sample.
    skip(int):
        Collect 1 gh sample every `skip` samples.
    cap(int):
        VectorSummary will collect this many GrammarHypothesis samples.
    print_stuff(str/list):
        What do we print? ['all' | 'ngh' | 'rules | 'grammar_h' | list(str)]
    plot_type(str):
        Indicate which type of plot to draw: ['violin' | 'line' | 'MLE' | 'MAP' | ''].
    gh_pickle(str):
        This tells us if we load of save the summary: ['load' | 'save' | ''].
    gh_file(str):
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
    # Sample/enumerate/load/save some NumberGameHypotheses

    # Load
    if 'load' in ngh:
        f = open(ngh_file, "rb")
        hypotheses = pickle.load(f)
    # MCMC
    elif 'mcmc' in ngh:
        h0 = DomainHypothesis(grammar=grammar, domain=domain, alpha=alpha)
        mh_sampler = MHSampler(h0, data[0].input, int(re.sub('[a-z]', '', ngh)))
        hypotheses = set([h for h in lot_iter(mh_sampler)])
        hypotheses = sorted(hypotheses, key=(lambda h: -h.posterior_score))
        if len(hypotheses) > 10000:
            hypotheses = hypotheses[0:10000]        # Only keep the top 10,000 ngame hypotheses
    # Enumerate
    elif 'enum' in ngh:
        hypotheses = []
        for fn in grammar.enumerate(d=int(re.sub('[a-z]', '', ngh))):
            h = DomainHypothesis(grammar=grammar, domain=domain, alpha=alpha)
            h.set_value(fn)
            hypotheses.append(h)
    else:
        hypotheses = []
    # Save
    if 'save' in ngh:
        f = open(ngh_file, "wb")
        pickle.dump(hypotheses, f)

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
    # Fill VectorSummary

    # Load from pickle file
    if 'load' in gh_pickle:
        f = open(gh_file, "rb")
        mh_grammar_summary = pickle.load(f)
    else:
        if csv_save:
            f = open(csv_save+'_values.csv', 'wb')
            writer = csv.writer(f)
            writer.writerow(['i', 'nt', 'name', 'to', 'p'])
            f.close()
            f = open(csv_save+'_bayes.csv', 'wb')
            writer = csv.writer(f)
            writer.writerow(['Prior', 'Likelihood', 'Posterior Score'])
            f.close()
        if 'samples' in print_stuff:
            print '^*'*60, '\nGenerating GrammarHypothesis Samples\n', '^*'*60

        grammar_h0 = ParameterHypothesis(grammar, hypotheses, propose_step=.1, propose_n=1)
        mh_grammar_sampler = MHSampler(grammar_h0, data, grammar_n, trace=False)
        mh_grammar_summary = VectorSummary(skip=skip, cap=cap)

        for i, h in enumerate(mh_grammar_summary(mh_grammar_sampler)):

            # Save to csv every N/1000 samples
            if csv_save:
                if i % (mh_grammar_sampler.steps/1000) is 0:
                    with open(csv_save+'_values.csv', 'rb') as r:
                        reader = csv.reader(r)
                        old_rows = [row for row in reader]
                    with open(csv_save+'_values.csv', 'wb') as w:
                        writer = csv.writer(w)
                        writer.writerows(old_rows)
                        writer.writerows([[i, r.nt, r.name, str(r.to), r.p] for r in h.rules])
                    with open(csv_save+'_bayes.csv', 'rb') as r:
                        reader = csv.reader(r)
                        old_rows = [row for row in reader]
                    with open(csv_save+'_bayes.csv', 'wb') as w:
                        writer = csv.writer(w)
                        writer.writerows(old_rows)
                        if mh_grammar_summary.sample_count:
                            writer.writerow([h.prior, h.likelihood, h.posterior_score])

            # Print every N/20 samples
            if 'samples' in print_stuff:
                if i % (mh_grammar_sampler.steps/20) == 0:
                    print ['%.3f' % v for v in h.value], '\n', i, '-'*100
                    print h.prior, h.likelihood, h.posterior_score

    # Save comparison of MAP gh to human data for each input/output combo
    if csv_save:
        with open(csv_save+'_data.csv', 'wb') as w:
            writer = csv.writer(w)
            writer.writerow(['input', 'output', 'human p', 'model p'])

            top_gh = mh_grammar_summary.get_top_samples(n=1)[0]
            hypotheses = top_gh.hypotheses

            for d in data:
                posteriors = [sum(h.compute_posterior(d.input)) for h in top_gh.hypotheses]
                Z = logsumexp(posteriors)
                weights = [(post-Z) for post in posteriors]

                for o in d.output.keys():
                    # Probability for yes on output `o` is sum of posteriors for hypos that contain `o`
                    p_human = d.output[o][0] / (d.output[o][0] + d.output[o][1])
                    p_model = sum([exp(w) if o in h() else 0 for h, w in zip(hypotheses, weights)])
                    writer.writerow([d.input, o, p_human, p_model])

    # Save GrammarHypothesis
    if 'save' in gh_pickle:
        mh_grammar_summary.pickle_summary(filename=gh_file)

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

    if gh_pickle == 'save':
        mh_grammar_summary.pickle_summary(gh_file=gh_file)


# ============================================================================================================

if __name__ == "__main__":
    path = os.getcwd()

    # --------------------------------------------------------------------------------------------------------
    # Mixture model
    # --------------------------------------------------------------------------------------------------------

    # import cProfile
    # cProfile.run("""run(grammar=mix_grammar, josh='mix', data=josh_data, domain=100,
    #                     alpha=0.9, ngh=5, grammar_n=50, skip=10, cap=100,
    #                     print_stuff=[], plot_type=[], gh_pickle=False)""",
    #              gh_file=path+'/out/profile/mix_model_50.profile')

    # run(grammar=mix_grammar, josh='mix', data=josh_data, domain=100,
    #     alpha=0.9, ngh=7, grammar_n=1000, skip=10, cap=100,
    #     print_stuff='', plot_type=[], gh_pickle='save',
    #     gh_file=path+'/out/p/mix_model_1000.p',
    #     csv_save=path+'/out/csv/mix_model_1000')

    # --------------------------------------------------------------------------------------------------------
    # Individual rule probabilities model
    # --------------------------------------------------------------------------------------------------------

    # import cProfile
    # cProfile.run("""run(grammar=individual_grammar, josh='lot', data=josh_data, domain=100,
    #                     alpha=0.9, ngh=5, grammar_n=100, skip=10, cap=100,
    #                     print_stuff=[], plot_type=[], gh_pickle=False)""",
    #              gh_file='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference'
    #                       '/NumberGame/out/profile/individual_100.profile')

    run(grammar=individual_grammar, josh='lot', data=josh_data, domain=100,
        alpha=0.9, ngh='enum7', grammar_n=500000, skip=10, cap=100,
        print_stuff='samples', plot_type='', gh_pickle='save',
        gh_file=path+'/out/p/individual_500000.p',
        csv_save=path+'/out/csv/individual_500000')

    # --------------------------------------------------------------------------------------------------------
    # LOT grammar
    # --------------------------------------------------------------------------------------------------------

    # run(grammar=lot_grammar, josh='', data=josh_data, domain=100,
    #     alpha=0.9, ngh='mcmc100000', grammar_n=1000, skip=1, cap=100,
    #     print_stuff='samples', plot_type='', gh_pickle='save',
    #     gh_file=path+'/out/p/lot_100individual_1000000.p',
    #     csv_save=path+'/out/csv/lot_1000')

    # --------------------------------------------------------------------------------------------------------
    # TESTING  |  Original number game
    # --------------------------------------------------------------------------------------------------------

    # run(grammar=complex_grammar, data=toy_2n, domain=20,
    #     alpha=0.99, ngh=6, grammar_n=1000, skip=10, cap=100,
    #     print_stuff='rules', plot_type=[], gh_pickle='save',
    #     gh_file='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference/NumberGame'
    #              '/out/newest_complex_2n_1000.p')

    # import cProfile
    # cProfile.run("""run(grammar=complex_grammar, data=toy_npow2p1, domain=20,
    #                     alpha=0.9, ngh=6, grammar_n=10000, skip=10, cap=100,
    #                     print_stuff=[], plot_type=[], gh_pickle=False)""",
    #              gh_file='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference'
    #                       '/NumberGame/out/1_14/vector_complex_npow2p1_10000.profile')

    # --------------------------------------------------------------------------------------------------------
    # Testing NumberGame hypothesis space
    # --------------------------------------------------------------------------------------------------------

    # h0 = NumberGameHypothesis(grammar=lot_grammar, domain=100, alpha=0.9)
    #
    # num_samples = 100000
    # num_chains = 10
    #
    # sample_set_sizes = {}
    # sample_size_means = {}
    #
    # # Loop for conditioned on each data input
    # for d in josh_data:
    #     sample_set_sizes[d.input] = []
    #
    #     # Number of chains to run on each datum
    #     for i in range(num_chains):
    #         mh_sampler = MHSampler(h0, d.input, num_samples)
    #         hypotheses = set([h for h in lot_iter(mh_sampler)])
    #         sample_set_sizes[d.input].append(len(hypotheses))
    #
    #     sample_size_means[d.input] = sum(sample_set_sizes[d.input]) / num_chains





    # --------------------------------------------------------------------------------------------------------


#
#
# '''print distribution over power rule:  [prior, likelihood, posterior]'''
# # vals, posteriors = grammar_h0.rule_distribution(data, 'ipowf_', np.arange(0.1, 5., 0.1))
# # print_dist(vals, posdfteriors)
# #visualize_dist(vals, posteriors, 'union_')


