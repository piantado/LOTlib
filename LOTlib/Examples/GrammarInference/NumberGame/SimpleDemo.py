"""
A simple demo of inference with GrammarHypothesis, VectorSummary, NumberGameHypothesis, & MHSampler.

"""
import os
import time
import re
import csv
from math import exp
import pickle
from LOTlib.Examples.NumberGame.JoshModel.Model import *
from LOTlib.Examples.NumberGame.NewVersion.Model import *
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.MPI.MPI_map import MPI_unorderedmap
from LOTlib.Visualization.MCMCSummary.TopN import TopN
from Model import *

# ------------------------------------------------------------------------------------------------------------
# CSV-saving methods

def csv_compare_model_human(filename, gh, idx, data):
    gh.update()
    for h in gh.hypotheses:
        h.compute_prior(recompute=True)
        h.update_posterior()

    with open(filename, 'a') as f:
        writer = csv.writer(f)
        hypotheses = gh.hypotheses
        for d in data:
            posteriors = [sum(h.compute_posterior(d.input)) for h in hypotheses]
            Z = logsumexp(posteriors)
            weights = [(post-Z) for post in posteriors]

            for o in d.output.keys():
                # Probability for yes on output `o` is sum of posteriors for hypos that contain `o`
                p_human = float(d.output[o][0]) / float(d.output[o][0] + d.output[o][1])
                p_model = sum([exp(w) if o in h() else 0 for h, w in zip(hypotheses, weights)])
                writer.writerow([idx, d.input, o, p_human, p_model])

def csv_initfiles(filename):
    """
    Create new csv files for filename_values, filename_bayes, filename_data_MAP, filename_data_h0.

    """
    with open(filename+'_values.csv', 'wb') as w:
        writer = csv.writer(w)
        writer.writerow(['i', 'nt', 'name', 'to', 'p'])
    with open(filename+'_bayes.csv', 'wb') as w:
        writer = csv.writer(w)
        writer.writerow(['i', 'Prior', 'Likelihood', 'Posterior Score'])
    with open(filename+'_data_MAP.csv', 'wb') as w:
        writer = csv.writer(w)
        writer.writerow(['i', 'input', 'output', 'human p', 'model p'])
    with open(filename+'_data_h0.csv', 'wb') as w:
        writer = csv.writer(w)
        writer.writerow(['i', 'input', 'output', 'human p', 'model p'])

def csv_appendfiles(filename, gh, i, mh_grammar_summary, data):
    """
    Append Bayes data to `_bayes` file, values to `_values` file, and MAP hypothesis human
    correlation data to `_data_MAP` file.

    """
    with open(filename+'_values.csv', 'a') as w:
        writer = csv.writer(w)
        writer.writerows([[i, r.nt, r.name, str(r.to), gh.value[j]] for j,r in enumerate(gh.rules)])
    with open(filename+'_bayes.csv', 'a') as w:
        writer = csv.writer(w)
        if mh_grammar_summary.sample_count:
            writer.writerow([i, gh.prior, gh.likelihood, gh.posterior_score])

    top_gh = sorted(mh_grammar_summary.samples, key=(lambda x: -x.posterior_score))[0]
    csv_compare_model_human(filename+'_data_MAP.csv', top_gh, i, data)

# ------------------------------------------------------------------------------------------------------------
# MPI

def mpirun(d):
    """
    Lets us generate our initial NumberGameHypotheses using MPI.

    """
    h0 = NoDoubleConstNGHypothesis(grammar=lot_grammar, domain=100, alpha=0.9)
    mh_sampler = MHSampler(h0, d.input, 100000)

    hypotheses = TopN(N=1000)

    for h in lot_iter(mh_sampler):
        # print h.posterior_score, h
        hypotheses.add(h)

    return [h for h in hypotheses.get_all()]


# ============================================================================================================
# The `run` script

def run(grammar=simple_test_grammar, mixture_model=0, data=josh_data,
        domain=100, alpha=0.99, ngh='enum6', ngh_file='',
        grammar_n=10000, skip=10, cap=100,
        print_stuff='samples', plot_type='', plot_widget=False,
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

    """
    if mixture_model:
        ParameterHypothesis = MixtureGrammarHypothesis
        DomainHypothesis = JoshConceptsHypothesis
    else:
        ParameterHypothesis = NoConstGrammarHypothesis
        DomainHypothesis = JoshConceptsHypothesis

    # --------------------------------------------------------------------------------------------------------
    # Sample/enumerate/load/save some NumberGameHypotheses

    # Load
    if 'load' in ngh:
        f = open(ngh_file, "rb")
        hypotheses = pickle.load(f)
        for h in hypotheses:
            h.grammar = grammar
            h.alpha = alpha
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

    # --------------------------------------------------------------------------------------------------------
    # Save NGH's (MCMC with parallelization!)

    if 'save' in ngh:

        ngh_samples = set()
        for i in range(0,10):
            results = MPI_unorderedmap(mpirun, [[d] for d in data * 1])
            for hypotheses in results:
                ngh_samples = ngh_samples.union(hypotheses)

        f = open(ngh_file, "wb")
        pickle.dump(ngh_samples, f)

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
            csv_initfiles(csv_save)

        if 'samples' in print_stuff:
            print '^*'*60, '\nGenerating GrammarHypothesis Samples\n', '^*'*60

        if grammar_n > 0:
            grammar_h0 = ParameterHypothesis(grammar, hypotheses, propose_step=.1, propose_n=1)
            mh_grammar_sampler = MHSampler(grammar_h0, data, grammar_n, trace=False)
            mh_grammar_summary = VectorSummary(skip=skip, cap=cap)

            csv_compare_model_human(csv_save+'_data_h0.csv', grammar_h0, 0, data)

            for i, gh in enumerate(mh_grammar_summary(mh_grammar_sampler)):

                # Save to csv every 200 samples from 0 to 10k, then every 1000
                if csv_save:
                    if (i < 10000 and i % 200 is 0) or (i % 5000 is 0):
                        csv_appendfiles(csv_save, gh, i, mh_grammar_summary, data)

                # Print every N/20 samples
                if 'samples' in print_stuff:
                    if i % (mh_grammar_sampler.steps/20) is 0:
                        print ['%.3f' % v for v in h.value], '\n', i, '-'*100
                        print h.prior, h.likelihood, h.posterior_score

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
        mh_grammar_summary.pickle_summary(filename=gh_file)


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
    #              filename=path+'/out/profile/mix_model_50.profile')

    # run(grammar=mix_grammar, mixture_model=1, data=josh_data, domain=100, alpha=0.9,
    #     ngh='enum7', grammar_n=1000, skip=10, cap=100,
    #     print_stuff='', plot_type='', gh_pickle='save', gh_file=path+'/out/1_29/mix_10k.p',
    #     csv_save=path+'/out/1_29/mix_10k')

    # --------------------------------------------------------------------------------------------------------
    # Individual rule probabilities model
    # --------------------------------------------------------------------------------------------------------

    run(grammar=individual_grammar, mixture_model=0, data=josh_data, domain=100, alpha=0.9,
        ngh='enum7', grammar_n=1000000, skip=200, cap=5000,
        print_stuff='', plot_type='', gh_pickle='save',
        gh_file=path+'/out/2_2/indep_1mil_1.p',
        csv_save=path+'/out/2_2/indep_1mil_1')

    # --------------------------------------------------------------------------------------------------------
    # LOT grammar
    # --------------------------------------------------------------------------------------------------------

    # run(grammar=lot_grammar, mixture_model=0, data=josh_data, domain=100, alpha=0.9,
    #     ngh='enum7', grammar_n=1000000, skip=200, cap=5000,
    #     print_stuff='', plot_type='', gh_pickle='save',
    #     gh_file=path+'/out/2_2/lot_1mil_1.p',
    #     csv_save=path+'/out/2_2/lot_1mil_1')

    # --------------------------------------------------------------------------------------------------------
    # TESTING  |  Original number game
    # --------------------------------------------------------------------------------------------------------

    # run(grammar=complex_grammar, data=toy_2n, domain=20,
    #     alpha=0.99, ngh=6, grammar_n=1000, skip=10, cap=100,
    #     print_stuff='rules', plot_type=[], gh_pickle='save',
    #     gh_file='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference/NumberGame'
    #              '/out/newest_complex_2n_1000.p')

    # --------------------------------------------------------------------------------------------------------


#
#
# '''print distribution over power rule:  [prior, likelihood, posterior]'''
# # vals, posteriors = grammar_h0.rule_distribution(data, 'ipowf_', np.arange(0.1, 5., 0.1))
# # print_dist(vals, posdfteriors)
# #visualize_dist(vals, posteriors, 'union_')


