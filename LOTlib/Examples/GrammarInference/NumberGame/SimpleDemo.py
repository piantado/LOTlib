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
from Model import *



def mpirun(d):
    h0 = NumberGameHypothesis(grammar=lot_grammar, domain=100, alpha=0.9)
    mh_sampler = MHSampler(h0, d.input, 100000)

    hypotheses = set()
    hypo_miniset = set()
    for h in lot_iter(mh_sampler):
        hypo_miniset.add(h)
        if len(hypo_miniset) > 200:
            hypotheses = hypotheses.union(hypo_miniset)
            hypo_miniset = set()
            if len(hypotheses) > 1000:
                hypotheses = sorted(hypotheses, key=(lambda h: -h.posterior_score))
                hypotheses = set(hypotheses[0:500])

    if len(hypotheses) > 500:
        hypotheses = sorted(hypotheses, key=(lambda h: -h.posterior_score))
        hypotheses = hypotheses[0:500]
    return hypotheses


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
            results = MPI_unorderedmap(mpirun, [[d] for d in data * 2])
            for hypotheses in results:
                ngh_samples = ngh_samples.union(hypotheses)
            if len(ngh_samples) > 50000:
                ngh_samples = sorted(ngh_samples, key=(lambda h: -h.posterior_score))
                ngh_samples = ngh_samples[0:50000]

        # Only keep the top 10,000 ngame hypotheses
        ngh_samples = sorted(ngh_samples, key=(lambda h: -h.posterior_score))
        if len(ngh_samples) > 50000:
            ngh_samples = ngh_samples[0:50000]

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
    # Save comparison of MAP gh to human data for each input/output combo

    def csv_compare_model_human(filename, gh, idx):
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

    # --------------------------------------------------------------------------------------------------------
    # Fill VectorSummary

    # Load from pickle file
    if 'load' in gh_pickle:
        f = open(gh_file, "rb")
        mh_grammar_summary = pickle.load(f)
    else:
        if csv_save:
            with open(csv_save+'_values.csv', 'wb') as w:
                writer = csv.writer(w)
                writer.writerow(['i', 'nt', 'name', 'to', 'p'])
            with open(csv_save+'_bayes.csv', 'wb') as w:
                writer = csv.writer(w)
                writer.writerow(['i', 'Prior', 'Likelihood', 'Posterior Score'])
            with open(csv_save+'_data_MAP.csv', 'wb') as w:
                writer = csv.writer(w)
                writer.writerow(['i', 'input', 'output', 'human p', 'model p'])
            with open(csv_save+'_data_h0.csv', 'wb') as w:
                writer = csv.writer(w)
                writer.writerow(['i', 'input', 'output', 'human p', 'model p'])

        if 'samples' in print_stuff:
            print '^*'*60, '\nGenerating GrammarHypothesis Samples\n', '^*'*60

        if grammar_n > 0:
            grammar_h0 = ParameterHypothesis(grammar, hypotheses, propose_step=.1, propose_n=1)
            mh_grammar_sampler = MHSampler(grammar_h0, data, grammar_n, trace=False)
            mh_grammar_summary = VectorSummary(skip=skip, cap=cap)

            csv_compare_model_human(csv_save+'_data_h0.csv', grammar_h0, 0)

            for i, h in enumerate(mh_grammar_summary(mh_grammar_sampler)):

                # Save to csv every 100 samples from 0 to 100k, then every 1000
                if csv_save:
                    if (i < 100000 and i % 100 is 0) or (i % 1000 is 0):
                        with open(csv_save+'_values.csv', 'a') as w:
                            writer = csv.writer(w)
                            writer.writerows([[i, r.nt, r.name, str(r.to), r.p] for r in h.rules])
                        with open(csv_save+'_bayes.csv', 'a') as w:
                            writer = csv.writer(w)
                            if mh_grammar_summary.sample_count:
                                writer.writerow([i, h.prior, h.likelihood, h.posterior_score])

                        top_gh = sorted(mh_grammar_summary.samples, key=(lambda x: -x.posterior_score))[0]
                        csv_compare_model_human(csv_save+'_data_MAP.csv', top_gh, i)

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
    #              gh_file=path+'/out/profile/mix_model_50.profile')

    # run(grammar=mix_grammar, josh='mix', data=josh_data, domain=100, alpha=0.9,
    #     ngh='enum7', grammar_n=1000, skip=1, cap=1000,
    #     print_stuff='samples', plot_type='', gh_pickle='',
    #     # gh_file=path+'/out/p/mix_model_1000.p',
    #     csv_save=path+'/out/csv/1_22/mix_1000')

    # --------------------------------------------------------------------------------------------------------
    # Individual rule probabilities model
    # --------------------------------------------------------------------------------------------------------

    # import cProfile
    # cProfile.run("""run(grammar=individual_grammar, josh='lot', data=josh_data, domain=100,
    #                     alpha=0.9, ngh=5, grammar_n=100, skip=10, cap=100,
    #                     print_stuff=[], plot_type=[], gh_pickle=False)""",
    #              gh_file='/Users/ebigelow35/Desktop/skool/piantado/LOTlib/LOTlib/Examples/GrammarInference'
    #                       '/NumberGame/out/profile/individual_100.profile')

    # run(grammar=individual_grammar, josh='lot', data=josh_data, domain=100, alpha=0.9,
    #     ngh='enum7', grammar_n=10000, skip=10, cap=1000,
    #     print_stuff='samples', plot_type='', gh_pickle='',
    #     # gh_file=path+'/out/p/1_22/individual_5000000.p',
    #     csv_save=path+'/out/csv/1_22/individual_10000')

    # --------------------------------------------------------------------------------------------------------
    # LOT grammar
    # --------------------------------------------------------------------------------------------------------

    # run(grammar=lot_grammar, data=josh_data, domain=100, alpha=0.9, grammar_n=0, print_stuff='',
    #     ngh='save', ngh_file=path+'/ngh_mcmc100k.p')

    run(grammar=lot_grammar, mixture_model=0, data=josh_data, domain=100, alpha=0.9, print_stuff='',
        grammar_n=5000000, skip=500, cap=10000,
        ngh_file=path+'/ngh_mcmc100k.p', ngh='load',
        gh_file=path+'/out/1_27/lot_5mil_1.p', gh_pickle='save',
        csv_save=path+'/out/1_27/lot_5mil_1')


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
    # Union size of NumberGame hypothesis space accross chains
    # --------------------------------------------------------------------------------------------------------

    # num_samples = 100000
    # num_chains = 10
    #
    # sample_set_sizes = {}
    # sample_size_means = {}
    # sample_set_union = {}
    #
    # # Loop for conditioned on each data input
    # for d in josh_data:
    #     sample_set_sizes[str(d.input)] = []
    #
    #     # Number of chains to run on each datum
    #     for i in range(num_chains):
    #         h0 = NumberGameHypothesis(grammar=lot_grammar, domain=100, alpha=0.9)
    #         mh_sampler = MHSampler(h0, d.input, num_samples)
    #         hypotheses = set([h for h in lot_iter(mh_sampler)])
    #
    #         sample_set_sizes[str(d.input)].append(len(hypotheses))
    #
    #         if not str(d.input) in sample_set_union:
    #             sample_set_union[str(d.input)] = hypotheses
    #         else:
    #             sample_set_union[str(d.input)] = sample_set_union[str(d.input)].union(hypotheses)
    #
    #         # Write to file each chain
    #         with open('out/hypothesis_space_lens_1_21.txt', 'a') as f:
    #             str_chain = 'chain' + str(i) + ' | ' + str(d.input) + ' ==> ' + str(len(hypotheses))
    #             str_union = '\t\t' + str(d.input) + ' ==> |Union(samples)| = ' + str(len(sample_set_union[str(d.input)]))
    #             str_mean  = '\t\t' + str(d.input) + ' ==> mean_len(samples) = ' + str(sum(sample_set_sizes[
    #                 str(d.input)]) / (i+1))
    #             f.write(str_chain + '\n' + str_union + '\n' + str_mean + '\n\n')
    #
    #     # Write final intersection/mean size for all chains for each datum
    #     sample_size_means[str(d.input)] = sum(sample_set_sizes[str(d.input)]) / num_chains
    #     str_union = str(d.input) + ' ==> |Union(samples)| = ' + str(len(sample_set_union[str(d.input)]))
    #     str_mean = str(d.input) + ' ==> mean(samples) = ' + str(sample_size_means[str(d.input)])
    #
    #     with open('out/hypothesis_space_lens_1_21.txt', 'a') as f:
    #         f.write(str_union + '\n' + str_mean + '%'*81 + '\n\n')
    #
    # with open('out/hypothesis_space_lens_1_21.txt', 'a') as f:
    #     all_union = set()
    #     for s in sample_set_union:
    #         all_union = all_union.union(sample_set_union[s])
    #     f.write('%'*81 + '\n' + '%'*81 + '\n' + 'OVERALL UNION SIZE = ' + str(len(all_union)))

    # --------------------------------------------------------------------------------------------------------


#
#
# '''print distribution over power rule:  [prior, likelihood, posterior]'''
# # vals, posteriors = grammar_h0.rule_distribution(data, 'ipowf_', np.arange(0.1, 5., 0.1))
# # print_dist(vals, posdfteriors)
# #visualize_dist(vals, posteriors, 'union_')


