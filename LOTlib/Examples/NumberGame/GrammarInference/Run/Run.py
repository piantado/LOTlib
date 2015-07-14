"""
A simple demo of inference with GrammarHypothesis, VectorSummary, NumberGameHypothesis, & MHSampler.


Command-line Args
-----------------

--pickle-summary
    If true, pickle VectorSummary
-P --pickle
    Pickle MAP & current sample  GH every n steps.
-S --save-file
    Save csv's and pickles to this file (no extension)
-f --ngh
    Where's the file with the NumberGameHypotheses?

-g --grammar
    Which grammar do we use? [mix_grammar | independent_grammar | lot_grammar]
--grammar-scale
    Do we use a gamma to semi-randomly init grammar rule probs?
-d --data
    Which data do we use? [josh_data | ??]

-i --iters
    Number of samples to run per chain
-s --skip
    Collect 1 gh sample every `skip` samples.
-c --cap
    VectorSummary will collect this many GrammarHypothesis samples.

-v --verbose
    Print everything!
-q --quiet
    Print nothing!

Example
-------
# Independent model
$ python Run.py -q -P -C out/gh_100k -H enum7 --domain=100 --alpha=0.9 -g independent_grammar -d josh_data -i 100000 -s 100 -c 1000

# LOT model
$ python Run.py -q -P -C out/gh_100k -H out/ngh_100k.p -g lot_grammar -d josh_data -i 100000 -s 100 -c 1000


"""

import os
import re
from optparse import OptionParser

from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.Examples.NumberGame.GrammarInference.Model import *

# ============================================================================================================
# The `run` script
# ============================================================================================================

def run(grammar=lot_grammar, mixture_model=0, data=toy_exp_3,
        iters=10000, skip=10, cap=100, print_stuff='sgr',
        ngh='out/ngh_100k', hypotheses=None, domain=100, alpha=0.9,
        save_file='', csv_freq=500,
        pickle_summary=False, pickle_gh=0):
    """
    Enumerate some NumberGameHypotheses, then use these to sample some GrammarHypotheses over `data`.

    Arguments
    ---------
    grammar : LOTlib.Grammar
        This is our grammar.
    mixture_model : bool
        Are we using the MixtureGrammarHypothesis
    data : list
        List of FunctionData to use as input/output data.
    ngh : str
        Where is the file we save/load our ngh's to/from?
    iters : int
        Number of GrammarHypotheses to sample.
    skip : int
        Collect 1 gh sample every `skip` samples.
    cap : int
        VectorSummary will collect this many GrammarHypothesis samples.
    print_stuff : str
        What do we print? ['s' | 'g' | 'r']
    save_file : str
        If we're pickling or saving csvs, this is the file name to save to.
    # csv_file : str
    #     If saving to csv, this is the file name to save to (don't include .csv!).
    # csv_compare_model : int
    #     Do we save model comparison (regression) plots as we iterate? These take ~15 minutes to save.

    """
    # --------------------------------------------------------------------------------------------------------

    if mixture_model:
        ParameterHypothesis = MixtureGrammarHypothesis
    else:
        ParameterHypothesis = NoConstGrammarHypothesis

    # --------------------------------------------------------------------------------------------------------
    # Load NumberGameHypotheses

    if hypotheses is None:
        # In case we want to enumerate hypotheses instead of loading from file
        if 'enum' in ngh:
            hypotheses = []
            for fn in grammar.enumerate(d=int(re.sub('[a-z]', '', ngh))):
                h = NumberGameHypothesis(grammar=grammar, domain=domain, alpha=alpha)
                h.set_value(fn)
                h.compute_prior()
                hypotheses.append(h)
            ngh += '.p'
        # Load NumberGameHypotheses
        else:
            f = open(ngh, "rb")
            hypotheses = pickle.load(f)
            for h in hypotheses:
                h.grammar = grammar

    # --------------------------------------------------------------------------------------------------------
    # Fill VectorSummary

    grammar_h0 = ParameterHypothesis(grammar, hypotheses, ngh_file=ngh, propose_scale=.1, propose_n=1)
    mh_grammar_sampler = MHSampler(grammar_h0, data, iters)
    mh_grammar_summary = VectorSummary(skip=skip, cap=cap)

    # Print all GrammarRules in grammar with corresponding value index
    if 'r' in print_stuff:
        print '='*100, '\nGrammarRules:'
        for idx in grammar_h0.get_propose_idxs():
            print idx, '\t|  ', grammar_h0.rules[idx]

    if 's' in print_stuff:
        print '^*'*60, '\nGenerating GrammarHypothesis Samples\n', '^*'*60

    # Initialize csv file
    if save_file:
        mh_grammar_summary.csv_initfiles(save_file)

    # Sample GrammarHypotheses!
    for i, gh in enumerate(mh_grammar_summary(mh_grammar_sampler)):

        if save_file and csv_freq and (i % csv_freq == 0):
            mh_grammar_summary.csv_appendfiles(save_file, data)

        # Save to N samples, where N=pickle_gh
        if pickle_gh and (i % pickle_gh == 0):
            mh_grammar_summary.pickle_MAPsample(save_file+'_map_'+str(i/pickle_gh)+'.p')
            mh_grammar_summary.pickle_cursample(save_file+'_cur_'+str(i/pickle_gh)+'.p')

        # Print every N/20 samples
        if 's' in print_stuff:
            if i % (iters/20) is 0:
                for idx in gh.get_propose_idxs():  print idx, '\t|  ', gh.rules[idx], ' --> ', gh.value[idx]
                # print i, '-'*100, '\n', {idx:gh.value[idx] for idx in gh.get_propose_idxs()}
                print gh.prior, gh.likelihood, gh.posterior_score

    # Save summary & print top samples
    if pickle_summary:
        mh_grammar_summary.pickle_summary(filename=save_file+'_summary.p')
    if 'g' in print_stuff:
        mh_grammar_summary.print_top_samples()


# ============================================================================================================
# Main
# ============================================================================================================

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("--pickle-summary",
                      action="store_true", dest="picklesummary", default=False,
                      help="If true, pickle VectorSummary.")
    parser.add_option("-P", "--pickle",
                      type="int", dest="pickle", default=0,
                      help="Pickle MAP & current sample  GH every n steps.")
    parser.add_option("-S", "--save-file",
                      dest="save_file", type="string", default="out/gh_100k",
                      help="Save csv's to this file.")
    parser.add_option("-f", "--ngh",
                      dest="ngh_file", type="string", default="out/ngh_100k.p",
                      help="Where's the file with the NumberGameHypotheses?")
    parser.add_option("--csv-freq",
                      dest="csv_freq", type="int", default=500,
                      help="How often do we save csv files?")

    parser.add_option("-g", "--grammar",
                      dest="grammar", type="string", default="lot_grammar",
                      help="Which grammar do we use? [mix_grammar | independent_grammar | lot_grammar]")
    parser.add_option("--grammar-scale",
                      dest="grammar_scale", type="int", default=0,
                      help="If >0, use a gamma dist. for each rule in the grammar w/ specified scale & " +
                           "shape equal to initial rule prob.")
    parser.add_option("-d", "--data",
                      dest="data", type="string", default="josh",
                      help="Which data do we use? [josh | filename.p]")

    parser.add_option("-i", "--iters",
                      dest="iters", type="int", default=1000000,
                      help="Number of samples to run per chain")
    parser.add_option("-s", "--skip",
                      dest="skip", type="int", default=1000,
                      help="Collect 1 gh sample every `skip` samples.")
    parser.add_option("-c", "--cap",
                      dest="cap", type="int", default=1000,
                      help="VectorSummary will collect this many GrammarHypothesis samples.")

    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=True,
                      help="Print everything!")
    parser.add_option("-q", "--quiet",
                      action="store_true", dest="quiet", default=True,
                      help="Print nothing!")

    (options, args) = parser.parse_args()

    # --------------------------------------------------------------------------------------------------------

    path = os.getcwd() + '/'

    if options.grammar == 'mix':
        grammar = mix_grammar
    elif options.grammar == 'indep':
        grammar = independent_grammar
    elif options.grammar == 'lot':
        grammar = lot_grammar
    else:
        grammar = independent_grammar

    if options.grammar_scale:
        grammar = grammar_gamma(grammar, options.grammar_scale)

    if options.data == 'josh':
        data = import_josh_data()
    elif '.p' in options.data:
        f = open(path + options.data)
        data = pickle.load(f)
    else:
        data = import_pd_data(path + options.data + '.p')

    if options.grammar == 'mix':
        mix = 1
    else:
        mix = 0

    if options.verbose:
        print_stuff = 'sgr'
    elif options.quiet:
        print_stuff = ''
    else:
        print_stuff = 's'

    # --------------------------------------------------------------------------------------------------------

    run(grammar=grammar, mixture_model=mix, data=data,
        iters=options.iters, skip=options.skip, cap=options.cap,
        ngh=options.ngh_file,
        print_stuff=print_stuff,
        save_file=path+options.save_file, csv_freq=options.csv_freq,
        pickle_summary=options.picklesummary, pickle_gh=options.pickle)
