"""
A simple demo of inference with GrammarHypothesis, VectorSummary, NumberGameHypothesis, & MHSampler.


Command-line Args
-----------------
-p --pickle
    If there's a value here, pickle VectorSummary.
-csv --csvfile
    Save csv's to this file.
-ngh --ngh_file
    Where's the file with the NumberGameHypotheses?

-g --grammar
    Which grammar do we use? [mix_grammar | independent_grammar | lot_grammar]
-d --data
    Which data do we use? [josh_data | ??]

-i --iters
    Number of samples to run per chain
-sk --skip
    Collect 1 gh sample every `skip` samples.
-cap --cap
    VectorSummary will collect this many GrammarHypothesis samples.

-v --verbose
    Print everything!
-q --quiet
    Print nothing!


Example
-------
$ python Run.py -q -p -csv out/gh_100k -ngh out/ngh_100k.p -g lot_grammar -d josh_data -i 100000 -sk 100 -cap 1000


"""

import pickle, os
from optparse import OptionParser
from LOTlib.Inference.MetropolisHastings import MHSampler
from LOTlib.MPI.MPI_map import MPI_unorderedmap
from LOTlib.MCMCSummary.TopN import TopN
from LOTlib.Examples.NumberGame.GrammarInference.Model import *


# ============================================================================================================
# Parsing command-line options
# ============================================================================================================

parser = OptionParser()

parser.add_option("-p", "--pickle",
                  action="store_true", dest="pickle", default=False,
                  help="If there's a value here, pickle VectorSummary.")
parser.add_option("-csv", "--csvfile",
                  dest="csv_file", type="string", default="out/gh_100k",
                  help="Save csv's to this file.")
parser.add_option("-ngh", "--ngh_file",
                  dest="ngh_file", type="string", default="out/ngh_100k.p",
                  help="Where's the file with the NumberGameHypotheses?")

parser.add_option("-g", "--grammar",
                  dest="grammar", type="string", default="lot_grammar",
                  help="Which grammar do we use? [mix_grammar | independent_grammar | lot_grammar]")
parser.add_option("-d", "--data",
                  dest="data", type="string", default="josh_data",
                  help="Which data do we use? [josh_data | ??]")

parser.add_option("-i", "--iters",
                  dest="iters", type="int", default=1000000,
                  help="Number of samples to run per chain")
parser.add_option("-sk", "--skip",
                  dest="skip", type="int", default=1000,
                  help="Collect 1 gh sample every `skip` samples.")
parser.add_option("-cap", "--cap",
                  dest="cap", type="int", default=1000,
                  help="VectorSummary will collect this many GrammarHypothesis samples.")

parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=True,
                  help="Print everything!")
parser.add_option("-q", "--quiet",
                  action="store_true", dest="quiet", default=True,
                  help="Print nothing!")

# TODO -- parallelize!
# parser.add_option("-c", "--chains", dest="chains", type="int", default=1,
#                   help="Number of chains to run on each data input")
# parser.add_option("-n", dest="N", type="int", default=1000,
#                   help="Only keep top N samples per MPI run (if we're doing MPI), or total (if not MPI)")
# parser.add_option("-mpi", action="store_true", dest="mpi", default=True,
#                   help="Do we use MPI?")

(options, args) = parser.parse_args()


# ============================================================================================================
# The `run` script
# ============================================================================================================

def run(grammar=lot_grammar, mixture_model=0, data=josh_data,
        iters=10000, skip=10, cap=100, ngh='out/ngh_100k',
        print_stuff='sgr', plot_type='',
        pickle_file='', csv_file=''):
    """
    Enumerate some NumberGameHypotheses, then use these to sample some GrammarHypotheses over `data`.

    Arguments
    ---------
    grammar : LOTlib.Grammar
        This is our grammar.
    mixture_model : bool
        Are we using the MixtureGrammarHypothesis
    data : list
        List of FunctionNodes to use as input/output data.
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
    pickle_file : str
        If we're pickling, this is the file name to save to.
    csv_file : str
        If saving to csv, this is the file name to save to (don't include .csv!).

    """
    # --------------------------------------------------------------------------------------------------------

    if mixture_model:
        ParameterHypothesis = MixtureGrammarHypothesis
    else:
        ParameterHypothesis = NumberGameHypothesis

    # --------------------------------------------------------------------------------------------------------
    # Load NumberGameHypotheses

    f = open(ngh, "rb")
    hypotheses = pickle.load(f)
    for h in hypotheses:
        h.grammar = grammar

    # --------------------------------------------------------------------------------------------------------
    # Print all GrammarRules in our Grammar, with corresponding value index

    if print_stuff is 'r' in print_stuff:
        print '='*100, '\nGrammarRules:'
        rules = [r for sublist in grammar.rules.values() for r in sublist]
        for i, r in enumerate(rules):
            print i, '\t|  ', r

    # --------------------------------------------------------------------------------------------------------
    # Fill VectorSummary

    if 's' in print_stuff:
        print '^*'*60, '\nGenerating GrammarHypothesis Samples\n', '^*'*60

    grammar_h0 = ParameterHypothesis(grammar, hypotheses, propose_step=.1, propose_n=1)
    mh_grammar_sampler = MHSampler(grammar_h0, data, n)
    mh_grammar_summary = VectorSummary(skip=skip, cap=cap)

    if csv_file:
        mh_grammar_summary.csv_initfiles(csv_file)
        mh_grammar_summary.csv_compare_model_human(csv_file+'_data_h0.csv', data, grammar_h0)

    for i, gh in enumerate(mh_grammar_summary(mh_grammar_sampler)):

        # Save to csv every 200 samples from 0 to 10k, then every 1000
        if csv_file:
            mh_grammar_summary.csv_appendfiles(csv_file, data)

        # Print every N/20 samples
        if 's' in print_stuff:
            if i % (iters/20) is 0:
                print ['%.3f' % v for v in [gh.value[idx] for idx in gh.propose_idxs]], '\n', i, '-'*100
                print gh.prior, gh.likelihood, gh.posterior_score

    # Save summary & print top samples
    if pickle_file:
        mh_grammar_summary.pickle_summary(filename=pickle_file)
    if 'g' in print_stuff:
        mh_grammar_summary.print_top_samples()


# ============================================================================================================
# Main
# ============================================================================================================

if __name__ == "__main__":
    path = os.getcwd()

    if options.pickle:
        pickle_file = path + options.csv_file + '.p'
    else:
        pickle_file = ''

    if options.grammar is 'mix_grammar':
        grammar = mix_grammar
    elif options.grammar is 'independent_grammar':
        grammar = independent_grammar
    elif options.grammar is 'lot_grammar':
        grammar = lot_grammar

    if options.data is 'josh_data':
        data = josh_data
    else:
        data = josh_data

    if options.grammar is 'mix_grammar':
        mix = 1
    else:
        mix = 0

    if options.verbose:
        print_stuff = 'sgr'
    elif options.quiet:
        print_stuff = ''
    else:
        print_stuff = 's'

    run(grammar=grammar, mixture_model=mix, data=data,
        iters=options.iters, skip=options.skip, cap=options.cap,
        ngh=options.ngh_file,
        print_stuff=print_stuff,
        pickle_file=pickle_file, csv_file=path+options.csv_file)
