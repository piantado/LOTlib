"""
    This version incrementally adds symbols and then does not change them.

    Todo: show highest prob "mistake" strings

"""
from search import *
from LOTlib.MPI.MPI_map import get_rank

LARGE_SAMPLE = 10000 # sample this many and then re-normalize to fractional counts


def run(options, ndata):
    if LOTlib.SIG_INTERRUPTED:
        return 0, set()

    print '*'*20, 'Running Simulation', '*'*20

    language = eval(options.LANG+"()")
    data = language.sample_data(LARGE_SAMPLE)
    assert len(data) == 1
    # renormalize the counts
    for k in data[0].output.keys():
        data[0].output[k] = float(data[0].output[k] * ndata) / LARGE_SAMPLE

    # modify data according to mode
    if options.MODE == 'si':
        # NOTE: 3 stages are hard coded here
        num = options.datamax / 3
        if ndata <= num:
            maxlen = 2
        elif ndata <= num * 2:
            maxlen = 4
        else:
            maxlen = 6
        AnBn.stageddata(data, maxlen)
    elif options.MODE == 'un':
        AnBn.uniformdata(data)

    # Now add the rules to the grammar
    grammar = deepcopy(base_grammar)
    for t in language.terminals():  # add in the specifics
        grammar.add_rule('ATOM', "'%s'" % t, None, 1.0)

    tn = TopN(N=options.TOP_COUNT)

    # set up the hypothesis
    h0 = IncrementalLexiconHypothesis(grammar=grammar)
    h0.set_word(0, h0.make_hypothesis(grammar=grammar)) # make the first word at random
    h0.N = 1

    step_cnt = 0
    flag = True

    for outer in xrange(options.N): # how many do we add?

        # and re-set the posterior or else it's something weird
        h0.compute_posterior(data)

        # now run mcmc
        for h in break_ctrlc(MHSampler(h0, data, steps=options.STEPS)):
            tn.add(copy(h))

            if step_cnt % 50 == 0:
                p, r, outputs = language.estimate_precision_and_recall(h, data, truncate=True)
                f_score = 0 if p == r == 0 else 2*p*r / (p+r)
                # if f-score larger than a threshold then a good hypothesis is found
                if f_score > 0.8:
                    outf = open('simulation_stats_'+str(get_rank())+'.txt', 'a')
                    print >> outf, ndata, step_cnt, p, r, f_score, h.posterior_score, h.prior, h.likelihood, h.likelihood / ndata, h,
                    print >> outf, outputs
                    flag = False
                    outf.close()
                    break

            if options.TRACE and step_cnt % options.SKIP == 0:
                p, r, outputs = language.estimate_precision_and_recall(h, data, truncate=True)
                f_score = 0 if p == r == 0 else 2*p*r / (p+r)
                print get_rank(), ndata, step_cnt, p, r, f_score, h.posterior_score, h.prior, h.likelihood, h.likelihood / ndata, h
                v = h()
                sortedv = sorted(v.items(), key=operator.itemgetter(1), reverse=True )
                print "{" + ', '.join(["'%s':%s"% sortedv[i] for i in xrange(min(len(sortedv), 6))]) + "}\n"

            step_cnt += 1

        if not flag:
            break

        # and start from where we ended
        h0 = copy(h)
        h0.deepen()

    if flag:
        outf = open('simulation_stats_'+str(get_rank())+'.txt', 'a')
        print >> outf, ndata, options.STEPS*options.N, 0, 0, 0
        outf.close()

    return ndata, tn



if __name__ == "__main__":
    """
        example:
            mpiexec -n 12 python simulation.py --mode=si
    """
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("--language", dest="LANG", type="string", default='AnBn', help="name of a language")
    parser.add_option("--steps", dest="STEPS", type="int", default=8000, help="Number of samples to run")
    parser.add_option("--skip", dest="SKIP", type="int", default=500, help="Print out every this many")
    parser.add_option("--top", dest="TOP_COUNT", type="int", default=10, help="Top number of hypotheses to store")
    parser.add_option("--N", dest="N", type="int", default=1, help="number of inner hypotheses")
    parser.add_option("--out", dest="OUT", type="str", default="out/", help="Output directory")
    parser.add_option("--trace", dest="TRACE", action="store_true", default=True, help="Show every step?")

    # following human experiment setup: starting with 12 strings, 12 blocks, each adds 12 more strings
    parser.add_option("--ndata", dest="ndata", type="int", default=12, help="number of data steps to run")
    parser.add_option("--datamin", dest="datamin", type="int", default=12, help="Min data to run (>0 due to log)")
    parser.add_option("--datamax", dest="datamax", type="int", default=144, help="Max data to run")

    # specify the mode: staged-input (si), uniformed (un) or skewed-frequency (default)
    parser.add_option("--mode", dest="MODE", type="string", default='default', help="si un or default")
    (options, args) = parser.parse_args()

    # Set the output
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    # Save options
    if is_master_process():
        display_option_summary(options)
        sys.stdout.flush()

    # different amount of data for each MCMC run
    DATA_RANGE = np.linspace(options.datamin, options.datamax, num=options.ndata)# [1000] # np.arange(1, 1000, 1)
    #random.shuffle(DATA_RANGE) # run in random order

    args = list(itertools.product([options], DATA_RANGE))

    for ndata, tn in MPI_unorderedmap(run, args):
        # for h in tn:
        #     print ndata, h.posterior_score, h.prior, h.likelihood, h.likelihood / ndata
        #     v = h()
        #     sortedv = sorted(v.items(), key=operator.itemgetter(1), reverse=True)
        #     print "{" + ', '.join(["'%s':%s" % i for i in sortedv]) + "}"
        #     print h  # must add \0 when not Lexicon
        pass
    sys.stdout.flush()

    print "# Finishing"


