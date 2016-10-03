
"""

Moved some stuff from GrammarInference

"""

from LOTlib.Inference.GrammarInference.GrammarInference import *

DEFAULT_STAN_FILE = os.path.join(os.path.dirname(__file__), 'model-DMPCFG-Binomial.stan')


def make_stan_code(counts, template=DEFAULT_STAN_FILE, log=None):
    """ Create a DM-PCFG stan file for rule_counts, where rule_counts is a dict
        mapping a nonterminal name to the number of rules it uses. This defines
            x_<NTNAME> -- parameters for that nonterminal
            count_<NTNAME> -- counts of each hypothesis' use of nonterminal
        So when you use this, you must define count_<NTNAME> in the data for each nonterminal, and
        make sure that it aligns with
    """
    nts = counts.keys()
    cnts = [counts[nt].shape[1] for nt in nts]

    with open(template, 'r') as f:
        model_code = ''.join( f.readlines() )

    variables = dict()

    variables['COUNT_DEF']     = '\n    '.join(['matrix<lower=0>[N_HYPOTHESES, %s] count_%s;'%(c,x) for x,c in zip(nts, cnts)])
    variables['X_DEF']         = '\n    '.join(['simplex[%s] x_%s;'%(c,x) for x,c in zip(nts, cnts)])

    variables['X_PRIOR']       = '\n    '.join(['x_%s ~ dirichlet(rep_vector(1,%s));'%(x,c) for x,c in zip(nts, cnts)])
    variables['COMPUTE_PRIOR'] = '\n    '.join(['priors <- priors + count_%s * log(x_%s);'%(x,x) for x in nts])

    # Just substitute in
    stan_code = model_code % variables

    if log is not None:
        with open(log+"/model.stan", 'w') as f:
            print >>f, stan_code


    return stan_code

#
# def pretty_print_optimized_parameters(fit):
#     pass




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    # the rule counts: for each nonterminal (dict key)
    # how often does each hypothesis (rows) use each
    # nonterminal (columns)
    # These define the prior
    counts = {'mynt': numpy.matrix([[1,1],[0,1]]),
              'mynt2':numpy.matrix([[2,2],[2,2]])}

    stan_data = {
        'N_HYPOTHESES': 2,
        'N_DATA': 2, # total number of data points
        'N_GROUPS': 2,

        'PriorOffset': [0,0], # prior over and above grammatical parts

        'GroupLength': [1,1], # one data point in each group

        # for computing the posterior in each group
        'L': [[-1, -1], [-2, -2]],

        'NYes':    [8,5],
        'NTrials': [8,20],

        'Output': [ [1,0], [1,0] ]
    }
    stan_data.update({ 'count_%s'%nt:counts[nt] for nt in counts.keys()})

    model_code = make_stan_code(counts)

    print "# Running with code\n", model_code

    sm = pystan.StanModel(model_code=model_code)

    fit = sm.optimizing(data=stan_data)
    print "# Fit:", fit

    samples = sm.sampling(data=stan_data, iter=100, chains=4, sample_file="./stan_samples")

    #
    # fit = pystan.stan(model_code=model_code,
    #                   data=stan_data, iter=100, chains=4)
    # fit.plot().show()
    # print(fit)

    # print(fit.extract(permuted=True))