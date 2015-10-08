"""
    Fit a grammar model in stan

    TODO: Include the option to NOT use some grammar rules!!
        Thsi will require a prior_offset, that takes into account those rules' probability mass

    TODO: FIX MISSING BOUND VARIABLES

    TODO: MAYBE MOVE LOGGING IN HERE?

    Each of these files optionally takes a "log" argument, which is a directory
    into which they will dump their information!

    -- USE ADVI!!! -- AUTOMATIC VARIATIONAL INFERENCE


"""

import pystan
import numpy
import os
from LOTlib.FunctionNode import BVUseFunctionNode

DEFAULT_STAN_FILE = os.path.join(os.path.dirname(__file__), 'model-DMPCFG-Binomial.stan')

from collections import Counter
def create_counts(grammar, trees, which_rules=None, log=None):
    """
        Make the rule counts for each nonterminal. This returns three things:
            count[nt][trees,rule] -- a hash from nonterminals to a matrix of rule counts
            sig2idx[signature] - a hash from rule signatures to their index in the count matrix
            prior_offset[trees] - how much the prior should be offset for each hypothesis

        which_rules -- optionally specify a subset of rules. If None, all are used.
    """

    grammar_rules = [r for r in grammar] # all of the rules

    if which_rules is None:
        which_rules = grammar_rules

    assert which_rules <= grammar_rules, "*** Somehow we got a rule not in the grammar!"

    # convert rules to signatures for checking below
    which_signatures = { r.get_rule_signature() for r in which_rules }

    prior_offset = [0.0 for _ in trees] # the log probability of all rules NOT in grammar_rules

    # First set up the mapping between signatures and indexes
    sig2idx = dict() # map each signature to a unique index in its nonterminal
    next_free_idx = Counter()
    for r in which_rules:
        nt, s = r.nt, r.get_rule_signature()

        if s not in sig2idx:
            sig2idx[s] = next_free_idx[nt]
            next_free_idx[nt] += 1

    # Now map through the trees and compute the counts, as well as
    # the prior_offsets

    # set up the counts as all zeros
    counts = { nt: numpy.zeros( (len(trees), next_free_idx[nt])) for nt in next_free_idx.keys() }

    for i, tree in enumerate(trees):

        for n in tree:

            if n.get_rule_signature() in which_signatures:

                if not isinstance(n, BVUseFunctionNode):
                    nt, s = n.returntype, n.get_rule_signature()
                    counts[nt][i, sig2idx[s]] += 1
            else:
                # else this expansion should be counted towards the offset
                # (NOT the entire grammar.log_probability, since that counts everything below)
                prior_offset[i] += grammar.single_probability(n)

    # Do our logging if we should
    if log is not None:

        for nt in counts.keys():
            with open(log+"/counts_%s.txt"%nt, 'w') as f:
                for r in xrange(counts[nt].shape[0]):
                    print >>f, r, ' '.join(map(str, counts[nt][r,:].tolist()))

        with open(log+"/sig2idx.txt", 'w') as f:
            for s in sorted(sig2idx.keys(), key=lambda x: (x[0], sig2idx[x]) ):
                print >>f,  s[0], sig2idx[s], "\"%s\"" % str(s)

        with open(log+"/prior_offset.txt", 'w') as f:
            for i, x in enumerate(prior_offset):
                print >>f, i, x

    return counts, sig2idx, prior_offset

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


def pretty_print_optimized_parameters(fit):
    pass




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