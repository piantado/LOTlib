"""
    Utility for precomputing some key information in doing grammar inference
"""

import numpy
import os
import re
from LOTlib.FunctionNode import BVUseFunctionNode

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
from LOTlib.FunctionNode import FunctionNode

from collections import Counter


def create_counts(grammar, hypotheses, which_rules=None, log=None):
    """
        Make the rule counts for each nonterminal. This returns three things:
            count[nt][trees,rule] -- a hash from nonterminals to a matrix of rule counts
            sig2idx[signature] - a hash from rule signatures to their index in the count matrix
            prior_offset[trees] - how much the prior should be offset for each hypothesis (incorporating the rules not in which_rules)

        which_rules -- optionally specify a subset of rules to include. If None, all are used.
    """

    grammar_rules = [r for r in grammar] # all of the rules

    if which_rules is None:
        which_rules = grammar_rules

    assert set(which_rules) <= set(grammar_rules), "*** Somehow we got a rule not in the grammar: %s" % str(set(grammar_rules)-set(which_rules))


    # Check the grammar rules
    for r in which_rules:
        assert not re.search(r"[^a-zA-Z_0-9]", r.nt), "*** Nonterminal names must be alphanumeric %s"%str(r)


    # convert rules to signatures for checking below
    which_signatures = { r.get_rule_signature() for r in which_rules }

    prior_offset = [0.0 for _ in hypotheses] # the log probability of all rules NOT in grammar_rules

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
    counts = { nt: numpy.zeros( (len(hypotheses), next_free_idx[nt])) for nt in next_free_idx.keys() }

    for i, h in enumerate(hypotheses):

        # Build up a lost of all the nodes, depending on what type of hypotheses we got
        if isinstance(h, FunctionNode):
            nodes = [n for n in h]
        if isinstance(h, LOTHypothesis):
            nodes = [n for n in h.value]
        elif isinstance(h, SimpleLexicon):
            nodes = []
            for w in h.value.keys():
                assert isinstance(h.value[w], LOTHypothesis), "*** Not implemented unless Lexicon values are LOTHypotheses"
                nodes.extend([n for n in h.value[w].value])

        # Iterate through the nodes and count rule usage
        for n in nodes:
            if n.get_rule_signature() in which_signatures:
                if not isinstance(n, BVUseFunctionNode): ## NOTE: Not currently doing bound variables
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

def export_rule_labels(which_rules):
    """ Exports an integer string coding a single line that C will print as a header for the rule names.
    This uses an integer array because coding char arrays seemed buggy.
    It takes which_rules so that the order is the same as in the other exported parts
    """

    out = []
    for r in which_rules:

        s = ''
        if r.name is '':
            s = r.nt
        else:
            s = r.name

        if r.to is not None:
            s += "." + '.'.join(r.to)

        # remove quotes
        s = re.sub(r"['\"]", "", s)

        out.append(s)

    return [ord(v) for v in '\t'.join(out) + '\0']
