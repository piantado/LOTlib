""" Copy proposal - choose two nodes of type X and copy one to the other
    (NOT ERGODIC)!
"""

from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import NodeSamplingException
from LOTlib.Hypotheses.Proposers import *
from LOTlib.Miscellaneous import lambdaOne, Infinity, logsumexp
from LOTlib.Subtrees import least_common_difference
from copy import copy, deepcopy
from math import log
from random import random

class CopyProposal(object):
    def propose(self, **kwargs):
        ret_value, fb = None, None
        while not ret_value: # keep trying to propose
            try:
                ret_value, fb =  copy_proposal(self.grammar, self.value, **kwargs)
            except ProposalFailedException:
                pass
        ret = self.__copy__(value=ret_value)
        return ret, fb

def copy_proposal(grammar, tree, resampleProbability=lambdaOne):
    t = copy_subtree(grammar,tree,resampleProbability)
    fb = copy_fb(grammar,tree,t,resampleProbability)
    return t,fb

def give_grammar(grammar,node):
    # BVRuleContextManager gives the grammar used inside a node, not
    # at the node itself, so we consider the node's parent
    with BVRuleContextManager(grammar, node.parent, recurse_up=True):
        g = deepcopy(grammar)
    return g

def copy_subtree(grammar,tree,resampleProbability=lambdaOne):
    new_t = copy(tree)

    # sample a source and (possibly identical) target with the same grammar.
    try:
        src, lp_choosing_src_in_old_tree = new_t.sample_subnode(resampleProbability)
        src_grammar = give_grammar(grammar,src)
        good_choice = lambda x: 1.0 if ((give_grammar(grammar,x) == src_grammar) and
                                        (x.returntype == src.returntype)) else 0.0
        target, lp_choosing_target_in_old_tree = new_t.sample_subnode(good_choice)
    except NodeSamplingException:
        raise ProposalFailedException
    
    new_src = deepcopy(src)
    new_src.parent = target.parent
    target.setto(new_src)
    
    return new_t

def copy_fb(grammar, t1, t2, resampleProbability=lambdaOne):
    return (copy_probability(grammar,t1,t2,resampleProbability) -
            copy_probability(grammar,t2,t1,resampleProbability))

def nodes_equal_except_parents(grammar,n1,n2):
    return ((n1.name == n2.name) and
            (n1.args == n2.args) and
            (n1.returntype == n2.returntype) and
            (give_grammar(grammar,n1) == give_grammar(grammar,n2)))

def copy_probability(grammar, t1, t2, resampleProbability=lambdaOne, recurse=True):
    chosen_node1 , chosen_node2 = least_common_difference(t1,t2)
    print "t1: {0}".format(t1)
    print "t2: {0}".format(t2)
    print "node 1: {0}".format(chosen_node1)
    print "node 2: {0}".format(chosen_node2)

    lps = []
    if chosen_node1 is None: # any node in the tree could have been copied
        for node in t1:
            could_be_source = lambda x: 1.0 * nodes_equal_except_parents(grammar,x,node) * resampleProbability(x)
            numerator =  (t1.sample_node_normalizer(could_be_source) - could_be_source(node))
            lp_of_choosing_source = (log(numerator) - log(t1.sample_node_normalizer(resampleProbability))) if numerator > 0 else -Infinity
            lp_of_choosing_target = t1.sampling_log_probability(chosen_node1,resampleProbability=resampleProbability)
            lps += [lp_of_choosing_source + lp_of_choosing_target]
    else: # we have a specific path up the tree
        while chosen_node1:
            could_be_source = lambda x: 1.0 * nodes_equal_except_parents(grammar,x,chosen_node2) * resampleProbability(x)

            mass_on_sources = t1.sample_node_normalizer(could_be_source)
            if mass_on_sources == 0.0:
                lps += [-Infinity]
            else:
                lp_of_choosing_source = log(t1.sample_node_normalizer(could_be_source)) - log(t1.sample_node_normalizer(resampleProbability))
                lp_of_choosing_target = t1.sampling_log_probability(chosen_node1,resampleProbability=resampleProbability)
                lps += [lp_of_choosing_source + lp_of_choosing_target]

            if recurse:
                chosen_node1 = chosen_node1.parent
                chosen_node2 = chosen_node2.parent
            else:
                chosen_node1 = None

    return logsumexp(lps)

if __name__ == "__main__": # test code

    # We'd probably see better performance on a grammar with fewer
    # distinct types, but this one is a good testbed *because* it's
    # complex (lambdas, etc.)
    from LOTlib.Examples.Magnetism.Simple import grammar, make_data
    from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
    from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    class CRHypothesis(BinaryLikelihood, CopyProposal, LOTHypothesis):
        """
        A recursive LOT hypothesis that computes its (pseudo)likelihood using a string edit
        distance
        """
        def __init__(self, **kwargs ):
            LOTHypothesis.__init__(self, grammar, display='lambda x,y: %s', **kwargs)

    def make_hypothesis(**kwargs):
        return CRHypothesis(**kwargs)

    standard_sample(make_hypothesis, make_data, save_top=False)
