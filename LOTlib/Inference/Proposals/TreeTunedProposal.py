"""
    A proposal function that allows you to specify large chunks of subtrees that are often proposed in the grammar
"""
from copy import copy

from math import log
from random import random
from collections import defaultdict

from LOTlib.Miscellaneous import lambdaTrue, weighted_sample
from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTProposal import LOTProposal


"""

What is the space of all types of tree operations we might try?
I want to be able to write an abitrary function, and have it propose forward and backwards correctly


class Proposal:
    forward, backward,

Inverse-Inlining:
    f(x,y) -> apply(f(x), y)

Insert-Delete:
    f(g(x,y)) -> f(x)
    f(x) -> g(x,y)


def tree_transform(t):
    n, n_ = sample_node(t)

    newn = generate(n.returntype)
    newn_ = newn.log_probability()

class RegenertationMove:
    def forward(t):
        # Sample forwards, returning (tree, forward_prob)


    def backward(t,z):
        # Compute probability of going from t to z




"""







class TreeTunedProposal(LOTProposal):
    """
        A tree-tuned proposal, meaning that we can specify a set of (potentially partial) trees that get higher generation probability.
        *trees* - a dictionary from trees to probabilities. We add "None" to this, and if we sample None, we generate from the grammar
    """

    def __init__(self, grammar, trees, regeneration_probability=0.5):

        self.grammar = grammar
        self.regeneration_probability = 0.5

        assert isinstance(trees, dict)
        # Store each tree by its nonterminal type
        self.nt2trees = defaultdict(dict)
        for t, p in trees.items():
            self.nt2trees[t.returntype][t] = p


    def ttp_generate(self, nt):
        """
        Generate from a nonterminal, returning a log probability. This preferentially upsamples trees in self.trees
        :param nt:
        :return:
        """
        if (nt not in self.nt2trees) or random() < self.regeneration_probability:
            t = self.grammar.generate(nt)
            # return the tree and log probability, keeping track of whether we flipped a coin or not
            return t, t.log_probability() + (log(self.regeneration_probability))*(nt not in self.nt2trees)
        else:
            # select a tree
            t, lp = weighted_sample(self.nt2trees[nt].keys(), probs=lambda x: self.nt2trees[nt][x])

            # and fill in the leaves that are nonterminals, aggregating probability
            for n in t:
                for i,a in enumerate(n.args):
                    if self.grammar.is_nonterminal(a):
                        n.args[i] = self.grammar.generate(a)
                        lp += n.args[i].log_probability()

            return t, lp

    def ttp_log_probability(self, t):
        """
        What is the probability of generating t under this move?

        :param t:
        :return:
        """


        if t in (self.nt2trees[t]):


    def propose_tree(self, t, separate_fb=False, predicate=lambdaTrue):
        """
                If separate_fb=True -- return [newt, f, b], instead of [newt,f-b]
                NOTE: This used to copy but no longer!
        """

        newt = copy(t)

        # sample a subnode
        n, lp = newt.sample_subnode(predicate=predicate)

        # In the context of the parent, resample n according to the grammar
        # We recurse_up in order to add all the parent's rules
        with BVRuleContextManager(self.grammar, n.parent, recurse_up=True):
            t, gp = self.ttp_generate(n.returntype)
            n.setto(t)



        NOOO THE FORWARD AND BACKWARD PROBS ARE WRONG --














        # compute the forward/backward probability
        f = lp + gp +  newt.log_probability()
        b = (log(n.resample_p) - log(newt.sample_node_normalizer(predicate=predicate))) + t.log_probability()

        if separate_fb:
            return [newt, f, b]
        else:
            return [newt,f-b]



