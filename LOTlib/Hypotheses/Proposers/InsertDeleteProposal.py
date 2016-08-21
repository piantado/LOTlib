
"""Insert / Delete proposals - nests an existing subtree inside a new
rule or deletes the uppermost portion of a subtree and promotes one of
its children (of the same type)

"""

from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import *
from LOTlib.GrammarRule import *
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Miscellaneous import sample1, nicelog

class InsertDeleteProposal(object):
    """The proposal function given by this class isn't ergodic!"""
    def propose(self, **kwargs):
        ret_value, fb = None, None
        while True: # keep trying to propose
            try:
                ret_value, fb = insert_delete_proposal(self.grammar, self.value, **kwargs)
                break
            except ProposalFailedException:
                pass

        ret = self.__copy__(value=ret_value)

        return ret, fb

def can_insert_GrammarRule(r):
    return any([r.nt==a for a in None2Empty(r.to)])

def can_insert_FunctionNode(x):
    return any([x.returntype == a.returntype for a in x.argFunctionNodes()])

def can_delete_GrammarRule(r):
    return any([r.nt==a for a in None2Empty(r.to)])

def can_delete_FunctionNode(x):
    if isinstance(x, BVAddFunctionNode) and x.uses_bv():
        return False
    else:
        return any([x.returntype == a.returntype for a in x.argFunctionNodes()])

def insert_delete_proposal(grammar, t):
    newt = copy(t)

    if random() < 0.5: # insert!

        # Choose a node at random to insert on
        # TODO: We could precompute the nonterminals we can do this move on, if we wanted
        try:
            ni, lp = newt.sample_subnode(can_insert_FunctionNode)
        except NodeSamplingException:
            raise ProposalFailedException

        # is there a rule that expands from ni.returntype to some ni.returntype?
        replicating_rules = filter(can_insert_GrammarRule, grammar.rules[ni.returntype])
        if len(replicating_rules) == 0:
            raise ProposalFailedException

        # sample a rule
        r = sample1(replicating_rules)

        # the functionNode we are building
        fn = r.make_FunctionNodeStub(grammar, ni.parent)

        # figure out which arg will be the existing ni
        replicatingindices = filter( lambda i: fn.args[i] == ni.returntype, xrange(len(fn.args)))
        if len(replicatingindices) <= 0: # should never happen
            raise ProposalFailedException
        
        replace_i = sample1(replicatingindices) # choose the one to replace
        
        ## Now expand the other args, with the right rules in the grammar
        with BVRuleContextManager(grammar, fn, recurse_up=True):

            for i,a in enumerate(fn.args):
                if i == replace_i:
                    fn.args[i] = copy(ni) # the one we replace
                else:
                    fn.args[i] = grammar.generate(a) #else generate like normal

        # we need a count of how many kids are the same afterwards
        after_same_children = sum([x==ni for x in fn.args])
                    
        # perform the insertion
        ni.setto(fn)

        # TODO: fix the fact that there are potentially multiple backward steps to give the equivalent tree
        # need to use the right grammar for log_probability calculations
        with BVRuleContextManager(grammar, fn, recurse_up=True):

            # what is the prob mass of the new stuff?
            new_lp_below =  sum([ grammar.log_probability(fn.args[i]) if (i!=replace_i and isFunctionNode(fn.args[i])) else 0. for i in xrange(len(fn.args))])

            # What is the new normalizer?
            newZ = newt.sample_node_normalizer(can_delete_FunctionNode)
            assert newZ > 0
            
            # forward: choose the node ni, choose the replicating rule, choose which "to" to expand, and generate the rest of the tree
            f = lp - nicelog(len(replicating_rules)) + (nicelog(after_same_children) - nicelog(len(replicatingindices))) + new_lp_below
            # backward: choose the inserted node, choose one of the children identical to the original ni, and deterministically delete
            b = (nicelog(1.0*can_delete_FunctionNode(fn)) - nicelog(newZ)) + (nicelog(after_same_children) - nicelog(len(replicatingindices)))

    else: # delete!

        try: # sample a node at random
            ni, lp = newt.sample_subnode(can_delete_FunctionNode) # this could raise exception

            if ni.args is None: # doesn't have children to promote
                raise NodeSamplingException

        except NodeSamplingException:
            raise ProposalFailedException

        # Figure out which of my children have the same type as me
        replicating_kid_indices = filter(lambda i: isFunctionNode(ni.args[i]) and ni.args[i].returntype == ni.returntype, range(len(ni.args)))
        nrk = len(replicating_kid_indices) # how many replicating kids
        if nrk == 0:
            raise ProposalFailedException

        replicating_rules = filter(can_delete_GrammarRule, grammar.rules[ni.returntype])
        assert len(replicating_rules) > 0 # better be some or where did ni come from?

        samplei = sample1(replicating_kid_indices) # who to promote; NOTE: not done via any weighting

        # We need to be in the right grammar state to evaluate log_probability
        with BVRuleContextManager(grammar, ni.args[samplei], recurse_up=True):

            # Now we must count the multiple ways we could go forward or back
            # Here, we could have sampled any of them equivalent to ni.args[i]
            before_same_children = sum([x==ni.args[samplei] for x in ni.args ]) # how many are the same after?

            # the lp of everything we'd have to create going backwards
            old_lp_below = sum([ grammar.log_probability(ni.args[i]) if (i!=samplei and isFunctionNode(ni.args[i])) else 0. for i in xrange(len(ni.args))])

            # and replace it
            ni.setto( ni.args[samplei] )

            newZ = newt.sample_node_normalizer(resampleProbability=can_insert_FunctionNode)
            
            # forward: choose the node, and then from all equivalent children
            f = lp + (log(before_same_children) - log(nrk))
            # backward: choose the node, choose the replicating rule, choose where to put it, and generate the rest of the tree
            b = (nicelog(1.0*can_insert_FunctionNode(ni)) - nicelog(newZ)) - nicelog(len(replicating_rules)) + (nicelog(before_same_children) - nicelog(nrk)) + old_lp_below

    return [newt, f-b]

if __name__ == "__main__": # test code
    ## NOTE: IN REAL LIFE, MIX WITH REGENERATION PROPOSAL -- ELSE NOT ERGODIC

    from LOTlib.Examples.Magnetism.Simple import grammar, make_data
    from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
    from LOTlib.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
    from LOTlib.Inference.Samplers.StandardSample import standard_sample

    class IDHypothesis(BinaryLikelihood, InsertDeleteProposal, LOTHypothesis):
        """
        A recursive LOT hypothesis that computes its (pseudo)likelihood using a string edit
        distance
        """
        def __init__(self, **kwargs ):
            LOTHypothesis.__init__(self, grammar, display='lambda x,y: %s', **kwargs)

    def make_hypothesis(**kwargs):
        return IDHypothesis(**kwargs)

    standard_sample(make_hypothesis, make_data, save_top=False, show_skip=9)
