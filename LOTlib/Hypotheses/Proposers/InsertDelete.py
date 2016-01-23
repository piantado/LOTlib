
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Miscellaneous import sample1, nicelog
from LOTlib.FunctionNode import *
from LOTlib.GrammarRule import *
from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import NodeSamplingException



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

    if random() < 0.5: # So we insert

        # Choose a node at random to insert on
        # TODO: We could precompute the nonterminals we can do this move on, if we wanted
        try:
            ni, lp = newt.sample_subnode(can_insert_FunctionNode)
        except NodeSamplingException:
            raise ProposalFailedException

        # Since it's an insert, see if there is a (replicating) rule that expands
        # from ni.returntype to some ni.returntype
        replicating_rules = filter(can_insert_GrammarRule, grammar.rules[ni.returntype])
        if len(replicating_rules) == 0:  return [newt, 0.0]

        # sample a rule and compute its probability (not under the predicate)
        r = sample1(replicating_rules)

        # the functionNode we are building
        fn = r.make_FunctionNodeStub(grammar, ni.parent)

        # figure out which arg will be the existing ni
        replicatingindices = filter( lambda i: fn.args[i] == ni.returntype, xrange(len(fn.args)))
        assert replicatingindices > 0 # since that's what a replicating rule is
        replace_i = sample1(replicatingindices) # choose the one to replace
        fn.args[replace_i] = copy(ni) # the one we replace

        ## Now expand the other args, with the right rules in the grammar
        with BVRuleContextManager(grammar, fn, recurse_up=True):

            # and generate the args below
            for i,a in enumerate(fn.args):
                if i != replace_i:
                    fn.args[i] = grammar.generate(a) #else generate like normalized

        # we need a count of how many kids are the same afterwards
        after_same_children = sum([x==ni for x in fn.args])

        ni.setto(fn)

        with BVRuleContextManager(grammar, fn, recurse_up=True):

            # what is the prob mass of the new stuff?
            new_lp_below =  sum([ grammar.log_probability(fn.args[i]) if (i!=replace_i and isFunctionNode(fn.args[i])) else 0. for i in xrange(len(fn.args))])
            # What is the new normalizer?
            newZ = newt.sample_node_normalizer(can_delete_FunctionNode)
            assert newZ > 0
            # To sample forward: choose the node ni, choose the replicating rule, choose which "to" to expand (we could have put it on any of the replicating rules that are identical), and genreate the rest of the tree
            f = lp + (-log(len(replicating_rules))) + (log(after_same_children)-log(len(replicatingindices))) + new_lp_below
            # To go backwards, choose the inserted rule, and any of the identical children, out of all replicators
            b = (log(1.0*can_delete_FunctionNode(fn)) - log(newZ)) + (log(after_same_children) - log(len(fn.args)))

    else: # A delete move!

        # Sample a node at random
        try:
            ni, lp = newt.sample_subnode(can_delete_FunctionNode) # this could raise exception

            # Really, it had to be not None
            if ni.args is None:
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
            ni.args[samplei].parent = ni.parent # update this first ;; TODO: IS THIS NECSESARY?
            ni.setto( ni.args[samplei] )

            # And compute f/b probs
            newZ = newt.sample_node_normalizer(resampleProbability=can_insert_FunctionNode)
            # To go forward, choose the node, and then from all equivalent children
            f = lp + (log(before_same_children) - log(nrk))
            # To go back, choose the node, choose the replicating rule, choose where to put it, and generate the rest of the tree
            b = (nicelog(1.0*can_insert_FunctionNode(ni)) - nicelog(newZ))  + -nicelog(len(replicating_rules)) + (nicelog(before_same_children) - nicelog(nrk)) + old_lp_below

    return [newt, f-b]

if __name__ == "__main__":

    from LOTlib import break_ctrlc
    #from LOTlib.Examples.Number.Shared import grammar, make_h0, generate_data
    #data = generate_data(300)

    ## NOTE: TO NORMALLY USE THIS, YOU MUST MIX WITH REGENERATION PROPOSAL -- ELSE NOT ERGODIC

    from LOTlib.Examples.Magnetism.Simple.Run import grammar, make_h0, data

    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler

    idp = InsertDeleteProposer(grammar)

    #data = generate_data(100)
    h = make_h0(proposal_function=idp)
    for h in break_ctrlc(MHSampler(h, data, 100000)):
        print h.posterior_score, h

    """
    for _ in xrange(100):
        t = grammar.generate()
        print "\n\n", t
        for _ in xrange(10):
            print "\t", idp.propose_tree(t)
    """



