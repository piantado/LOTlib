from math import log
from random import random

from LOTlib.Miscellaneous import sample1
from LOTlib.FunctionNode import *
from LOTProposal import LOTProposal
from LOTlib.GrammarRule import *
from LOTlib.BVRuleContextManager import BVRuleContextManager


def is_replicating_GrammarRule(r):
    """
        A Grammar is replicating if one if its children has the same returntype, and it's not a BVAdd
        It cant' be a BVAdd because by deleting it we will lose its bound variable
    """
    return (not isinstance(r, BVAddGrammarRule)) and any([r.nt==a for a in None2Empty(r.to)])

def is_replicating_FunctionNode(x):
    """
            A function node is replicating (by definition) if one of its children is of the same type
    """
    return any([x.returntype == a.returntype for a in x.argFunctionNodes()])

def is_not_BVAddFunctionNode(x):
    return not isinstance(x, BVAddFunctionNode)

class InsertDeleteProposal(LOTProposal):
    """
            This class is a mixture of standard rejection proposals, and insert/delete proposals
            
            TODO: Allow variable insert-vs-delete probabilities (must be tracked in forward/back)
            
            NOTE: Without these moves, you will often generate a useful part of a function in, say, an AND, and
                    you can't remove the AND, meaning you will just make it some subtree equal to "True" --
                    e.g. this allows AND(T, True)  -> T, which is what you want. Otherwise, you have trouble moving out of those

            NOTE: This does not go on lambdas -- they're too hard to think about for now.. But even "not" doing them, there are asymmetries--we want to not treat them as "replicating rules", so we can't have sampled them, and also can't delete them

    """

    def __init__(self, grammar, insert_delete_probability=0.5):
        self.__dict__.update(locals())

    
    def propose_tree(self, t):

        newt = copy(t)
        fb = 0.0 # the forward/backward prob we return

        if random() < 0.5: # So we insert
            
            # Choose a node at random to insert on
            # TODO: We could precompute the nonterminals we can do this move on, if we wanted
            ni, lp = newt.sample_subnode(is_not_BVAddFunctionNode)
            if ni is None or ni.args is None: return [newt, fb]
            
            # Since it's an insert, see if there is a (replicating) rule that expands
            # from ni.returntype to some ni.returntype
            replicating_rules = filter(is_replicating_GrammarRule, self.grammar.rules[ni.returntype])
            if len(replicating_rules) == 0:  return [newt, fb] 
            
            # sample a rule and compute its probability (not under the predicate)            
            r = sample1(replicating_rules)
            gp = log(r.p) - log(sum([x.p for x in self.grammar.rules[ni.returntype]])) # this is the probability overall in the grammar, not my prob of sampling
            
            # the functionNode we are building
            fn = r.make_FunctionNodeStub(self, gp, ni.parent) 
            
            # figure out which arg will be the existing ni
            replicatingindices = filter( lambda i: fn.args[i] == ni.returntype, xrange(len(fn.args)))
            assert replicatingindices > 0 # since that's what a replicating rule is
            replace_i = sample1(replicatingindices) # choose the one to replace
            fn.args[replace_i] = copy(ni) # the one we replace
            
            ## Now expand the other args, with the right rules in the grammar
            with BVRuleContextManager(self.grammar, fn, recurse_up=True):
                
                # fix the fact that ni's generation probabilities may be wrong
                # TODO: MAY NOT BE NEEDED?
                self.grammar.recompute_generation_probabilities(ni)
                
                # and generate the args below
                for i,a in enumerate(fn.args):
                    if i != replace_i:
                        fn.args[i] = self.grammar.generate(a) #else generate like normalized
            
            # we need a count of how many kids are the same afterwards
            after_same_children = sum([x==ni for x in fn.args])
            
            ni.setto(fn)
                        
            # what is the prob mass of the new stuff?
            new_lp_below =  sum([fn.args[i].log_probability() if i!=replace_i else 0. for i in xrange(len(fn.args))])
            # What is the new normalizer?
            newZ = newt.sample_node_normalizer(is_not_BVAddFunctionNode)
            assert newZ > 0
            # To sample forward: choose the node ni, choose the replicating rule, choose which "to" to expand (we could have put it on any of the replicating rules that are identical), and genreate the rest of the tree
            f = lp + (-log(len(replicating_rules))) + (log(after_same_children)-log(len(replicatingindices))) + new_lp_below
            # To go backwards, choose the inserted rule, and any of the identical children, out of all replicators
            b = (log(fn.resample_p) - log(newZ)) + (log(after_same_children) - log(len(fn.args)))
            
        else: # A delete move!
            
            # Sample a node at random
            ni, lp = newt.sample_subnode(is_not_BVAddFunctionNode)
            if ni is None or ni.args is None:  return [newt, fb]
            
            # Figure out which of my children have the same type as me
            replicating_kid_indices = filter(lambda i: isFunctionNode(ni.args[i]) and ni.args[i].returntype == ni.returntype, range(len(ni.args)))
            nrk = len(replicating_kid_indices) # how many replicating kids
            if nrk == 0:  return [newt, fb] # if no replicating rules here
            
            replicating_rules = filter(is_replicating_GrammarRule, self.grammar.rules[ni.returntype])
            assert len(replicating_rules) > 0 # better be some or where did ni come from?
            
            samplei = sample1(replicating_kid_indices) # who to promote; NOTE: not done via any weighting
            
            # Now we must count the multiple ways we could go forward or back
            # Here, we could have sampled any of them equivalent to ni.args[i]
            before_same_children = sum([x==ni.args[samplei] for x in ni.args ]) # how many are the same after?

            # the lp of everything we'd have to create going backwards
            old_lp_below = sum([ni.args[i].log_probability() if i!=samplei else 0. for i in xrange(len(ni.args))])

            # and replace it
            ni.args[samplei].parent = ni.parent # update this first ;; TODO: IS THIS NECSESARY?
            ni.setto( ni.args[samplei] ) 
            
            # fix the generation probs
            # TODO: Is this necessary? probably not w/o bound variables
            self.grammar.recompute_generation_probabilities(ni)
            
            # And compute f/b probs
            newZ = newt.sample_node_normalizer(is_not_BVAddFunctionNode)
            # To go forward, choose the node, and then from all equivalent children
            f = lp + (log(before_same_children) - log(nrk))
            # To go back, choose the node, choose the replicating rule, choose where to put it, and generate the rest of the tree
            b = (log(ni.resample_p) - log(newZ))  + -log(len(replicating_rules)) + (log(before_same_children) - log(nrk)) + old_lp_below

        return [newt, f-b]
    
if __name__ == "__main__":

    from LOTlib import lot_iter
    #from LOTlib.Examples.Number.Shared import grammar, make_h0, generate_data
    #data = generate_data(300)
    
    from LOTlib.Examples.Magnetism.Simple.Global import grammar, make_h0, data
    
    
    
    from LOTlib.Inference.MetropolisHastings import MHSampler
    
    idp = InsertDeleteProposal(grammar)
     
    #data = generate_data(100)
    h = make_h0(proposal_function=idp) 
    for h in lot_iter(MHSampler(h, data, 100000)):
        print h.posterior_score, h
        
    """
    for _ in xrange(100):
        t = grammar.generate()
        print "\n\n", t
        for _ in xrange(10):
            print "\t", idp.propose_tree(t)
    """        
        
    
    
