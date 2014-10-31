from LOTProposal import LOTProposal
from LOTlib.Miscellaneous import sample1, lambdaTrue, dropfirst
from random import random
from math import log
from collections import defaultdict
from LOTlib.GrammarRule import *
from LOTlib.FunctionNode import pystring


def lp_sample_equal_to(n,x, predicate=lambdaTrue):
    """
        What is the log probability of sampling x or something equal to it from n? 
    """
    assert x in n, str(x)+"\t\t"+str(n)
    
    Z = n.sample_node_normalizer(predicate=predicate)
    return log(sum([t.resample_p if (t==x) and predicate(t) else 0.0 for t in n])) - log(Z)

class InverseInlineProposal(LOTProposal):
    """
        Inverse inlinling proposals.
    """
    
    def __init__(self, grammar):
        """
            This takes a grammar and a regex to match variable names
        """
        self.__dict__.update(locals())
        LOTProposal.__init__(self, grammar)
        
        self.insertable_rules = defaultdict(list) # Hash each nonterminal to (a,l) where a and l are the apply and lambda rules you need
        for nt in self.grammar.rules.keys():
            
            for a in filter(lambda r: (r.name=="apply_") and (r.nt == nt), self.grammar):
                for l in filter( lambda r: isinstance(r, BVAddGrammarRule) and (r.nt == a.to[0]) and (r.bv_args is None), self.grammar): # For each lambda whose "below" is the right type. bv_args are not implemented yet
                    self.insertable_rules[nt].append( (a,l) )                   
            
    
    def can_abstract_at(self, x):
        """
            Can I put a lambda at x?
        """
        return len(self.insertable_rules[x.returntype]) > 0
    
    def is_valid_argument(self, n, x):
        """
            The only valid arguments that can be extracted contain lambdas defined above n OR below t
        """
        allowed_bvs = set()
        for t in dropfirst(n.up_to(to=None)):  # don't count n in the bvs we allow!
            if isinstance(t, BVAddFunctionNode):
                allowed_bvs.add(t.added_rule.name)
        
        for t in x:
            # we also allow bvs of things defined in t
            if isinstance(t, BVAddFunctionNode)  and t is not x:
                allowed_bvs.add(t.added_rule.name)
            elif isinstance(t, BVUseFunctionNode) and (t.name not in allowed_bvs):
                return False
        
        return True
                
                
    
    def can_inline_at(self, n):
        """
            Can we inline this? Only if its an apply whose lambda bv takes no args. ALSO, the argument must occur in the lambda, or else this could not have been created via inlining
        """
        
        if n.name != "apply_":
            return False
        
        l, a = n.args
        
        return (l.rule.bv_args is None or len(l.rule.bv_args) == 0) and (a in l.args[0]) and self.is_valid_argument(l.args[0], a) and self.can_abstract_at(l.args[0])
    
        
    def propose_tree(self, t):
        """
            Delete:
                - find an apply
                - take the interior of the lambdathunk and sub it in for the lambdaarg everywhere, remove the apply
            Insert:
                - Find a node
                - Find a subnode s
                - Remove all repetitions of s, create a lambda
                - and add an apply
        """

        newt = copy(t) 
        f,b = 0.0, 0.0
            
        # ------------------
        if random() <0 : # Am inverse-inlining move
        
            # where the lambda goes
            n, np = newt.sample_subnode(self.can_abstract_at)
            if n is None: return [newt, 0.0]
            
            # Pick the rule we will use
            ir = self.insertable_rules[n.returntype]
            ar,lr = sample1(ir) # the apply and lambda rules
            assert ar.nt == n.returntype
            assert lr.nt == ar.to[0]
            
            # what the argument is. Must have a returntype equal to the second apply type
            arg_predicate = lambda z: z.returntype == ar.to[1] and self.is_valid_argument(n, z) #how do we choose args?
            argval, _ = n.sample_subnode(predicate=arg_predicate )
            if argval is None: return [newt, 0.0]
            argval = copy(argval) # necessary since the argval in the tree gets overwritten
            below = copy(n) # necessary since n gets setto the new apply rule  
            
            # now make the function nodes. The generation_probabiltiies will be reset later, as will the parents for applyfn and bvfn
            n.setto(ar.make_FunctionNodeStub(self.grammar, 0.0, None)) # n's parent is preserved
            
            lambdafn = lr.make_FunctionNodeStub(self.grammar, 0.0, n) ## this must be n, not applyfn, since n will eventually be setto applyfn
            bvfn = lambdafn.added_rule.make_FunctionNodeStub(self.grammar, 0.0, None) # this takes the place of argval everywhere below
            below.replace_subnodes(lambda x:x==argval, bvfn) # substitute below the lambda            
            lambdafn.args[0] = below
            
            below.parent = lambdafn
            argval.parent = n
            
            # build our little structure
            n.args = lambdafn, argval
            
            assert self.can_inline_at(n) # this had betterb e true
            #assert newt.check_parent_refs()
            
            # to go forward, you choose a node, a rule, and an argument
            f = np + (-log(len(ir))) + lp_sample_equal_to(n,argval, predicate=arg_predicate)
            newZ = newt.sample_node_normalizer(self.can_inline_at)
            b = (log(n.resample_p) - log(newZ))
            
        else: # An inlining move
            
            
            n, np = newt.sample_subnode(self.can_inline_at)
            if n is None: return [newt, 0.0]
         
            #print "CHOOSING n=", n
            #print "PARENT n=", n.parent
            
            # Replace the subnodes 
            newn = n.args[0].args[0] # what's below the lambda
            argval = n.args[1]
            bvn = n.args[0].added_rule.name # the name of the added variable
            
            newn.replace_subnodes(lambda x: x.name == bvn, argval)
            
            n.setto(newn)            
            assert self.can_abstract_at(n) # this had better be true
            ir = self.insertable_rules[n.returntype] # for the backward probability
            
            # just the probability of choosing this apply
            f = np
            
            # choose n, choose a, choose the rule
            new_nZ = newt.sample_node_normalizer(self.can_abstract_at) # prob of choosing n
            argvalp = lp_sample_equal_to(newn, argval, predicate= lambda z: (z.returntype == argval.returntype) and self.is_valid_argument(newn, z))
            b = (log(newn.resample_p) - log(new_nZ)) + argvalp + (-log(len(ir)))
        
        ## and fix the generation probabilites, because otherwiset hey are ruined by all the mangling above
        self.grammar.recompute_generation_probabilities(newt)
        assert newt.check_parent_refs() # Can comment out -- here for debugging
        
        
        return [newt, f-b]
            
            
    
if __name__ == "__main__":
        from LOTlib import lot_iter
        #from LOTlib.Examples.Magnetism.SimpleMagnetism import data, grammar,  make_h0  DOES NOT WORK FOR BV ARGS
        from LOTlib.Examples.Number.Model.Utilities import grammar, make_h0, generate_data, get_knower_pattern
        
        grammar.add_rule('LAMBDA_WORD', 'lambda', ['WORD'], 1.0, bv_type='WORD')
        grammar.add_rule('WORD', 'apply_', ['LAMBDA_WORD', 'WORD'], 1.0)
        
        p = InverseInlineProposal(grammar)
        
        """
        # Just look at some proposals
        for _ in xrange(200):    
            t = grammar.generate()
            print ">>", t
            #assert t.check_generation_probabilities(grammar)
            #assert t.check_parent_refs()
            
            for _ in xrange(10):
                t =  p.propose_tree(t)[0]
                print "\t", t
            
        """
        # Run MCMC -- more informative about f-b errors    
        from LOTlib.Inference.MetropolisHastings import MHSampler

        from LOTlib.Inference.Proposals.MixtureProposal import MixtureProposal          
        from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
                
        h = make_h0(proposal_function=MixtureProposal([InverseInlineProposal(grammar), RegenerationProposal(grammar)] ))
        data = generate_data(100)
        for h in lot_iter(MHSampler(h, data)):
            print h.posterior_score, h.prior, h.likelihood, get_knower_pattern(h), h
        
            
