# *- coding: utf-8 -*-
try:                import numpy as np
except ImportError: import numpypy as np

from copy import copy
from collections import defaultdict

import LOTlib
from LOTlib.Miscellaneous import *
from LOTlib.FunctionNode import isFunctionNode
from LOTlib.GrammarRule import GrammarRule, BVAddGrammarRule, BVUseGrammarRule
from LOTlib.Hypotheses.Hypothesis import Hypothesis
from LOTlib.BVRuleContextManager import BVRuleContextManager



class Grammar:
    """
            A PCFG-ish class that can handle special types of rules:
                    - Rules that introduce bound variables
                    - Rules that sample from a continuous distribution
                    - Variable resampling probabilities among the rules

                    TODO: Implement this:
                    - This has a few special values of returntype, which are expected on the rhs of rules, and then generate continuous values
                            - \*uniform\* - when the name is this, the value printed and evaled is a uniform random sample of [0,1) (which can be resampled via the PCFG)
                            - \*normal\*  - 0 mean, 1 sd
                            - \*exponential\* - rate = 1

            NOTE: Bound variables have a rule id < 0

            This class fixes a bunch of problems that were in earlier versions, such as
    """

    def __init__(self, BV_P=10.0, BV_RESAMPLE_P=1.0, start='START'):
        self.__dict__.update(locals())

        self.rules = defaultdict(list) # A dict from nonterminals to lists of GrammarRules
        self.rule_count = 0
        self.bv_count = 0 # How many rules in the grammar introduce bound variables?
        self.bv_rule_id = 0 # A unique idenifier for each bv rule id (may get quite large)      . The actual stored rule are negative this
    
        
    def is_nonterminal(self, x):
        """ A nonterminal is just something that is a key for self.rules"""

    #       if x is a string       if x is a key
        return isinstance(x, str) and (x in self.rules)


    def display_rules(self):
        """
                Prints all the rules to the console.
        """
        for rule in self:
            print rule

    def __iter__(self):
        """
                Define an iterator so we can say "for rule in grammar..."
        """
        for k in self.rules.keys():
            for r in self.rules[k]:
                yield r

    def nonterminals(self):
        """
                Returns all non-terminals.
        """
        return self.rules.keys()

    def add_rule(self, nt, name, to, p, resample_p=1.0, bv_type=None, bv_args=None, bv_prefix='y', bv_p=None):
        """
                Adds a rule and returns the added rule.

                *nt* - The Nonterminal. e.g. S in "S -> NP VP"

                *name* - The name of this function. NOTE: If you are introducing a bound variable, the name of this function must reflect that it is a lambda node! Currently, the only way to do this is to name it 'lambda'.

                *to* - What you expand to (usually a FunctionNode).

                *p* - Unnormalized probability of expansion

                *resample_p* - In resampling, what is the probability of choosing this node?

                *bv_type* - What bound variable was introduced

                *bv_args* - What are the args when we use a bv (None is terminals, else a type signature)

        """
        self.rule_count += 1
        
        if bv_type is not None:
            
            # Check the name
            assert (name.lower() == 'lambda' or name.lower() == 'applylambda'), "When introducing bound variables, the name of the expanded function must be 'lambda'."
            
            newrule = BVAddGrammarRule(nt,name,to, p=p, resample_p=resample_p, bv_type=bv_type, bv_args=bv_args, bv_prefix=bv_prefix, bv_p=bv_p)
            
        else:
            newrule = GrammarRule(nt,name,to, p=p, resample_p=resample_p)
            
        # actually add it
        self.rules[nt].append(newrule)
        return newrule
    
    def is_terminal_rule(self, r):
        """
            Check if a rule is "terminal" meaning that it doesn't contain any nonterminals in its expansion
        """ 
        return not any([self.is_nonterminal(a) for a in None2Empty(r.to)])  
    
    
        
    ############################################################
    ## Generation
    ############################################################


    def generate(self, x=None):
        """
                Generate from the PCFG -- default is to start from x - either a
                nonterminal or a FunctionNode.

                Returns a FunctionNode.

        """
        #print "# Calling grammar.generate", d, type(x), x

        # Decide what to start from based on the default if start is not specified
        if x is None:
            x = self.start
            assert self.start in self.rules, "The default start symbol %s is not a defined nonterminal"%self.start

        # Dispatch different kinds of generation
        if isinstance(x,list):            
            # If we get a list, just map along it to generate. We don't count lists as depth--only FunctionNodes
            return map(lambda xi: self.generate(x=xi), x)
        elif self.is_nonterminal(x):

            # sample a grammar rule
            r, gp = weighted_sample(self.rules[x], probs=lambda x: x.p, return_probability=True, log=False)
            #print "SAMPLED:", gp, r, type(r)

            # Make a stub for this functionNode 
            fn = r.make_FunctionNodeStub(self, gp) ## NOT SURE WHY BU TCOPY IS NECESSARY HERE

            # Define a new context that is the grammar with the rule added. Then, when we exit, it's still right 
            with BVRuleContextManager(self, fn.added_rule): # not sure why I can't use with/as:
                if fn.args is not None:  # Can't recurse on None or else we genreate from self.start
                    fn.args = self.generate(fn.args)  # and generate below *in* this context (e.g. with the new rules added)

            return fn

        else:  # must be a terminal
            assert isinstance(x, str), ("*** Terminal must be a string! x="+x)
            return x


    def iterate_subnodes(self, t, d=0, predicate=lambdaTrue, do_bv=True, yield_depth=False):
        """
                Iterate through all subnodes of node *t*, while updating the added rules (bound variables)
                so that at each subnode, the grammar is accurate to what it was.

                if *do_bv*=False, we don't do bound variables (useful for things like counting nodes, instead of having to update the grammar)

                *yield_depth*: if True, we return (node, depth) instead of node
                *predicate*: filter only the ones that match this

                NOTE: if you DON'T iterate all the way through, you end up acculmulating bv rules
                so NEVER stop this iteration in the middle!
                TODO: Make this more elegant -- use BVCM
        """

        if isFunctionNode(t):
            #  print "iterate subnode: ", t, t.added_rule

            if predicate(t):
                yield (t,d) if yield_depth else t

            #Define a new context that is the grammar with the rule added. Then, when we exit, it's still right 
            with BVRuleContextManager(self, t.added_rule):                    

                if t.args is not None:
                    for g in self.iterate_subnodes(t.args, d=d+1, do_bv=do_bv, yield_depth=yield_depth, predicate=predicate): # pass up anything from below
                        yield g

        elif isinstance(t, list):
            for a in t:
                for g in self.iterate_subnodes(a, d=d, do_bv=do_bv, yield_depth=yield_depth, predicate=predicate):
                    yield g

    def resample_normalizer(self, t, predicate=lambdaTrue):
        """
                Returns the sum of all of the non-normalized probabilities.
        """
        Z = 0.0
        for ti in self.iterate_subnodes(t, do_bv=True, predicate=predicate):
            Z += ti.resample_p
        return Z


    def sample_node_via_iterate(self, t, predicate=lambdaTrue, do_bv=True):
        """
                This will yield a random node in the middle of its iteration, allowing us to expand bound variables properly
                (e.g. when the node is yielded, the state of the grammar is correct)
                It also yields the probability and the depth

                So use this via

                for ni, di, resample_p, resample_Z, in sample_node_via_iterate():
                        ... do something

                and it should only execute once, despite the "for"
                The "for" is nice so that it will yield back and clean up the bv

        """

        Z = self.resample_normalizer(t, predicate=predicate) # the total probability
        r = random() * Z # now select a random number (giving a random node)
        sm = 0.0
        foundit = False

        for ni, di in self.iterate_subnodes(t, predicate=predicate, do_bv=do_bv, yield_depth=True):
            sm += ni.resample_p
            if sm >= r and not foundit: # our node
                foundit=True
                yield [ni, di, ni.resample_p, Z]
        
        
    def depth_to_terminal(self, x, openset=None, current_d=None):
        """
            Return a dictionary that maps both this grammar's rules and its nonterminals to a number, giving how quickly we
            can go from that nonterminal or rule to a terminal.

            *openset* -- stores the set of things we're currently trying to compute for. We must skip rules that contain anything in there, since
            they have to be defined still, and so we want to avoid a loop.
        """
        
        if current_d is None: 
            current_d = dict()
            
        if openset is None:
            openset = set()
            
        openset.add(x)
        
        if isinstance(x, GrammarRule):
            if x.to is None or len(x.to) == 0:
                current_d[x] = 0 # we are a terminal
            else:
                current_d[x] = 1+max([ (self.depth_to_terminal(a, openset=openset, current_d=current_d) if a not in openset else 0) for a in x.to])
        elif isinstance(x, str):
            if x not in self.rules:
                current_d[x] = 0 # a terminal
            else:
                current_d[x] = min([(self.depth_to_terminal(r, openset=openset, current_d=current_d) if r not in openset else Infinity) for r in self.rules[x] ])
        else:
            assert False, "Shouldn't get here!"
        
        openset.remove(x)
        
        return current_d[x]
        

    def increment_tree(self, x=None, depth=0, max_depth=Infinity):
        """
            A wrapper to increment_tree to pre-compute some info on depths and handle corner cases
        """
        if x is None:
            x = self.start
        
        # Now really call increment_tree, using depthdict to sort
        for y in LOTlib.lot_iter(self.increment_tree_(x, depth=depth, max_depth=max_depth, depthdict=dict())):
            yield y
            
    
    def increment_tree_(self, x=None, depth=0, max_depth=Infinity, depthdict=None):
        """
                A lazy version of tree enumeration. Here, we generate all trees, starting from a rule or a nonterminal symbol and going up to max_depth

                This is constant memory and should produce each tree *once* (However: if a grammar has multiple derivations of the same
                str(tree), then you will see repeats!). 
                
                TODO: CHANGE THIS TO ENUMERATE SHALLOW->DEEP
                
                *x*: A node in the tree
                *depth*: Depth of the tree
                *depthdict* : memoizes depth_to_terminal so that we can order rules in order to make enumeration small->large
        """
        # wrap no specification for x
        
        if depth >= max_depth:
            raise StopIteration

        if isFunctionNode(x):
            # NOTE: WE don't need to handle BV here since they are handled below when we use the rule
            
            original_x = copy(x)
            
            # go all odometer on the kids below::
            iters = [self.increment_tree_(x=y,depth=depth,max_depth=max_depth, depthdict=depthdict) if self.is_nonterminal(y) else None for y in x.args]
            if len(iters) == 0:
                yield copy(x)
            else:
                #print "HERE", iters
                for i in xrange(len(iters)):
                    if iters[i] is not None:
                        x.args[i] = iters[i].next()
                
                # the index of the last terminal symbol (may not be len(iters)-1),
                last_terminal_idx = max( [i if iters[i] is not None else -1 for i in xrange(len(iters))] )

                ## Now loop through the args, counting them up
                while True:
                    
                    yield copy(x) # yield the initial tree, and then each successive tree

                    # and then process each carry:
                    for carry_pos in xrange(len(iters)): # index into which tree we are incrementing
                        if iters[carry_pos] is not None: # we are not a terminal symbol (mixed in)
                            
                            ## NOTE: This *MUST* go here in order to prevent adding a rule and then not removing it when you carry (thus introducing a bv of a1 into a2)
                            with BVRuleContextManager(self, x.added_rule):
                                
                                try:
                                    x.args[carry_pos] = iters[carry_pos].next()
                                    break # if we increment successfully, no carry, so break out of i loop
                                except StopIteration: # if so, then "carry"
                                    if carry_pos == last_terminal_idx:
                                        raise StopIteration
                                    elif iters[carry_pos] is not None:
                                        # reset the incrementer since we just carried
                                        iters[carry_pos] = self.increment_tree_(x=original_x.args[carry_pos],depth=depth,max_depth=max_depth, depthdict=depthdict)
                                        x.args[carry_pos] = iters[carry_pos].next() # reset this
                                        # and just continue your loop over i (which processes the carry)

        elif self.is_nonterminal(x): # just a single nonterminal
          
            ## TODO: somewhat inefficient since we do this each time:
            ## Here we change the order of rules to be terminals *first*
            terminals = []
            nonterminals = []
            for k in self.rules[x]:
                if not self.is_terminal_rule(k):      #AAH this used to be called "x" and that ruined the scope of the outer "x"
                    nonterminals.append(k)
                else:                       
                    terminals.append(k)
            
            # sort by probability, so high probability trees *tend* to come first
            terminals    = sorted(terminals,    key=lambda r: self.depth_to_terminal(r, current_d=depthdict) )
            nonterminals = sorted(nonterminals, key=lambda r: self.depth_to_terminal(r, current_d=depthdict) )
            Z = logsumexp([ log(r.p) for r in self.rules[x]] ) # normalizer
            
            #print terminals
            #print nonterminals
            #print "---------------------------------------"
            
            # yield each of the rules that lead to terminals -- always do this since depth>=0 (above)
            for r in terminals:
                fn = r.make_FunctionNodeStub(self, (log(r.p) - Z))
                # Do not need to set added_rule since they can't exist here
                yield fn
                
            if depth < max_depth: # if we can go deeper
                for r in nonterminals:#expand each nonterminals
                    fn = r.make_FunctionNodeStub(self, (log(r.p) - Z))
                    
                    for q in self.increment_tree_(x=fn, depth=depth+1,max_depth=max_depth, depthdict=depthdict):
                        yield q
            else:
                yield x


    def lp_regenerate_propose_to(self, x, y, xZ=None, yZ=None):
        """
                Returns a log probability of starting at x and ending up at y from a regeneration move.
                Any node is a candidate if the trees are identical except for what's below those nodes
                (although what's below *can* be identical!)

                NOTE: This does NOT take into account insert/delete
                NOTE: Not so simple because we must count multiple paths
        """

        # TODO: Can we remove yZ?
        # TODO: TEST THIS:

        # Wrap for hypotheses instead of trees
        if isinstance(x, Hypothesis):
            assert isinstance(y, Hypothesis), ("*** If x is a hypothesis, y must be! "+str(y) )
            return self.lp_regenerate_propose_to(x.value,y.value,xZ=xZ, yZ=yZ)

        RP = -Infinity

        if (isinstance(x,str) and isinstance(y,str) and x==y) or (x.returntype == y.returntype):

            # compute the normalizer
            if xZ is None: xZ = self.resample_normalizer(x)
            if yZ is None: yZ = self.resample_normalizer(y)

            # Well we could select x's root to go to Y
            RP = logplusexp(RP, log(x.resample_p) - log(xZ) + y.log_probability() )

            if x.name == y.name:

                # how many kids are not equal, and where was the last?
                mismatch_count, mismatch_index = 0, 0

                # print 'args are', x.args

                if x.args is not None:
                    for i, xa, ya in zip(xrange(len(x.args)), x.args if x.args is not None else [],
                                                              y.args if y.args is not None else []):
                        if xa != ya: # checks whole subtree!
                            mismatch_count += 1
                            mismatch_index = i

                if mismatch_count > 1: # We have to have only selected x,y to regenerate
                    pass
                elif mismatch_count == 1: # we could propose to x, or x.args[mismatch_index], but nothing else (nothing else will fix the mismatch)
                    RP = logplusexp(RP, self.lp_regenerate_propose_to(x.args[mismatch_index], y.args[mismatch_index], xZ=xZ, yZ=yZ))
                else: # identical trees -- we could propose to any, so that's just the tree probability below convolved with the resample p
                    for xi in dropfirst(x): # we already counted ourself
                        RP = logplusexp(RP, log(xi.resample_p) - log(xZ) + xi.log_probability() )

        return RP


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == "__main__":
    pass
    #from LOTlib.Examples.FOL.FOL import grammar
    #from LOTlib.Examples.Magnetism.SimpleMagnetism import grammar
    #from LOTlib.Examples.Number.Shared import grammar
    from LOTlibTest.Grammars.FiniteWithBVArgs import g as grammar
    
    #for t in grammar.increment_tree(max_depth=9):
    #    print t.depth(), t
        #t.fullprint()
        #print "\n\n"
     
    #for r in grammar.rules.keys():
    #    print ">>", grammar.depth_to_terminal(r), r
    """
    for _ in xrange(1000):
        t = grammar.generate()
        print type(t), t
        
        from LOTlib.Proposals.RegenerationProposal import RegenerationProposal
        rp = RegenerationProposal(grammar)
        x = rp.propose_tree(t)[0]
        print type(x), x
        
        print "\n\n"
    """
    
     
    #    print t.depth(), t

"""
    #from LOTlib.Examples.RationalRules.Shared import grammar
    #from LOTlib.Examples.Number.Shared import grammar
    #from LOTlib.DefaultGrammars import SimpleBoolean as grammar
    SimpleBoolean= Grammar()
    SimpleBoolean.add_rule('START', 'False', None, 1)
    SimpleBoolean.add_rule('START', 'True', None, 1)
    SimpleBoolean.add_rule('START', '', ['BOOL'], 1.0)
    
    SimpleBoolean.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
    SimpleBoolean.add_rule('BOOL', 'not_', ['BOOL'], 1.0)
    
    SimpleBoolean.add_rule('BOOL', '', ['PREDICATE'], 5)
     
    for t in SimpleBoolean.increment_tree(x='BOOL', depth=6):
        print t
"""


"""
        #AB_GRAMMAR = PCFG()
        #AB_GRAMMAR.add_rule('START', '', ['EXPR'], 1.0)
        #AB_GRAMMAR.add_rule('EXPR', '', ['A', 'EXPR'], 1.0)
        #AB_GRAMMAR.add_rule('EXPR', '', ['B', 'EXPR'], 1.0)

        #AB_GRAMMAR.add_rule('EXPR', '', ['A'], 1.0)
        #AB_GRAMMAR.add_rule('EXPR', '', ['B'], 1.0)

        #for i in xrange(1000):
                #x = AB_GRAMMAR.generate('START')
                #print x.log_probability(), x


        grammar = Grammar()
        grammar.add_rule('START', '', ['EXPR'], 1.0)
        #grammar.add_rule('EXPR', 'somefunc_', ['EXPR', 'EXPR', 'EXPR'], 1.0, resample_p=5.0)
        grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 4.0, resample_p=10.0)
        grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 3.0, resample_p=5.0)
        grammar.add_rule('EXPR', 'divide_', ['EXPR', 'EXPR'], 2.0)
        grammar.add_rule('EXPR', 'subtract_', ['EXPR', 'EXPR'], 1.0)
        grammar.add_rule('EXPR', 'x', [], 15.0) # these terminals should have None for their function type; the literals
        grammar.add_rule('EXPR', '1.0', [], 3.0)
        grammar.add_rule('EXPR', '13.0', [], 2.0)

        ## We generate a few ways, mapping strings to the actual things
        #print "Testing increment (no lambda)"
        #TEST_INC = dict()

        #for t in grammar.increment_tree('START',3):
                #TEST_INC[str(t)] = t

        #print "Testing generate (no lambda)"
        TEST_GEN = dict()
        TARGET = dict()
        from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
        for i in xrange(10000):
                t = grammar.generate('START')
                # print ">>", t, ' ', dir(t)
                TEST_GEN[str(t)] = t

                if t.count_nodes() < 10:
                        TARGET[LOTHypothesis(grammar, value=copy(t) )] = t.log_probability()
                        # print out log probability and tree
                        print t, ' ', t.log_probability()




        #print "Testing MCMC (no counts) (no lambda)"
        #TEST_MCMC = dict()
        #MCMC_STEPS = 10000
        #import LOTlib.MetropolisHastings
        #from LOTlib.Hypothesis import LOTHypothesis
        #hyp = LOTHypothesis(grammar)
        #for x in LOTlib.MetropolisHastings.mh_sample(hyp, [], MCMC_STEPS):
                ##print ">>", x
                #TEST_MCMC[str(x.value)] = copy(x.value)

        ### Now print out the results and see what's up
        #for x in TEST_GEN.values():

                ## We'll only check values that appear in all
                #if str(x) not in TEST_MCMC or str(x) not in TEST_INC: continue

                ## If we print
                #print TEST_INC[str(x)].log_probability(),  TEST_GEN[str(x)].log_probability(),  TEST_MCMC[str(x)].log_probability(), x

                #assert abs( TEST_INC[str(x)].log_probability() - TEST_GEN[str(x)].log_probability()) < 1e-9
                #assert abs( TEST_INC[str(x)].log_probability() -  TEST_MCMC[str(x)].log_probability()) < 1e-9


        ## # # # # # # # # # # # # # # #
        ### And now do a version with bound variables
        ## # # # # # # # # # # # # # # #

        grammar.add_rule('EXPR', 'apply', ['FUNCTION', 'EXPR'], 5.0)
        grammar.add_rule('FUNCTION', 'lambda', ['EXPR'], 1.0, bv_type='EXPR', bv_args=None) # bvtype means we introduce a bound variable below

        print "Testing generate (lambda)"
        TEST_GEN = dict()
        for i in xrange(10000):
                x = grammar.generate('START')
                TEST_GEN[str(x)] = x
                #print x
                #x.fullprint()

        print "Testing MCMC (lambda)"
        TEST_MCMC = dict()
        TEST_MCMC_COUNT = defaultdict(int)
        MCMC_STEPS = 50000
        import LOTlib.MetropolisHastings
        from LOTlib.Hypothesis import LOTHypothesis
        hyp = LOTHypothesis(grammar)
        for x in LOTlib.MetropolisHastings.mh_sample(hyp, [], MCMC_STEPS):
                TEST_MCMC[str(x.value)] = copy(x.value)
                TEST_MCMC_COUNT[str(x.value)] += 1 # keep track of these
                #print x
                #for kk in grammar.iterate_subnodes(x.value, do_bv=True, yield_depth=True):
                        #print ">>\t", kk
                #print "\n"
                #x.value.fullprint()

        # Now print out the results and see what's up
        for x in TEST_GEN.values():

                # We'll only check values that appear in all
                if str(x) not in TEST_MCMC: continue

                #print TEST_GEN[str(x)].log_probability(),  TEST_MCMC[str(x)].log_probability(), x

                if abs( TEST_GEN[str(x)].log_probability() - TEST_MCMC[str(x)].log_probability()) > 1e-9:
                        print "----------------------------------------------------------------"
                        print "--- Mismatch in tree probabilities:                          ---"
                        print "----------------------------------------------------------------"
                        TEST_GEN[str(x)].fullprint()
                        print "----------------------------------------------------------------"
                        TEST_MCMC[str(x)].fullprint()
                        print "----------------------------------------------------------------"

                assert abs( TEST_GEN[str(x)].log_probability() - TEST_MCMC[str(x)].log_probability()) < 1e-9

        # Now check that the MCMC actually visits the nodes the right number of time
        keys = [x for x in TEST_MCMC.keys() if TEST_MCMC[x].count_nodes() <= 8 ] # get a set of common trees
        cntZ = log(sum([ TEST_MCMC_COUNT[x] for x in keys]))
        lpZ  = logsumexp([ TEST_MCMC[x].log_probability() for x in keys])
        for x in sorted(keys, key=lambda x: TEST_MCMC[x].log_probability()):
                #x.fullprint()
                #print "))", x
                print TEST_MCMC_COUNT[x], log(TEST_MCMC_COUNT[x])-cntZ, TEST_MCMC[x].log_probability() - lpZ,  TEST_MCMC[x].log_probability(), q(TEST_MCMC[x]), hasattr(x, 'my_log_probability')


                ## TODO: ADD ASSERTIONS ETC


        # To check the computation of lp_regenerate_propose_to, which should return how likely we are to propose
        # from one tree to another
        #from LOTlib.Examples.Number.Shared import *
        #x = NumberExpression(grammar).value
        #NN = 100000
        #d = defaultdict(int)
        #for i in xrange(NN):
                #y,_ = grammar.propose(x, insert_delete_probability=0.0)
                #d[y] += 1
        ## Show counts and expected counts
        #for y in sorted( d.keys(), key=lambda z: d[z]):
                #EC = round(exp(grammar.lp_regenerate_propose_to(x,y))*NN)
                #if d[y] > 10 or EC > 10: # print only highish prob things
                        #print d[y], EC, y
        #print ">>", x








        # If success....
        print "---------------------------"
        print ":-) No complaints here!"
        print "---------------------------"


















        ## We will use three four methods to generate a set of trees. Each of these should give the same probabilities:
        # - increment_tree('START')
        # - generate('START')
        # - MCMC with proposals (via counts)
        # - MCMC with proposals (via probabilities of found trees)



        #for i in xrange(1000):
                #x = ARITHMETIC_GRAMMAR.generate('START')

                #print x.log_probability(), ARITHMETIC_GRAMMAR.RR_prior(x), x
                #for xi in ARITHMETIC_GRAMMAR.iterate_subnodes(x):
                        #print "\t", xi
"""
