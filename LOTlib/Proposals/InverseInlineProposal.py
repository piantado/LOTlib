
from LOTProposal import LOTProposal

class InverseInlineProposal(LOTProposal):
    """
            Inverse inlinling for non-functions

            TODO: HOW TO DEAL WITH ENSURING THE LAMBDA RULES ARE IN THE GRAMMAr, HAVE THE RIGHT PROBS, ETC

            TODO: NOT QUITE WORKING RIGHT -- 2014 JUL 25 -- BV NAMES ARE WRONG

    """

    def __init__(self, grammar):
        """
                This takes a grammar and computes which lambdas are allowed in the grammar
        """
        self.__dict__.update(locals())
        LOTProposal.__init__(self, grammar)

        # Figure out which of what can be extracted
        canlambda = set()
        for rule in grammar:
            if rule.name == 'lambda':
                assert len(rule.to) == 1
                canlambda.add(  (rule.nt, rule.to[0], rule.bv_type) )

        # check to see if we can insert on each given nonterminal type
        canapply = set()
        for rule in grammar:
            if rule.name == 'apply_':
                assert len(rule.to) == 2
                canapply.add( (rule.to, rule.to[0], rule.to[1]) )

        # check to see which we can apply on
        self.can_apply = set()
        for nt in grammar.rules.keys():


























































    def is_extractable(self, n, a):
        """
                Inside of n, is a extractable?

                take a node n and a subnode a, and check if a can be extracted from n, according to the lambdas
                present in the grammar and

                NOTE: Does not actually check that a is a child of n
        """

        # First check that a is extractable via the grammar, meaning that we can insert
        # using a lambda that expands to the same type
        # NOTE: we could in principle have an apply and a lambda that gave the right returntype, but this is more complex to implement

        # Well, n is extractable if






























        # Finally, we must check that this doesn't contain any bound variables of outer lambdas
        introduced_bvs = set() # the bvs that are introduced below n (and are thus okay)
        for ai in a:
            if ai.ruleid < 0 and ai.name not in introduced_bvs: # If it's a bv
                return False
            elif ai.islambda() and ai.bv_name is not None:
                introduced_bvs.add(ai.bv_name)

        return True


    def propose_tree(self, t):
        """
                Delete:
                        - find an apply
                        - take the interior of the lambdathunk and sub it in for the lambdaarg everywhere, remove the apply
                Insert:
                        - Find a node
                        - Find a subnode s
                        - Remove all repetitions of s, create a lambda thunk
                        - and add an apply with the appropriate machinery
        """

        newt = copy(t)
        f,b = 0.0, 0.0
        success = False #acts to tell us if we found and replaced anything



        def is_apply(x):
            return (x.name == 'apply_') and (len(x.args)==2) and x.args[0].islambda() and not x.args[1].islambda()

        # ------------------
        if random() < 0.5: #INSERT MOVE

            # sample a node
            for ni, di, resample_p, Z in self.grammar.sample_node_via_iterate(newt):


            # Sample a subnode -- NOTE: we must use copy(ni) here since we modify this tree, and so all hell breaks loose otherwise
                for a, adi, aresample_p, sZ in self.grammar.sample_node_via_iterate(copy(ni), predicate=lambda a: self.is_extractable(ni,a)):
                    success = True


                    print self.is_extractable(ni, a)

                    """

                    below = copy(ni)
                    varname = 'Y'+str(di+1)

                    # replace with the variables
                    # TODO: FIX THE RID HERE -- HOW DO WE TREAT IT?
                    below.replace_subnodes(a, FunctionNode(a.returntype, varname, None, ruleid=-999))

                    # create a new node, the lambda abstraction
                    fn = FunctionNode(below.returntype, 'apply_', [ \
                            FunctionNode('LAMBDAARG', 'lambda', [ below ], bv_prefix='Y', bv_name=varname, bv_type=a.returntype, bv_args=[] ),\
                            s
                            #FunctionNode('LAMBDATHUNK',  'lambda', [ s  ], bv_name=None, bv_type=None, bv_args=None)\
                                    ] )

                    # Now convert into a lambda abstraction
                    ni.setto(fn)

                    f += (log(resample_p) - log(Z)) + (log(aresample_p) - log(sZ))

                    """
        """
        else: # DELETE MOVE

                resample_p = None
                for ni, di, resample_p, Z in self.grammar.sample_node_via_iterate(newt, predicate=is_apply):
                        success = True

                        ## what does the variable look like? Here a thunk with bv_name
                        var = FunctionNode( ni.args[0].bv_type , ni.args[0].bv_name, None)

                        assert len(ni.args) == 2
                        assert len(ni.args[0].args) == 1

                        newni = ni.args[0].args[0] # may be able to optimize away?

                        ## and remove
                        newni.replace_subnodes(var, ni.args[1])

                        ##print ":", newni
                        ni.setto(newni)
                        f += (log(resample_p) - log(Z))

                if resample_p is None: return [newt,0.0]

                #newZ = self.grammar.resample_normalizer(newt, predicate=is_replacetype)
                ##to go back, must choose the
                #b += log(resample_p) - log(newZ)
        """

    #       newt.fix_bound_variables()
    #       newt.reset_function() # make sure we update the function
        if not success:
            return [copy(t),0.0]
        else:
            return [newt, f-b]

if __name__ == "__main__":
    from LOTlib.Examples.Number.Shared import generate_data, grammar,  make_h0, NumberExpression

    grammar.add_rule('WORD', 'apply_', ['LWORD', 'WORD'], 0.1)
    grammar.add_rule('LWORD', 'lambda', ['WORD'], 0.1, bv_type='WORD' )

    p = InverseInlineProposal(grammar)

    for _ in xrange(1000):
        t = grammar.generate()
        print "\n\n", t
        #for _ in xrange(10):
        #       t =  p.propose_tree(t)[0]
        #       print "\t", t
