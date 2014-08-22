
"""
        This class is a wrapper for representing "rules" in the grammar.
"""
# TODO: One day we will change "nt" to returntype to match with the FunctionNodes

from FunctionNode import FunctionNode
from copy import copy
from LOTlib.Miscellaneous import None2Empty

class GrammarRule(object):
    def __init__(self, nt, name, to, p=1.0, resample_p=1.0, bv_prefix=None):
        """
                *nt* - the nonterminal

                *name* - the name of this function

                *to* - what you expand to (usually a FunctionNode).

                *p* - unnormalized probability of expansion
                
                *bv_previx* may be needed for GrammarRules introduced *by* BVGrammarRules, so that when we display them we can map to bv_prefix+depth

                Examples:
                A rule where "expansion" is a nonempty list is a real expansion:
                        GrammarRule( "EXPR", "plus", ["EXPR", "EXPR"], ...) -> plus(EXPR,EXPR)
                A rule where "expansion" is [] is a thunk
                        GrammarRule( "EXPR", "plus", [], ...) -> plus()
                A rule where "expansion" is [] is a real terminal (non-thunk)
                        GrammarRule( "EXPR", "five", None, ...) -> five
                A rule where "name" is '' expands without parens:
                        GrammarRule( "EXPR", '', "SUBEXPR", ...) -> EXPR->SUBEXPR

                NOTE: The rule id (rid) is very important -- it's what we use expansion determine equality
        """
        p = float(p)  # make sure these are floats
        
        self.__dict__.update(locals())
        
        for a in None2Empty(to):
            assert isinstance(a,str)
        
        if name == '':
            assert (to is None) or (len(to) == 1), "*** GrammarRules with empty names must have only 1 argument"

    def __repr__(self):
        return str(self.nt) + " -> " + self.name + (str(self.to) if self.to is not None else '') + " w/ p=" +str(self.p)+ ", resample_p=" + str(self.resample_p) 

    def __eq__(self, other):
        """
            Equality is determined through "is" so that we can remove a rule from lists via list.remove
        """
        return (self is other)
        

    def __ne__(self, other):
        return not self.__eq__(other)

    def make_FunctionNodeStub(self, grammar, gp):
        # NOTE: It is VERY important to copy to, or else we end up wtih garbage
        return FunctionNode(returntype=self.nt, name=self.name, args=copy(self.to), generation_probability=gp, added_rule=None)




from uuid import uuid4

class BVGrammarRule(GrammarRule):
    """
        A kind of GrammarRule that supports introducing BVs. This gives a little type checking so that we don't call this with the wrong rules
        
    """
    def __init__(self, nt, name, to, p=1.0, resample_p=1.0, bv_type=None, bv_args=None, bv_prefix="y", bv_p=None):
        """
                *nt* - the nonterminal

                *name* - the name of this function

                *to* - what you expand to (usually a FunctionNode).

                *rid* - the rule id number

                *p* - unnormalized probability of expansion

                *resample_p* - the probability of choosing this node in an expansion

                *bv_type* - return type of the introduced bound variable

                *bv_args* - what are the args when we use a bv (None is terminals, else a type signature)

        """
        p = float(p)  # make sure these are floats

        self.__dict__.update(locals())
        
        # If we use this, we should have BV
        assert bv_type is not None, "Did you mean to use a GrammarRule instead of a BVGrammarRule?"
    
    def __repr__(self):
        return str(self.nt) + " -> " + self.name + (str(self.to) if self.to is not None else '') + " w/ p=" +str(self.p)+ ", resample_p=" + str(self.resample_p) + "BV:"+ str(self.bv_type)+";"+str(self.bv_args)+";"+self.bv_prefix+""
    
    def make_bv_rule(self, grammar):
        """
            Construct the rule that I introduce at a given depth. 
           
            NOTE: This is a GrammarRule and NOT a BVGrammarRule because the introduced rules should *not* themselves introduce rules!
            NOTE: This is a little awkward because it must look back in grammar, but I don't see how to avoid that
        """
        bvp = self.bv_p
        if bvp is None:
            bvp = grammar.BV_P

        return GrammarRule(self.bv_type, uuid4().hex, self.bv_args, p=bvp, resample_p=grammar.BV_RESAMPLE_P, bv_prefix=self.bv_prefix)
   
    def make_FunctionNodeStub(self, grammar, gp):
        """
            Return a FunctionNode with none of the arguments realized. That's a "stub"
            
            *d* -- the current depth
            *gp* -- the generation probability 
        """
        
        # The None's in the next line need to get set elsewhere, since they will depend on the depth and other rules
        # NOTE: It is VERY important to copy to, or else we end up wtih garbage
        return  FunctionNode(returntype=self.nt, name=self.name, args=copy(self.to), generation_probability=gp, added_rule=self.make_bv_rule(grammar) )




        
    
    
        