from GrammarRule import GrammarRule

class BVRuleContextManager(object):
    
    def __init__(self, grammar, *rules):
        """
            This manages rules that we add and subtract in the context of grammar generation. This is a class that is somewhat
            inbetween Grammar and GrammarRule. It manages creating, adding, and subtracting the bound variable rule via "with" clause in Grammar.
            
            NOTE: The "rule" here is the added rule, not the "bound variable" one (that adds the rule)
            NOTE: If rule is None, then nothing happens
            
            This actually could go in FunctionNode, *except* that it needs to know the grammar, which FunctionNodes do not
        """
        self.__dict__.update(locals())
                
    def __str__(self):
        return "<Managing context for %s>"%str(self.rule)
    
    
    def __enter__(self):
        """
            Here, we construct the bound variable rule if any and then remove it later
        """
        #print "# The rules:\t", self.rules
        
        for r in self.rules:
            if r is not None:
                assert isinstance(r, GrammarRule), r
                self.grammar.rules[r.nt].append(r)
    
    def __exit__(self, t, value, traceback):
        for r in self.rules:
            if r is not None:
                self.grammar.rules[r.nt].remove(r)
        
        return False #re-raise exceptions
        
    