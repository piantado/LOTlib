"""
# John showed Bill to himself. 
 

"""

from LOTlib import lot_iter
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import *
from LOTlib.DataAndObjects import *
from LOTlib.Parsing import parseScheme,list2FunctionNode

import re
from collections import defaultdict

from BindingTheoryLexicon import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a grammar for tree relations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

grammar = Grammar()

SYNTAX_TYPES = ["S", "NP", "VP", "PP", "DET", "POS", "CC", "N", "SUB"]

grammar.add_rule('START', 'True', None, 1.0)
grammar.add_rule('START', 'False', None, 1.0)
grammar.add_rule('START', '', ['BOOL'], 1.0)
grammar.add_rule('START', '', ['QEXPR'], 1.0)

grammar.add_rule('QEXPR', 'exists_', ['F', 'TREE-LIST'], 1.0) # quantify only over everything *else* in the whole tree
grammar.add_rule('QEXPR', 'forall_', ['F', 'TREE-LIST'], 1.0)
grammar.add_rule('F', 'lambda', ['BOOL'], 10.0, bv_type='TREE', bv_args=None) # bvtype means we introduce a bound variable below
grammar.add_rule('F', 'lambda', ['QEXPR'], 1.0, bv_type='TREE', bv_args=None) # bvtype means we introduce a bound variable below

grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.0)

grammar.add_rule('BOOL', 'co_refers_', ['TREE', 'TREE'], 1.0)
grammar.add_rule('BOOL', 'is_nonterminal_type_', ['TREE', 'SYNTAX-TYPE'], 1.0) # check different types

grammar.add_rule('BOOL', 'empty_', ['TREE-LIST'], 1.0)

grammar.add_rule('BOOL', 'tree_is_', ['TREE', 'TREE'], 1.0)
grammar.add_rule('BOOL', 'dominates_', ['TREE', 'TREE'], 1.0) ## Maybe we deal with list-ops, so we ignore these
grammar.add_rule('BOOL', 'sisters_', ['TREE', 'TREE'], 1.0)

grammar.add_rule('TREE-LIST', 'ancestors_',   ['TREE'], 1.0) # parents of parents, etc. ## TODO: MAKE THIS TRANSITIVE
grammar.add_rule('TREE-LIST', 'descendants_', ['TREE'], 1.0) # children of children, etc.
grammar.add_rule('TREE-LIST', 'children_',    ['TREE'], 1.0)
grammar.add_rule('TREE', 'tree_up_',    ['TREE'], 1.0) # Same as "parent"

grammar.add_rule('TREE-LIST', 'filter_',      ['F', 'TREE-LIST'], 1.0)
grammar.add_rule('TREE-LIST', 'rest_',      ['TREE-LIST'], 1.0)
grammar.add_rule('TREE', 'first_',      ['TREE-LIST'], 1.0)

grammar.add_rule('TREE', 'T', None, 1.0)
grammar.add_rule('TREE', 'x', None, 10.0)

for t in SYNTAX_TYPES:
    grammar.add_rule('SYNTAX-TYPE', q(t), None, 1.0) # check different types

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# How we create a hypothesis in the lexicon
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# How we make a hypothesis inside the lexicon
def make_hypothesis(v=None):
    return LOTHypothesis(grammar, value=v, args=['T', 'x'])



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Convert string data into FunctionNodes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_data(strs):
    """
        Parse treebank-style trees into FunctionNodes and return a list of FunctionData with the right format

        The data is in the following format:

            di.args = T
            di.output = []

        where we the data function "output" is implicit in T (depending on where we choose pronouns)
        SO: All the fanciness is handled in the likelihood
    """
    return map(lambda s: FunctionData(input=[list2FunctionNode(parseScheme(s))], output=None), strs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The data for the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
    Data must be of the form where the co-reference is annotated on the Syntactic category, NOT the terminal (which is a leaf, not a FunctionNode)


    John.1 said that he.1 was happy.

"""

data = make_data(["(S (NP.1 Jim) (VP (V saw) (NP.2 Lyndon)))", \
          "(S (NP.1 Jim) (VP (V saw) (NP.2 he/him)))", \
          "(S (NP.1 Jim) (VP (V saw) (NP.1 himself)))", \
          "(S (NP.1 he/him) (VP (V saw) (NP.2 Lyndon)))", \

          # Behavior in embedded clauses
          "(S (NP.1 Frank) (VP (V believed)  (CC that) (S (NP.2 Joe) (VP (V tickled) (NP.2 himself)))))", \
          "(S (NP.1 Frank) (VP (V believed)  (CC that) (S (NP.2 Joe) (VP (V tickled) (NP.1 he/him)))))", \
          "(S (NP.1 Frank) (VP (V believed)  (CC that) (S (NP.2 Joe) (VP (V tickled) (NP.3 he/him)))))", \
          "(S (NP.1 he/him) (VP (V believed) (CC that) (S (NP.2 Joe) (VP (V tickled) (NP.3 Lyndon)))))", \

          # Need not be dominated by an NP
          "(S (NP.1 Jim) (VP (V saw) (NP.2 (NP.3 Frank) (CONJ and) (NP.1 himself))))", \
          "(S (NP.1 Jim) (VP (V saw) (NP.2 (NP.3 Frank) (CONJ and) (NP.4 he/him))))", \
          "(S (NP.1 Jim) (VP (V found) (NP.2 (DET a) (PP (N picture) of (NP.1 himself)))))", \
          "(S (NP.1 Jim) (VP (V found) (NP.2 (DET a) (PP (N picture) of (NP.3 he/him)))))", \
          "(S (NP.1 he/him) (VP (V saw) (NP.2 (NP.3 Frank) (CONJ and) (NP.4 Lyndon))))", \

          # How to interact with possessives / structured NPs
          "(S (NP.1 (NP.2 Joe) (POS -s) (NP.3 father)) (VP (V believed) (NP.1 himself)))", \
          "(S (NP.1 (NP.2 Joe) (POS -s) (NP.3 father)) (VP (V believed) (NP.2 he/him)))", \
          "(S (NP.1 (NP.2 Joe) (POS -s) (NP.3 father)) (VP (V believed) (NP.4 he/him)))", \
          "(S (NP.1 (NP.2 he/him) (POS -s) (NP.3 father)) (VP (V believed) (NP.4 Lyndon)))", \

          ## Bounding nodes -- sentences
          "(S (S (NP.1 Jim) (VP laughed)) (CC and) (S (NP.2 Larry) (VP (V met) (NP.2 himself))))", \
          "(S (S (NP.1 Jim) (VP laughed)) (CC and) (S (NP.2 Larry) (VP (V met) (NP.1 he/him))))", \
          "(S (S (NP.1 Jim) (VP laughed)) (CC and) (S (NP.2 Larry) (VP (V met) (NP.3 he/him))))", \
          "(S (S (NP.1 he/him) (VP laughed)) (CC and) (S (NP.2 Larry) (VP (V met) (NP.3 Lyndon))))", \
                  
          # can have an NP as a parent
          #"(S (NP.1 Jim) (VP (V saw) (NP (NP.2 Bill) and (NP.3 he/him))))", \
          #"(S (NP (NP.1 Bill) and (NP.2 he/him)) (VP (V saw) (NP.3 Jim)))", \

          # Some examples in more complex syntax:
          "(S (NP.1 (NP.2 Joe) (POS -s) (NP.3 father)) (VP (V believed) (CC that) (S (NP.4 Frank) (VP (V tickled) (NP.1 he/him)))))", \
          "(S (NP.1 (NP.2 Joe) (POS -s) (NP.3 father)) (VP (V believed) (CC that) (S (NP.4 Frank) (VP (V tickled) (NP.2 he/him)))))", \
          "(S (NP.1 (NP.2 Joe) (POS -s) (NP.3 father)) (VP (V believed) (CC that) (S (NP.4 Frank) (VP (V tickled) (NP.5 he/him)))))", \
          "(S (NP.1 (NP.2 Joe) (POS -s) (NP.3 father)) (VP (V believed) (CC that) (S (NP.4 Frank) (VP (V tickled) (NP.4 himself)))))", \
          "(S (NP.1 (NP.2 Joe) (POS -s) (NP.3 father)) (VP (V believed) (CC that) (S (NP.2 he/him) (VP (V tickled) (NP.4 Lyndon)))))", \
          "(S (NP.1 (NP.2 Joe) (POS -s) (NP.3 father)) (VP (V believed) (CC that) (S (NP.1 he/him) (VP (V tickled) (NP.4 Lyndon)))))", \

          "(S (NP.1 (NP.2 he/him) (POS -s) (NP.3 father)) (VP (V believed) (CC that) (S (NP.2 Frank) (VP (V tickled) (NP.4 Lyndon)))))", \
          "(S (NP.1 (NP.2 he/him) (POS -s) (NP.3 father)) (VP (V believed) (CC that) (S (NP.4 Frank) (VP (V tickled) (NP.5 Lyndon)))))",

         ## Not sure what to do with this guy:
          #"(S (NP.1 he/him) (VP (V found) (NP.2 (DET a) (PP (N picture) of (NP.1 himself)))))", \
                 
                  ]) # USED TO MULTIPLY DATA BEFORE, BUT NOW WE SET LL_WEIGHT
                   ##"(S (NP.1 Jim) (VP (V found) (NP.2 (N.3 Sam) (POS -s) (PP (N picture) of (NP.1 he/him)))))", \ ## HMM How do you get this one?
                   
#for di in data: print di

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The target lexicon (or something like it)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# We need to import these in order to use the primitives below
from LOTlib.Evaluation.Eval import *

target = BindingTheoryLexicon(make_hypothesis, words=('<UNCHANGED>', 'himself', 'he/him'))

target.force_function('<UNCHANGED>', lambda T, x: True)

target.force_function('himself', lambda T, x: exists_(lambda y0: and_(co_refers_(x,y0), dominates_(tree_up_(T,y0),x)), descendants_(first_dominating_(T,x, "S"))))

target.force_function('he/him', lambda T,x: and_( is_nonterminal_type_(x,"NP"), not_exists_(lambda y0: co_refers_(x,y0), descendants_(first_dominating_(T,x, "S")))))



