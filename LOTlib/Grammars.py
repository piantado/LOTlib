"""
	A bunch of standard grammars. 
	
	NOTE: These do not have terminal expansions, since those will vary by program...
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Now define the grammars:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grammar_SimpleBoolean = PCFG()
Grammar_SimpleBoolean.add_rule('START', 'False', [], 1.0)
Grammar_SimpleBoolean.add_rule('START', 'True', [], 1.0)
Grammar_SimpleBoolean.add_rule('START', '', ['BOOL'], 1.0)

Grammar_SimpleBoolean.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
Grammar_SimpleBoolean.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
Grammar_SimpleBoolean.add_rule('BOOL', 'not_', ['BOOL'], 1.0)

## ~ ~ ~ ~ ~ ~ ~ 

Grammar_TF = PCFG()
Grammar_TF.add_rule('START', 'False', [], 1.0)
Grammar_TF.add_rule('START', 'True', [], 1.0)

## ~ ~ ~ ~ ~ ~ ~ 

Grammar_NAND = PCFG()
Grammar_NAND.add_rule('START', '', ['BOOL'], 1.0)
Grammar_NAND.add_rule('BOOL', 'nand_', ['BOOL', 'BOOL'], 1.0)
Grammar_NAND.add_rule('BOOL', 'True', [], 1.0)
Grammar_NAND.add_rule('BOOL', 'False', [], 1.0)

## ~ ~ ~ ~ ~ ~ ~ 

Grammar_CNF = PCFG()
Grammar_CNF.add_rule('START', '', ['CONJ'], 1.0)
Grammar_CNF.add_rule('START', 'True', [], 1.0)
Grammar_CNF.add_rule('START', 'False', [], 1.0)
Grammar_CNF.add_rule('CONJ', '',     ['DISJ'], 1.0)
Grammar_CNF.add_rule('CONJ', '',     ['PREDICATE'], 1.0)
Grammar_CNF.add_rule('CONJ', 'not_', ['PREDICATE'], 1.0)
Grammar_CNF.add_rule('CONJ', 'and_', ['PREDICATE', 'CONJ'], 1.0)

Grammar_CNF.add_rule('DISJ', '',     ['PREDICATE'], 1.0)
Grammar_CNF.add_rule('DISJ', 'not_', ['PREDICATE'], 1.0)
Grammar_CNF.add_rule('DISJ', 'or_',  ['PREDICATE', 'DISJ'], 1.0)

## ~ ~ ~ ~ ~ ~ ~ 

Grammar_DNF = PCFG()
Grammar_DNF.add_rule('START', '', ['DISJ'], 1.0)
Grammar_DNF.add_rule('START', 'True', [], 1.0)
Grammar_DNF.add_rule('START', 'False', [], 1.0)
Grammar_DNF.add_rule('DISJ', '',     ['CONJ'], 1.0)
Grammar_DNF.add_rule('DISJ', '',     ['PREDICATE'], 1.0)
Grammar_DNF.add_rule('DISJ', 'not_', ['PREDICATE'], 1.0)
Grammar_DNF.add_rule('DISJ', 'or_', ['PREDICATE', 'DISJ'], 1.0)

Grammar_DNF.add_rule('CONJ', '',     ['PREDICATE'], 1.0)
Grammar_DNF.add_rule('CONJ', 'not_', ['PREDICATE'], 1.0)
Grammar_DNF.add_rule('CONJ', 'and_',  ['PREDICATE', 'CONJ'], 1.0)
