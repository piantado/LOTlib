
## TODO: Vary resample_p to make sure that works here!


from LOTlib.Grammar import Grammar

grammar = Grammar()

grammar.add_rule('START', '', ['A'], 1.0)

grammar.add_rule('A', 'A', ['A', 'A'], 0.2)
grammar.add_rule('A', 'A', ['a'], 0.7)

grammar.add_rule('A', 'apply_', ['L', 'A'], 0.10)
grammar.add_rule('L', 'lambda', ['A'], 0.11, bv_p=0.07, bv_type='A')

grammar.add_rule('A', 'apply_', ['LF', 'A'], 0.10)
grammar.add_rule('LF', 'lambda', ['A'], 0.11, bv_p=0.07, bv_type='A', bv_args=['A'], bv_prefix='F')

## NOTE: DOES NTO HANDLE THE CASE WITH TWO A->APPLY, L->LAMBDAS

if __name__ == "__main__":
    from LOTlib import lot_iter

    for t in lot_iter(grammar.enumerate()):
        print t