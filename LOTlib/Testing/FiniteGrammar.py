
## TODO: Vary resample_p to make sure that works here!


from LOTlib.Grammar import Grammar

grammar = Grammar()

grammar.add_rule('START', '', ['A0'], 1.0)

grammar.add_rule('A0', 'A0_', ['A1', 'a'], 0.1)
grammar.add_rule('A0', 'A0_', ['b'], 0.6)

grammar.add_rule('A1', 'A1_', ['A2', 'c'], 0.2)
grammar.add_rule('A1', 'A1_', ['d'], 0.8)

grammar.add_rule('A2', 'A2_', ['e'], 1.0)

# NOTE: we give these different names so we can't derive identical trees multiple ways
# (by, e.g., not using a lambda)
grammar.add_rule('A0', 'apply_', ['L0', 'A1'], 0.20)
grammar.add_rule('L0', 'lambda', ['B0'], 0.11, bv_p=0.07, bv_type='B1')

grammar.add_rule('A0', 'apply2_', ['L1', 'A1'], 0.20)
grammar.add_rule('L1', 'lambda', ['B0'], 0.12, bv_p=0.08, bv_type='B1', bv_args=['B2'])

grammar.add_rule('A0', 'apply3_', ['L2', 'A1'], 0.20)
grammar.add_rule('L2', 'lambda', ['B0'], 0.13, bv_p=0.09, bv_type='B1', bv_args=[])

grammar.add_rule('B0', 'B0_', ['f', 'B1'], 0.89)
grammar.add_rule('B0', 'B0_', ['g'],       0.77)

grammar.add_rule('B1', 'B1_', ['h', 'B2'], 0.84)
grammar.add_rule('B1', 'B1_', ['i'],       0.16)

grammar.add_rule('B2', 'B2_', ['o'], 1.0)


if __name__ == "__main__":

    for t in grammar.enumerate():
        print t