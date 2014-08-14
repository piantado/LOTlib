"""
        A finite English grammar that's used to test functions in FunctionNode
"""

from LOTlib.Grammar import Grammar

g = Grammar()

g.add_rule('START','S',['NP', 'VP'],1)
g.add_rule('NP','',['the boy'],1)
g.add_rule('NP','',['the ball'],1)
g.add_rule('VP','',['ate the dog'],1)
g.add_rule('VP','',['ate the chicken'],1)


def log_probability(tree):
    return 0 # TODO: stub

if __name__ == "__main__":
    for i in xrange(100):
        print(g.generate())
