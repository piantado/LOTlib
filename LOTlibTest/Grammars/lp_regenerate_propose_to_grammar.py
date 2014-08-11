"""
        A finite English grammar to test the lp_regenerate_propose_to() function
"""

from LOTlib.Grammar import Grammar


g = Grammar()

g.add_rule('START', 'S', ['NP', 'VP'], 0.1)
g.add_rule('START', 'S', ['INTERJECTION'], 0.3)
g.add_rule('NP', 'NV', ['DET', 'N'], 0.6)
g.add_rule('NP', 'NV', ['DET', 'ADJ', 'N'], 0.4)
g.add_rule('NP', 'NV', ['PN'], 0.3)
g.add_rule('VP', 'NV', ['V', 'NP'], 0.5)
g.add_rule('N', 'ball', None, 0.2)
g.add_rule('N', 'computer', None, 0.2)
g.add_rule('N', 'phone', None, 0.2)
g.add_rule('PN', 'Chomsky', None, 0.3)
g.add_rule('PN', 'Samay', None, 0.3)
g.add_rule('PN', 'Steve', None, 0.3)
g.add_rule('PN', 'Hassler', None, 0.3)
g.add_rule('V', 'eats', None, 0.25)
g.add_rule('V', 'kills', None, 0.25)
g.add_rule('V', 'maims', None, 0.25)
g.add_rule('V', 'sees', None, 0.25)
g.add_rule('ADJ', 'peculiar', None, 0.4)
g.add_rule('ADJ', 'strange', None, 0.4)
g.add_rule('ADJ', 'red', None, 0.4)
g.add_rule('ADJ', 'queasy', None, 0.4)
g.add_rule('ADJ', 'happy', None, 0.4)
g.add_rule('DET', 'the', None, 0.5)
g.add_rule('DET', 'a', None, 0.5)
g.add_rule('INTERJECTION', 'sh*t', None, 0.6)
g.add_rule('INTERJECTION', 'fu*k pi', None, 0.6)


def log_probability(tree):
    return 0 # TODO: stub

if __name__ == "__main__":
    for i in xrange(100):
        print(g.generate())
