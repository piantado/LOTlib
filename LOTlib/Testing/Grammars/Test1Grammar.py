"""
        A finite grammar used in the test-1.py file
        TODO: I don't know what this test file is supposed to test!
"""

from LOTlib.Grammar import Grammar


g = Grammar()

g.add_rule('START', '', ['NT1'], 1.0)

g.add_rule('NT1', 'A', [], 1.00)
g.add_rule('NT1', 'B', ['NT2'], 2.00)
g.add_rule('NT1', 'C', ['NT3', 'NT3'], 3.70)

g.add_rule('NT2', 'X', None, 1.0)

g.add_rule('NT3', 'Y', None, 1.0)
g.add_rule('NT3', 'Z', None, 1.25)

def log_probability(tree):
    return 0 # TODO: stub

if __name__ == "__main__":
    for i in xrange(100):
        print(g.generate())
