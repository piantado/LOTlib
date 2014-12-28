"""
class to test MetropolisHastings.py
follows the standards in https://docs.python.org/2/library/unittest.html
"""


from math import exp
from TreeTesters import FiniteTreeTester # defines check_tree

class GrammarTest(FiniteTreeTester):

    def test_log_probability(self):
        """
        Let's just make sure that our prior sums to 1.0 (finite case)
        """
        self.assertAlmostEquals(sum([exp(t.log_probability()) for t in self.trees]), 1.0)

    def test_enumerate_at_depth(self):
        for d in xrange(6):
            for t in self.grammar.enumerate_at_depth(d):
                self.assertTrue(t.depth() == d)

    def test_generate(self):
        """
        Test generating from the grammar -- that the trees have the same log_probability
        as enumerating
        """
        nrules = self.grammar.nrules()
        for _ in xrange(1000):
            t = self.grammar.generate()

            self.check_tree(t)

        # And make sure all rules are removed
        self.assertEquals(self.grammar.nrules(), nrules)


if __name__ == '__main__':
    import unittest

    unittest.main()
