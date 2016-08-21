
import unittest

from DefaultGrammars import finiteTestGrammar, infiniteTestGrammar
from LOTlib.FunctionNode import FunctionNode, BVUseFunctionNode, BVAddFunctionNode

class EnumerationTest(unittest.TestCase):
    def runTest(self):
        print "# Testing grammar enumeration"
        n = 0
        for t in finiteTestGrammar.enumerate():
            n += 1
        self.assertEqual(n,27)


import re
class GrammarTreeTest(unittest.TestCase):
    def runTest(self):
        for i, grammar in enumerate([finiteTestGrammar, infiniteTestGrammar]):
            print "# Testing grammar ", i

            for _ in xrange(10000):
                t = grammar.generate()

                # correct overall return type
                self.assertTrue(t.returntype == grammar.start)

                # correct argument types for each subnode
                for ti in t.iterate_subnodes(grammar):
                    r = grammar.get_matching_rule(ti) # find the rule that generated this
                    # print ti, r
                    self.assertTrue(r is not None) # Better have been one!

                    if ti.args is None:
                        self.assertTrue( r.to is None)
                    else:
                        for ri, ai in zip(r.to, ti.args):
                            if isinstance(ai, FunctionNode):
                                self.assertTrue(ai.returntype == ri)
                                # and check parent refs
                                self.assertTrue(ai.parent == ti)
                            else:
                                self.assertTrue(ai == ri)

                # Check that the bv function nodes are of the right type
                # And that we added and removed rules appropriately
                added_rules = [] # just see what we added
                for ti in t.iterate_subnodes(grammar):
                    if re.match(r'bv_', ti.name):
                        self.assertTrue(isinstance(ti, BVUseFunctionNode))

                        r = grammar.get_matching_rule(ti)

                        # NOTE: We cannot use "in" here since that uses rule "is", but we've created
                        # a new thing that is equivalent to the rule. So instead, we check the bv name
                        self.assertTrue(r.name in [r.name for r in grammar.rules[ti.returntype]])
                        added_rules.append(r)

                    if re.match(r'lambda', ti.name):
                        self.assertTrue(isinstance(ti, BVAddFunctionNode))

                        # assert that this rule isn't already there
                        self.assertTrue(ti.added_rule.name not in [r.name for r in grammar.rules[ti.returntype]])

                # Then assert that none of the rules are still in the grammar
                for therule in added_rules:
                    self.assertTrue(therule.name not in [r.name for r in grammar.rules[ti.returntype]])


class FinitePackTest(unittest.TestCase):
    def runTest(self):
        print "# Testing packing (finite)"
        for _ in xrange(5000):
            t = finiteTestGrammar.generate()
            s = finiteTestGrammar.pack_ascii(t)
            t2 = finiteTestGrammar.unpack_ascii(s)

            self.assertTrue(t==t2)


class InfinitePackTest(unittest.TestCase):
    def runTest(self):
        print "# Testing packing (infinite)"
        for _ in xrange(5000):
            t = infiniteTestGrammar.generate()
            s = infiniteTestGrammar.pack_ascii(t)
            t2 = infiniteTestGrammar.unpack_ascii(s)

            self.assertTrue(t==t2)

