"""
class to test FunctionNode.py
follows the standards in https://docs.python.org/2/library/unittest.html
"""

import unittest
from LOTlib.FunctionNode import *
from LOTlib.Grammar import *
from Grammars import FunctionNodeGrammar
from LOTlib import lot_iter
        


class FunctionNodeTest(unittest.TestCase):

    # initialization that happens before each test is carried out
    def setUp(self):
        self.G = FunctionNodeGrammar.g

    # tests the .pystring() method
    def test_str(self):
        t = self.G.generate()
        string = pystring(t)
        # test whether the object is a string
        self.assertEqual(type(string), str)
        self.assertEqual(type(t.__str__()), str)

    def test_eq(self):
        counter = 0
        for i in lot_iter(xrange(10000)):
            x = self.G.generate()
            y = self.G.generate()

            if pystring(x) == pystring(y):
                counter += 1
                # print(counter)
                #print( pystring(x)+'\n'+ pystring(y)+'\n')

            self.assertEqual( pystring(x) == pystring(y), x == y, "Without bvs, the pystrings should be the same")

    def test_refs(self):
        """
            Test the setting of parents and rules
        """
        for _ in lot_iter(xrange(1000)):
            x = self.G.generate()
            self.assertTrue(x.check_parent_refs())
            
    def test_check_generation_probabilities(self):
        """
            Test the generation probabilities
        """
        for _ in lot_iter(xrange(1000)):
            x = self.G.generate()
            self.assertTrue(x.check_generation_probabilities(self.G))
    
    def test_copy(self):
        """
            Test the copy operation
        """
        for _ in xrange(1000):
            x = self.G.generate()
            y = copy(x)
            
            for a,b in zip(x.subnodes(), y.subnodes()):
                self.assertEqual(a,b)
                self.assertTrue(a is not b)
                
    def test_setto(self):
        """
            Test the operation of setting a function node to another. 
        """
        for _ in lot_iter(xrange(1000)):
            x = self.G.generate()
            x0 = copy(x)
            y = self.G.generate()
            y_subnodes = y.subnodes()
            x.setto(y)
            
            self.assertTrue(x.check_parent_refs())
            self.assertTrue(x.check_generation_probabilities(self.G))
            
            for xi in x:
                self.assertTrue(xi in y_subnodes)
            
    #@unittest.skip('Skipping test_substitute')         
    def test_substitute(self):
        """
            Test how substitution works
        """
        for _ in lot_iter(xrange(1000)):
            x = self.G.generate()
            y, _ = x.sample_subnode()
            oldy = copy(y)
            repl = self.G.generate(y.returntype)
            
            # pick a novel replacemnet (must NOT equal y)
            # NOTE: We can't just rejection sample here to pick repl because it may not be possible for a given x
            if repl == y: continue
            
            x.replace_subnodes(lambda z: z==y, repl)
            
            self.assertTrue(x.check_parent_refs())
            # We cannot check generation_probabilites because replace_subnodes breaks that!
            
            # and ensure that y is not left!
            for xi in x:
                self.assertTrue(xi != oldy, "\n%s\n%s\n%s\n%s"%(x,y,repl,xi))
       

    # function that is executed after each test is carried out
    def tearDown(self):
        pass


# A Test Suite composed of all tests in this class
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(FunctionNodeTest)

if __name__ == '__main__':
    unittest.main()
