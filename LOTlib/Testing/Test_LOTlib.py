# Test_LOTlib.py
# Runs tests associated with particular LOTlib functions

import unittest
import importlib # http://stackoverflow.com/questions/8718885/import-module-from-string-variable

# out all tests in a testing file
def test_module(module):
    res = unittest.TestResult()
    # import the package package
    f = importlib.import_module(module)
    # NOTE: the testing file MUST have a method named suite(), otherwise we'll skip this test
    # TODO: find a more robust way of testing whether a given module has a suite() method
    if 'suite' not in dir(f):
        print "LOTlibTest." + module + " must have a suite() function, skipping tests for LOTlibTest." + module + "..."
        return
    # run the tests
    f.suite().run(res)
    # make sure all the tests passed
    passed = res.wasSuccessful()
    if passed:
        print "All tests in LOTlibTest." + module + " passed!"
    else:
        print "Some tests in LOTlibTest." + module + " failed..."

    return res


# executes all tests within the LOTlibTest package
def test_all():
    # a list of test modules we should execute
    tests = ['FunctionNodeTest','GrammarTest','SubtreesTest','FiniteBestSetTest','MiscellaneousTest','ProposalsTest'
            # ,'ParsingTest' # we need the pyparsing module to do this test
    ]
    # a list of test results, wrapped in unittest.TestResult objects
    results = []
    # specifies whether all tests have passed
    passed = True
    # run all the tests
    for test in tests:
        res = test_module(test)
        results.append(res)
        passed = passed and res.wasSuccessful()
    if passed:
        # all tests passed
        print "All specified tests in LOTlibTest passed! Assuming the tests cover all LOTlib code, LOTlib is ready to ship!!!"
    else:
        # some tests failed
        print "Some tests in LOTlibTest failed. Assuming the tests are valid, something is wrong with the LOTlib code..."
    return passed


if __name__ == '__main__':
    test_all()
