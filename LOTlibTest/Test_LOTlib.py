# Test_LOTlib.py
# Runs tests associated with particular LOTlib functions

import unittest

# runs all tests in GrammarTest.py
def test_grammar(res):
	# import the GrammarTest package
	from LOTlibTest import GrammarTest
	# run the tests
	GrammarTest.suite().run(res)
	# make sure all the tests passed
	passed = res.wasSuccessful()
	if passed:
		print "All tests in GrammarTest.py passed!"
	else:
		print "Some tests in GrammarTest.py failed..."

	return passed


# executes all tests within the LOTlibTest package
def test_all():
	res = unittest.TestResult()
	# run all the tests
	test_grammar(res)
	passed = res.wasSuccessful()
	print res, passed
	return passed


if __name__ == '__main__':
	test_all()
