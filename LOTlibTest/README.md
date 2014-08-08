LOTlibTest
------

LOTlibTest is a python 2.x library for testing the LOTlib library, with the ultimate goal of providing a complete set of tests for LOTlib. It is heavily under development as of early August 2014. 



REQUIREMENTS
------------

- LOTlib
	- numpy
	- scipy
	- SimpleMPI (optional, if using MPI; requires mpi4py)
	- pyparsing (optional, for parsing)
	- cachetools (optional, for memoization)
- unittest

INSTALLATION
------------

This library should be included with LOTlib. If you do not have LOTlib installed, please follow the instructions for installing LOTlib in LOTlib/README.md.

RUNNING
-------

To run all tests, execute
        $ cd LOTlibTest
        $ python LOTlibTest.py

To run tests for a specific file (e.g. GrammarTest.py), execute
        $ cd LOTlibTest
        $ python GrammarTest.py

TESTS
--------

- Test_LOTlib: Runs all tests described below, and prints information about passes/failures
- DataAndObjectsTest (not started): tests functions in DataAndObjects.py
- FiniteBestSetTest: tests that the FiniteBestSet priority queue correctly orders its elements
- FunctionNodeTest: tests that the .pystring() method correctly stringifies FunctionNodes
- GrammarRuleTest (not started): tests functions in GrammarRule.py 
- GrammarTest:
	- tests that the probabilities returned by lp_regenerate_propose_to() correspond to the frequencies of proposals returned by RegenerationProposal.propose_tree()
	- tests that the log_probability() function correctly computes log probabilities for finite testing grammars in LOTlibTest/Grammars/, with and without bound variables
	- tests that the log_probability() function correctly computes log probabilities under proposals returned by RegenerationProposal.propose_tree(), for finite testing grammars in LOTlibTest/Grammars/, with and without bound variables
- MiscellaneousTest: tests functions in Miscellaneous.py
- ProposalsTest (not started): tests functions in Proposals.py
- SubtreesTest (not started): tests functions in Subtrees.py
- Obsolete/: obsolete testing functions, may or may not be incorporated into the main LOTlibTest/ functions in the future

Testing files in other subdirectories have not been started.

