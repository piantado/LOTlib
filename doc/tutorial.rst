Introduction to LOTlib
======================


Welcome to LOTlib! LOTlib stands for "Language of Thought Library", and can be used for modeling the way people think about the world. To get started, please follow the instructions below for downloading and installing LOTlib.

Installation:
0. Install LOTlib dependencies
	- numpy
	- scipy
	- simpleMPI (optional)
	- pyparsing (optional)
	- cachetools (optional)
1. Clone the repository from GitHub:
	.. code:: sh

		$ cd path/to/where/you/want/LOTlib/to/go
		$ git clone https://github.com/piantado/LOTlib.git
2. Add the LOTlib directory to your path. If you're using bash, you can do
	.. code:: sh
	
		$ cd ~
		$ echo export PYTHONPATH=\$PYTHONPATH:/path/to/LOTlib >> .bashrc
3. If everything goes well, you should be able to do:
	.. code:: sh
	
		$ python
	.. code:: python

		>>> import LOTlib

Once you've successfully installed LOTlib, it's time to go through some examples to see LOTlib's power. I suggest starting with a simple demo that shows how to set up a PCFG_ with LOTlib, and how to evaluate it. Please see 


.. _PCFG: http://en.wikipedia.org/wiki/Stochastic_context-free_grammar