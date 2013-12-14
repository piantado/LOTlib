
"""
	This generates random scheme code with cons, cdr, and car, and evaluates it on some simple list structures. 
	No inference here--just random sampling from a grammar.
"""



example_input = [   [], [[]], [ [], [] ], [[[]]]  ]

## Generate some and print out unique ones
seen = set()
for i in xrange(10000):
	x = G.generate('START')
	
	if x not in seen:
		seen.add(x)
		print x.log_probability(), x
		for ei in example_input:
			print "\t", ei, " -> ", x(*ei)




