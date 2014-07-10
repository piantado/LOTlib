
from LOTlib.Miscellaneous import unique
from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

G = Grammar()

G.add_rule('START', '', ['NT1'], 1.0)

G.add_rule('NT1', 'A', [], 1.00)
G.add_rule('NT1', 'B', ['NT2'], 2.00) 
G.add_rule('NT1', 'C', ['NT3', 'NT3'], 3.70) 

G.add_rule('NT2', 'X', None, 1.0)

G.add_rule('NT3', 'Y', None, 1.0)
G.add_rule('NT3', 'Z', None, 1.25)



for i in xrange(100):
	t = G.generate()
	tstr = str(t)
	print t.log_probability(), tstr

	