#from LOTlib.Grammar import Gramma
import LOTlib
from LOTlib.DataAndObjects import *
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import *


SHAPES = ['square', 'triangle', 'rectangle']
COLORS = ['blue', 'red', 'green']

# ------------------------------------------------------------------
# Some concepts to try to learn
# ------------------------------------------------------------------

TARGET_CONCEPTS = [lambda x: and_(is_shape_(x,'square'), is_color_(x,'blue')), \
	    lambda x: or_(is_shape_(x,'triangle'), is_color_(x,'green')), \
	    lambda x: or_(is_shape_(x,'square'), is_color_(x,'red')), \
	    lambda x: and_(not_(is_shape_(x,'rectangle')), is_color_(x,'red')), \
	    lambda x: and_(not_(is_shape_(x,'square')), not_(is_color_(x,'blue'))), \
	    lambda x: and_(is_shape_(x,'rectangle'), is_color_(x,'green')), \
	    lambda x: or_(not_(is_shape_(x,'triangle')), is_color_(x,'red')) ]

# ------------------------------------------------------------------
# Set up the grammar
# Here, we create our own instead of using DefaultGrammars.Nand because
# we don't want a BOOL/PREDICATE distinction
# ------------------------------------------------------------------
FEATURE_WEIGHT = 2. # Probability of expanding to a terminal

G = Grammar()

G.add_rule('START', '', ['BOOL'], 1.0)

G.add_rule('BOOL', 'nand_', ['BOOL', 'BOOL'], 1.0/3.)
G.add_rule('BOOL', 'nand_', ['True', 'BOOL'], 1.0/3.)
G.add_rule('BOOL', 'nand_', ['False', 'BOOL'], 1.0/3.)

# And finally, add the primitives
for s in SHAPES: G.add_rule('BOOL', 'is_shape_', ['x', q(s)], FEATURE_WEIGHT)
for c in COLORS: G.add_rule('BOOL', 'is_color_', ['x', q(c)], FEATURE_WEIGHT)

# ------------------------------------------------------------------
# Set up the objects
# ------------------------------------------------------------------

all_objects = make_all_objects( shape=SHAPES, color=COLORS )

# ------------------------------------------------------------------
# Generator for data
# ------------------------------------------------------------------

def generate_data(N, f):
	data = [] 
	for _ in xrange(N):
		o = sample_one(all_objects)
		data.append( FunctionData(input=[o], output=f(o) ) )
	return data