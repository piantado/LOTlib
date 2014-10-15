from LOTlib.DataAndObjects import *
from LOTlib.Evaluation.Primitives.Logic import *
from LOTlib.Evaluation.Primitives.Features import *

SHAPES = ['square', 'triangle', 'rectangle']
COLORS = ['blue', 'red', 'green']

# ------------------------------------------------------------------
# Some concepts to try to learn
# ------------------------------------------------------------------

TARGET_CONCEPTS = [lambda x: and_(is_shape_(x,'square'), is_color_(x,'blue')),
            lambda x: or_(is_shape_(x,'triangle'), is_color_(x,'green')),
            lambda x: or_(is_shape_(x,'square'), is_color_(x,'red')),
            lambda x: and_(not_(is_shape_(x,'rectangle')), is_color_(x,'red')),
            lambda x: and_(not_(is_shape_(x,'square')), not_(is_color_(x,'blue'))),
            lambda x: and_(is_shape_(x,'rectangle'), is_color_(x,'green')),
            lambda x: or_(not_(is_shape_(x,'triangle')), is_color_(x,'red')) ]


# ------------------------------------------------------------------
# Set up the objects
# ------------------------------------------------------------------

all_objects = make_all_objects( shape=SHAPES, color=COLORS )

