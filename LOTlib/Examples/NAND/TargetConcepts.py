# ------------------------------------------------------------------
# Some concepts to try to learn
# ------------------------------------------------------------------

from LOTlib.Primitives.Logic import *
from LOTlib.Primitives.Features import *
TargetConcepts = [lambda x: and_(is_shape_(x,'square'), is_color_(x,'blue')),
                  lambda x: or_(is_shape_(x,'triangle'), is_color_(x,'green')),
                  lambda x: or_(is_shape_(x,'square'), is_color_(x,'red')),
                  lambda x: and_(not_(is_shape_(x,'rectangle')), is_color_(x,'red')),
                  lambda x: and_(not_(is_shape_(x,'square')), not_(is_color_(x,'blue'))),
                  lambda x: and_(is_shape_(x,'rectangle'), is_color_(x,'green')),
                  lambda x: or_(not_(is_shape_(x,'triangle')), is_color_(x,'red')) ]
