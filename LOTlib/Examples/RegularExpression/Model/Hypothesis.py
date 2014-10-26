from LOTlib.FunctionNode import isFunctionNode
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
import re


class RegexHypothesis(LOTHypothesis):
    """Define a special hypothesis for regular expressions.

    This requires overwritting value2function to use our custom interpretation model on trees -- not just
    simple eval.

    Note:
        This doesn't require any basic_primitives -- the grammar node names are used by to_regex too

    """
    def value2function(self, v):
        regex = to_regex(v)
        c = re.compile(regex)
        return (lambda s: (c.match(s) is not None))

    def __str__(self):
        return to_regex(self.value)


def to_regex(fn):
    """Map a tree to a regular expression.

    Custom mapping from a function node to a regular expression string (like, e.g. "(ab)*(c|d)" )
    """
    assert isFunctionNode(fn)

    if fn.name == 'star_':         return '(%s)*'% to_regex(fn.args[0])
    elif fn.name == 'plus_':       return '(%s)+'% to_regex(fn.args[0])
    elif fn.name == 'question_':   return '(%s)?'% to_regex(fn.args[0])
    elif fn.name == 'or_':         return '(%s|%s)'% tuple(map(to_regex, fn.args))
    elif fn.name == 'str_append_': return '%s%s'% (fn.args[0], to_regex(fn.args[1]))
    elif fn.name == 'terminal_':   return '%s'%fn.args[0]
    elif fn.name == '':            return to_regex(fn.args[0])
    else:
        assert False, fn
