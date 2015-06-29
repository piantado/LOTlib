"""
        Parse simplified lambda expressions--enough for ATIS.

        TODO: No quoting etc. supported yet.

        NOTE: This is based on http://pyparsing.wikispaces.com/file/view/sexpParser.py/278417548/sexpParser.py
"""


from pyparsing import Suppress, alphanums, Group, Forward, ZeroOrMore, Word
from LOTlib.FunctionNode import FunctionNode, BVAddFunctionNode
from LOTlib.Miscellaneous import unlist_singleton

#####################################################################
## Here we define a super simple grammar for lambdas

LPAR, RPAR = map(Suppress, "()")
token = Word(alphanums + "-./_:*+=!<>$")

sexp = Forward()
sexpList = Group(LPAR + ZeroOrMore(sexp) + RPAR)
sexp << ( token | sexpList )

#####################################################################

def parseScheme(s):
    """
            Return a list of list of lists... for a string containing a simple lambda expression (no quoting, etc)
            Keeps uppercase and lowercase letters exactly the same

            e.g. parseScheme("(S (K (S I)) (S (K K) I) x y)") --> ['S', ['K', ['S', 'I']], ['S', ['K', 'K'], 'I'], 'x', 'y']
    """

    x = sexp.parseString(s, parseAll=True)
    return unlist_singleton( x.asList() ) # get back as a list rather than a pyparsing.ParseResults


def list2FunctionNode(l, style="atis"):
    """
            Takes a list *l* of lambda arguments and maps it to a function node.

            The *style* of lambda arguments could be "atis", "scheme", etc.
    """

    if isinstance(l, list):

        fn = None
        if len(l) == 0: return None
        elif style is 'atis':
            rec = lambda x: list2FunctionNode(x, style=style) # a wrapper to my recursive self
            if l[0] == 'lambda':
                fn = BVAddFunctionNode(None, 'FUNCTION', 'lambda', [rec(l[3])], bv_type=l[1], bv_args=None ) # TOOD: HMM WHAT IS THE BV?
            else:
                fn = FunctionNode(None, l[0], l[0], map(rec, l[1:]))
        elif style is 'scheme':
            raise NotImplementedError

        # set the parents
        for a in fn.argFunctionNodes():
            a.parent=fn

        return fn

    else: # for non-list
        return l


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
if __name__ == '__main__':

    test1 = "(defun factorial (x) (if (= x 0) 1 (* x (factorial (- x 1)))))"
    test2 = "(lambda $0 e (and (day_arrival $0 thursday:da) (to $0 baltimore:ci) (< (arrival_time $0) 900:ti) (during_day $0 morning:pd) (exists $1 (and (airport $1) (from $0 $1)))))"

    y = parseScheme(test1)
    #print y
    assert str(y) == str(['defun', 'factorial', ['x'], ['if', ['=', 'x', '0'], '1', ['*', 'x', ['factorial', ['-', 'x', '1']]]]])

    x = parseScheme(test2)
    #print x
    assert str(x) == str(['lambda', '$0', 'e', ['and', ['day_arrival', '$0', 'thursday:da'],
            ['to', '$0', 'baltimore:ci'], ['<', ['arrival_time', '$0'], '900:ti'],
            ['during_day', '$0', 'morning:pd'], ['exists', '$1', ['and', ['airport', '$1'], ['from', '$0', '$1']]]]])

    print list2FunctionNode(x)
