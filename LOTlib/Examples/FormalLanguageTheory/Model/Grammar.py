from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str
from LOTlib.Evaluation.Eval import register_primitive


def get_Grammar(s):
    register_primitive(flatten2str)

    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 0.01)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 0.50)
    grammar.add_rule('BOOL', 'flip_', [''], 0.49)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 0.092)
    grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 0.039)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 0.254)
    grammar.add_rule('LIST', 'car_', ['LIST'], 0.253)
    grammar.add_rule('LIST', '\'\'', None, 0.091)

    if s != 'AnBnCn' and s != 'SimpleEnglish':
        grammar.add_rule('LIST', 'recurse_', [], .260)

    if s != 'SimpleEnglish':
        grammar.add_rule('ATOM', q('a'), None, .33)
        grammar.add_rule('ATOM', q('b'), None, .33)
        grammar.add_rule('ATOM', q('c'), None, .33)
    else:
        grammar.add_rule('ATOM', q('D'), None, .156)
        grammar.add_rule('ATOM', q('A'), None, .186)
        grammar.add_rule('ATOM', q('N'), None, .183)
        grammar.add_rule('ATOM', q('P'), None, .312)
        grammar.add_rule('ATOM', q('V'), None, .164)

    return grammar