from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str
from LOTlib.Evaluation.Eval import register_primitive


def get_Grammar(s):
    register_primitive(flatten2str)

    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 0.09)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 0.56)
    grammar.add_rule('BOOL', 'flip_', [''], 0.43)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 0.203)
    grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 0.203)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 0.15)
    grammar.add_rule('LIST', 'car_', ['LIST'], 0.15)
    if s != 'AnBnCn':
        grammar.add_rule('LIST', 'recurse_', [], .203)
    grammar.add_rule('LIST', '\'\'', None, 0.23)
    grammar.add_rule('ATOM', q('a'), None, .33)
    grammar.add_rule('ATOM', q('b'), None, .33)
    grammar.add_rule('ATOM', q('c'), None, .33)

    return grammar