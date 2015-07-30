from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str
from LOTlib.Evaluation.Eval import register_primitive


def get_Grammar(s):
    return {
        'An': AnGrammar(),
        'AnBn': AnBnGrammar(),
        'AnB2n': AnBnGrammar(),
        'Dyck': DyckGrammar(),
        'AnCstarBn': AnCstarBnGrammar(),
        'AnBnCn': AnBnCnGrammar()
    }[s]


def AnGrammar():
    register_primitive(flatten2str)

    TERMINAL_WEIGHT = 2.
    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
    grammar.add_rule('BOOL', 'flip_', [''], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'recurse_', [], 1.)
    grammar.add_rule('LIST', '[]', None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('a'), None, TERMINAL_WEIGHT)

    return grammar


def AnBnGrammar():
    register_primitive(flatten2str)

    TERMINAL_WEIGHT = 2.
    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
    grammar.add_rule('BOOL', 'flip_', [''], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'recurse_', [], 1.)
    grammar.add_rule('LIST', '[]', None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('a'), None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('b'), None, TERMINAL_WEIGHT)

    return grammar


def DyckGrammar():
    register_primitive(flatten2str)

    TERMINAL_WEIGHT = 2.
    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
    grammar.add_rule('BOOL', 'flip_', [''], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'recurse_', [], 1.)
    grammar.add_rule('LIST', '[]', None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('('), None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q(')'), None, TERMINAL_WEIGHT)

    return grammar


def AnCstarBnGrammar():
    register_primitive(flatten2str)

    TERMINAL_WEIGHT = 2.
    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
    grammar.add_rule('BOOL', 'flip_', [''], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'recurse_', [], 1.)
    grammar.add_rule('LIST', '[]', None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('a'), None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('b'), None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('c'), None, TERMINAL_WEIGHT)

    return grammar


def AnBnCnGrammar():
    register_primitive(flatten2str)

    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 0.09)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 0.56)
    grammar.add_rule('BOOL', 'flip_', [''], 0.43)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 0.203)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 0.15)
    grammar.add_rule('LIST', 'car_', ['LIST'], 0.15)
    grammar.add_rule('LIST', '\'\'', None, 0.23)
    grammar.add_rule('ATOM', q('a'), None, .33)
    grammar.add_rule('ATOM', q('b'), None, .33)
    grammar.add_rule('ATOM', q('c'), None, .33)

    return grammar