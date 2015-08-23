from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str
from LOTlib.Evaluation.Eval import register_primitive


def get_Grammar(s, terminals=None):
    """
    terminals: extra terminals can be added into one grammar
    """
    register_primitive(flatten2str)
    # t_set = set()
    #
    grammar = Grammar()
    # grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    # grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 0.01)
    # grammar.add_rule('BOOL', 'empty_', ['LIST'], 0.50)
    # grammar.add_rule('BOOL', 'flip_', [''], 0.49)
    # grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 0.092)
    # grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 0.039)
    # grammar.add_rule('LIST', 'cdr_', ['LIST'], 0.254)
    # grammar.add_rule('LIST', 'car_', ['LIST'], 0.253)
    # grammar.add_rule('LIST', '\'\'', None, 0.091)
    #
    # if s != 'AnBnCn' and s != 'SimpleEnglish':
    #     grammar.add_rule('LIST', 'recurse_', [], .260)
    # if s != 'SimpleEnglish':
    #     grammar.add_rule('ATOM', q('a'), None, .33); t_set.add('a')
    #     if s != 'An':
    #         grammar.add_rule('ATOM', q('b'), None, .33); t_set.add('b')
    #         if s != 'AnBn' and s != 'AnB2n' and s != 'Dyck' and s != 'LongDependency':
    #             grammar.add_rule('ATOM', q('c'), None, .33); t_set.add('c')
    # else:
    #     grammar.add_rule('ATOM', q('D'), None, .156); t_set.add('D')
    #     grammar.add_rule('ATOM', q('A'), None, .186); t_set.add('A')
    #     grammar.add_rule('ATOM', q('N'), None, .183); t_set.add('N')
    #     grammar.add_rule('ATOM', q('P'), None, .312); t_set.add('P')
    #     grammar.add_rule('ATOM', q('V'), None, .164); t_set.add('V')
    #
    # if terminals is not None:
    #     for e in terminals:
    #         if e not in t_set:
    #             grammar.add_rule('ATOM', q(e), None, .33)

    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
    grammar.add_rule('BOOL', 'flip_', [''], 1.)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
    grammar.add_rule('LIST', '\'\'', None, 2)
    grammar.add_rule('ATOM', q('a'), None, 2)
    grammar.add_rule('ATOM', q('b'), None, 2)
    # grammar.add_rule('ATOM', q('c'), None, 2)
    # grammar.add_rule('ATOM', q('d'), None, 2)
    # grammar.add_rule('ATOM', q('e'), None, 2)
    # grammar.add_rule('ATOM', q('f'), None, 2)

    return grammar