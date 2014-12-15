import graphviz
from LOTlib.FunctionNode import FunctionNode

def make_dot(fn, ls=None, n='0', parent_n=None):
    """Setup self.dot as a graphviz / DOT format graph for this FunctionNode.

    We proceed through each node in the graph, creating unique names for each child by concat'ing
    child int to the parent string. E.g. node '01' may have children '010', '011', '012', ...

    This is done because in DOT format we need to enumerate edges between nodes, so we need a unique
    string name to draw edges. For example, node name 'plus_' will be ambiguous for '(plus_ (plus_ ...'

    Args:
        ls (list): List of items... the first item in the list is either the operator, or the only
            thing in the list.
        n (str): Name of this FunctionNode.
        parent_n (str): Name of parent FunctionNode.

    Requires:
        graphviz: install graphviz from www.graphviz.org, then enter::  $ pip install graphviz

    References:
        en.wikipedia.org/wiki/DOT_(graph_description_language)
        pypi.python.org/pypi/graphviz

    """
    # Initialize
    if not hasattr(fn, 'dot'):
        fn.dot = graphviz.Digraph(comment='The DOT Graph')
    if ls is None:
        ls = fn.as_list()[0]
    this_n = n
    d = fn.dot

    # handle items like 'bound=100' (are there others that will be like this?
    if not isinstance(ls, list) and (parent_n is None):
        if not (parent_n is None):
            d.node(this_n, label=str(ls), shape='square')
            d.edge(parent_n, this_n, style='dotted')

    if isinstance(ls, list):
        # node for this FunctionNode
        if len(ls) >= 1:
            d.node(this_n, label=str(ls[0]), shape='plaintext')
            if not (parent_n is None):
                d.edge(parent_n, this_n, style='solid')
        # children FunctionNodes...
        if len(ls) > 1:
            for i in range(1, len(ls)):
                # Recursive call
                fn.make_dot(ls[i], n=this_n+str(i), parent_n=this_n)

def dot_string(fn):
    """Return DOT graph format string; see make_dot docstring."""
    if not hasattr(fn, 'dot'):
        fn.make_dot()
    return fn.dot.source