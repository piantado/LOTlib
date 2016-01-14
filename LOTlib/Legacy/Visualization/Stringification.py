"""
    Functions for mappings FunctionNodes to strings
"""
from LOTlib.FunctionNode import isFunctionNode, BVUseFunctionNode, BVAddFunctionNode
import re
percent_s_regex = re.compile(r"%s")

def schemestring(x, d=0, bv_names=None):
    """Outputs a scheme string in (lambda (x) (+ x 3)) format.

    Arguments:
        x: We return the string for this FunctionNode.
        bv_names: A dictionary from the uuids to nicer names.

    """
    if isinstance(x, str):
        return x
    elif isFunctionNode(x):

        if bv_names is None:
            bv_names = dict()

        name = x.name
        if isinstance(x, BVUseFunctionNode):
            name = bv_names.get(x.name, x.name)

        if x.args is None:
            return name
        else:
            if x.args is None:
                return name
            elif isinstance(x, BVAddFunctionNode):
                assert name is 'lambda'
                return "(%s (%s) %s)" % (name, x.added_rule.name,
                                         map(lambda a: schemestring(a, d+1, bv_names=bv_names), x.args))
            else:
                return "(%s %s)" % (name, map(lambda a: schemestring(a,d+1, bv_names=bv_names), x.args))


def fullstring(x, d=0, bv_names=None):
    """
    A string mapping function that is for equality checking. This is necessary because pystring silently ignores
    FunctionNode.names that are ''. Here, we print out everything, including returntypes
    :param x:
    :param d:
    :param bv_names:
    :return:
    """

    if isinstance(x, str):
        return x
    elif isFunctionNode(x):

        if bv_names is None:
            bv_names = dict()


        if x.name == 'lambda':
            # On a lambda, we must add the introduced bv, and then remove it again afterwards

            bvn = ''
            if isinstance(x, BVAddFunctionNode) and x.added_rule is not None:
                bvn = x.added_rule.bv_prefix+str(d)
                bv_names[x.added_rule.name] = bvn

            assert len(x.args) == 1
            ret = 'lambda<%s> %s: %s' % ( x.returntype, bvn, fullstring(x.args[0], d=d+1, bv_names=bv_names) )

            if isinstance(x, BVAddFunctionNode) and x.added_rule is not None:
                try:
                    del bv_names[x.added_rule.name]
                except KeyError:
                    x.fullprint()

            return ret
        else:

            name = x.name
            if isinstance(x, BVUseFunctionNode):
                name = bv_names.get(x.name, x.name)

            if x.args is None:
                return "%s<%s>"%(name, x.returntype)
            else:
                return "%s<%s>(%s)" % (name,
                                       x.returntype,
                                       ', '.join(map(lambda a: fullstring(a, d=d+1, bv_names=bv_names), x.args)))




def pystring(x, d=0, bv_names=None):
    """Output a string that can be evaluated by python; gives bound variables names based on their depth.

    Args:
        bv_names: A dictionary from the uuids to nicer names.

    """
    if isinstance(x, str):
        return x
    elif isFunctionNode(x):

        if bv_names is None:
            bv_names = dict()

        if x.name == "if_": # this gets translated
            assert len(x.args) == 3, "if_ requires 3 arguments!"
            # This converts from scheme (if bool s t) to python (s if bool else t)
            b = pystring(x.args[0], d=d+1, bv_names=bv_names)
            s = pystring(x.args[1], d=d+1, bv_names=bv_names)
            t = pystring(x.args[2], d=d+1, bv_names=bv_names)
            return '( %s if %s else %s )' % (s, b, t)
        elif x.name == '':
            assert len(x.args) == 1, "Null names must have exactly 1 argument"
            return pystring(x.args[0], d=d, bv_names=bv_names)
        elif x.name == ',': # comma join
            return ', '.join(map(lambda a: pystring(a, d=d, bv_names=bv_names), x.args))
        elif x.name == "apply_":
            assert x.args is not None and len(x.args)==2, "Apply requires exactly 2 arguments"
            #print ">>>>", self.args
            return '( %s )( %s )' % tuple(map(lambda a: pystring(a, d=d, bv_names=bv_names), x.args))
        elif x.name == 'lambda':
            # On a lambda, we must add the introduced bv, and then remove it again afterwards

            bvn = ''
            if isinstance(x, BVAddFunctionNode) and x.added_rule is not None:
                bvn = x.added_rule.bv_prefix+str(d)
                bv_names[x.added_rule.name] = bvn

            assert len(x.args) == 1
            ret = 'lambda %s: %s' % ( bvn, pystring(x.args[0], d=d+1, bv_names=bv_names) )

            if isinstance(x, BVAddFunctionNode) and x.added_rule is not None:
                try:
                    del bv_names[x.added_rule.name]
                except KeyError:
                    x.fullprint()

            return ret
        elif percent_s_regex.search(x.name): # If we match the python string substitution character %s, then format
            return x.name % tuple(map(lambda a: pystring(a, d=d+1, bv_names=bv_names), x.args))
        else:

            name = x.name
            if isinstance(x, BVUseFunctionNode):
                name = bv_names.get(x.name, x.name)

            if x.args is None:
                return name
            else:
                return name+'('+', '.join(map(lambda a: pystring(a, d=d+1, bv_names=bv_names), x.args))+')'


def lambdastring(fn, d=0, bv_names=None):
    """
            A nicer printer for pure lambda calculus. This can use unicode for lambdas
    """
    if bv_names is None:
        bv_names = dict()

    if fn is None: # just pass these through -- simplifies a lot
        return None
    elif fn.name == '':
        assert len(fn.args)==1
        return lambdastring(fn.args[0])
    elif isinstance(fn, BVAddFunctionNode):
        assert len(fn.args)==1 and fn.name == 'lambda'
        if fn.added_rule is not None:
            bvn = fn.added_rule.bv_prefix+str(d)
            bv_names[fn.added_rule.name] = bvn
        return u"\u03BB%s.%s" % (bvn, lambdastring(fn.args[0], d=d+1, bv_names=bv_names)) # unicode version with lambda
        #return "L%s.%s" % (bvn, lambda_str(fn.args[0], d=d+1, bv_names=bv_names))
    elif fn.name == 'apply_':
        assert len(fn.args)==2
        if fn.args[0].name == 'lambda':
            return "((%s)(%s))" % tuple(map(lambda a: lambdastring(a, d=d+1, bv_names=bv_names), fn.args))
        else:
            return "(%s(%s))"   % tuple(map(lambda a: lambdastring(a, d=d+1, bv_names=bv_names), fn.args))
    elif isinstance(fn, BVUseFunctionNode):
        assert fn.args is None
        return bv_names[fn.name]
    else:
        assert fn.args is None
        assert not percent_s_regex(fn.name), "*** String formatting not yet supported for lambdastring"
        return str(fn.name)