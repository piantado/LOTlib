# -*- coding: utf-8 -*-

"""
        A function node -- a tree part representing a function and its arguments.
        Also used for PCFG rules, where the arguments are nonterminal symbols.

"""
# TODO: This could use some renaming FunctionNode.bv is not really a bound variable--its a list of rules that were added
## TODO: We should be able to eval list-builders like cons_ and list_ without calling python eval -- should be as fast as mapping to strings!


import re
from LOTlib.Miscellaneous import None2Empty
from copy import copy, deepcopy
from random import random

def isFunctionNode(x):
    # just because this is nicer, and allows us to map, etc.
    """
            Returns true if *x* is of type FunctionNode.
    """
    return isinstance(x, FunctionNode)


def cleanFunctionNodeString(x):
    """
            Makes FunctionNode strings easier to read
    """
    s = re.sub("lambda", u"\u03BB", str(x)) # make lambdas the single char
    s = re.sub("_", '', s) # remove underscores
    return s


class FunctionNode(object):
    """
            *returntype*
                    The return type of the FunctionNode

            *name*
                    The name of the function

            *args*
                    Arguments of the function

            *generation_probability*
                    Unnormalized generation probability.

            *resample_p*
                    The probability of choosing this node in resampling. Takes any number >0 (all are normalized)

            *bv_name*
                    Name of the Bound Variable e.g. y1, y2, y3...

            *bv_type*
                    Bound variable type

            *bv_args*
                    Arguments of the Bound Variable. "None" implies this is a terminal, otherwise a type signature.

            *bv_prefix*
                    Bound variable Prefix e.g. the 'y' in y1, y2, y3...

            *bv_p*
                    Unnormalized probability of the rule expanding to the bound variable.

            *ruleid*
                    The rule ID number

            NOTE: If a node has [ None ] as args, it is treated as a thunk

            bv - stores the actual *rule* that was added (so that we can re-add it when we loop through the tree)
    """

    def __init__(self, returntype, name, args, generation_probability=0.0,
                                    resample_p=1.0, bv_name=None, bv_type=None, bv_args=None,
                                    bv_prefix=None, bv_p=None, ruleid=None):
        self.__dict__.update(locals())

    def setto(self, q):
        """
                Makes all the parts the same as q, not copies.
        """
        self.__dict__.update(q.__dict__)

    def __copy__(self, shallow=False):
        """
                Copy a function node

                *shallow* - if True, this does not copy the children (self.to points to the same as what we return)
        """
        if (not shallow) and self.args is not None:
            newargs = map(copy, self.args)
        else:
            newargs = self.args

        return FunctionNode(self.returntype, self.name, newargs,
                generation_probability=self.generation_probability,
                resample_p=self.resample_p, bv_type=self.bv_type,
                bv_name=self.bv_name, bv_args=deepcopy(self.bv_args),
                bv_prefix=self.bv_prefix, bv_p=self.bv_p,
                ruleid=self.ruleid)

    def is_nonfunction(self):
        """
                Returns True if the Node contains no function arguments, False otherwise.
        """
        return (self.args is None)

    def is_function(self):
        """
                Returns True if the Node contains function arguments, False otherwise.
        """
        return not self.is_nonfunction()

    def is_leaf(self):
        """
                Returns True if none of the kids are FunctionNodes, meaning that this should be considered a "leaf"
                NOTE: A leaf may be a function, but its args are specified in the grammar.
        """
        return (self.args is None) or all([ not isFunctionNode(c) for c in self.args])

    def as_list(self):
        """
                Returns a list representation of the FunctionNode with function/self.name as the first element.

                NOTE: This does not handle BV yet
        """
        x = [self.name] if self.name != '' else []
        if self.args is not None:
            x.extend( [a.as_list() if isFunctionNode(a) else a for a in self.args] )
        return x

    def islambda(self):
        """
                Is this a lambda node? Right now it
                just checks if the name is 'lambda' (but in the future, we might want to
                allow different types of lambdas or something, which is why its nice to
                have this function)
        """
        if self.name is None:
            return False
        elif self.name.lower() == 'lambda':
            assert len(self.args) == 1
            return True
        else:
            return False

    # NOTE: Here we do a little fanciness -- with "if" -- we convert it to the "correct" python
    # form with short circuiting instead of our fancy ifelse function
    def pystring(self, serialize_bvs=False):
        """
        Outputs a string that can be evaluated by python
        """
        #print ">>", self.name
        if self.is_nonfunction(): # a terminal
            return str(self.name)
        elif self.name == "if_": # this gets translated
            assert len(self.args) == 3, "if_ requires 3 arguments!"
            return '(' + str(self.args[1]) + ' if ' + str(self.args[0]) + ' else ' + str(self.args[2]) + ')'
            #return '(' + str(self.args[1]) + ') if (' + str(self.args[0]) + ') else (' + str(self.args[2]) + ')'
        elif self.name == '':
            assert len(self.args) == 1, "Null names must have exactly 1 argument"
            return str(self.args[0])
        elif self.name is not None and self.name.lower() == 'apply_':
            assert self.args is not None and len(self.args)==2, "Apply requires exactly 2 arguments"
            return '('+str(self.args[0])+')('+str(self.args[1])+')'
        elif self.islambda():

            return 'lambda '+ (self.bv_name if self.bv_name is not None else '') +': '+str(self.args[0])

        else:

            if self.args is None:
                return str(self.name)+'()' # simple call
            else:
                return str(self.name)+'('+', '.join(map(str,self.args))+')'

    def quickstring(self):
        """
                A (maybe??) faster string function used for hashing -- doesn't handle any details and is meant
                to just be quick
        """
        if self.args is None:
            return str(self.name) # simple call
        else:
            # Using + on strings is very slow, this needs changing.
            return str(self.name)+' '+','.join(map(str,self.args))

    def fullprint(self, d=0):
        """ A handy printer for debugging"""
        tabstr = "  .  " * d
        print tabstr, self.returntype, self.name, self.bv_type, self.bv_name, self.bv_args, self.bv_prefix, "\t", self.generation_probability, "\t", self.resample_p
        if self.args is not None:
            for a in self.args:
                if isFunctionNode(a):
                    a.fullprint(d+1)
                else:
                    print tabstr, a

    def schemestring(self):
        """
                Print out in scheme format (+ 3 (- 4 5)).
        """
        if self.args is None:
            return self.name
        else:
            return '('+self.name + ' ' + ' '.join(map(lambda x: x.schemestring(), None2Empty(self.args)))+')'

    def liststring(self, cons="cons_"):
        """
                This "evals" cons_ so that we can conveniently build lists (of lists) without having to eval.
                Mainly useful for combinatory logic, or "pure" trees
        """
        if self.args is None:
            return self.name
        elif self.name == cons:
            return map(lambda x: x.liststring(), self.args)
        else:
            assert False, "FunctionNode must only use cons to call liststring!"

    # NOTE: in the future we may want to change this to do fancy things
    def __str__(self):
        return self.pystring()

    def __repr__(self):
        return self.pystring()

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __eq__OLD(self, other):
         return isFunctionNode(other) and (cmp(self, other) == 0)

    def __eq__(self, other, bv_dict=None):
        """
                Tests equality of FunctionNodes up to bound variables.

                NOTE: BV equality has not been tested yet.
        
        """
        
        if bv_dict is None:
            bv_dict = dict()

        # If they have different names, they aren't equal
        if (not isFunctionNode(other)) or (self.name != other.name):
            return False

        # If both don't have args, they are equal.
        if self.args is None:
            return other.args is None

        # If they have a different number of args, they aren't equal
        if other.args is not None and len(self.args) != len(other.args):
            return False

        # If the bound variable already exists in the dict, see if
        # they're the same
        if self.name in bv_dict and bv_dict[self.name] != other.name:
            return False

        # If it doesn't exist in the dict, add it.
        if self.islambda():
            if other.islambda():
                bv_dict[self.bv_name]=other.bv_name
            else: # if the other isn't a lambda, must be false
                return False

        # so args must be a list
        for a,b in zip(self.args, other.args):
            if isFunctionNode(a):
                if not (isFunctionNode(b) and a.__eq__(b, bv_dict)):
                    return False
            elif a != b: # fall back on default comparison
                return False

        return True

    ## TODO: overwrite these with something faster
    # hash trees. This just converts to string -- maybe too slow?
    def __hash__(self):

        # An attempt to speed things up -- not so great!
        #hsh = self.ruleid
        #if self.args is not None:
            #for a in filter(isFunctionNode, self.args):
                #hsh = hsh ^ hash(a)
        #return hsh

        # normal string hash -- faster?
        return hash(str(self))

        # use a quicker string hash
        #return hash(self.quickstring())

    def __cmp__(self, x):
        return cmp(str(self), str(x))

    def __len__(self):
        return len([a for a in self])

    def log_probability(self):
        """
                Compute the log probability of a tree
        """
        lp = self.generation_probability # the probability of my rule

        if self.args is not None:
            lp += sum([x.log_probability() for x in self.argFunctionNodes() ])
        return lp

    def subnodes(self):
        """
                Return all subnodes -- no iterator.
                Useful for modifying

                NOTE: If you want iterate using the grammar, use Grammar.iterate_subnodes
        """
        return [g for g in self]

    def argFunctionNodes(self):
        """
                Yield FunctionNode immediately below
                Also handles args is None, so we don't have to check constantly
        """
        if self.args is not None:
            # TODO: In python 3, use yeild from
            for n in filter(isFunctionNode, self.args):
                yield n

    def is_terminal(self):
        """
            A FunctionNode is considered a "terminal" if it has no FunctionNodes below
        """
        return len(filter(isFunctionNode, self.args)) == 0

    def __iter__(self):
        """
                Iterater for subnodes.
                NOTE: This will NOT work if you modify the tree. Then all goes to hell.
                      If the tree must be modified, use self.subnodes()
        """
        yield self

        if self.args is not None:
            for a in self.argFunctionNodes():
                for ssn in a:
                    yield ssn

    def iterdepth(self):
        """
                Iterates subnodes, yielding node and depth
        """
        yield (self,0)

        if self.args is not None:
            for a in self.argFunctionNodes():
                for ssn,dd in a.iterdepth():
                    yield (ssn,dd+1)

    def all_leaves(self):
        """
                Returns a generator for all leaves of the subtree rooted at the instantiated FunctionNode.
        """
        if self.args is not None:
            for i in range(len(self.args)): # loop through kids
                if isFunctionNode(self.args[i]):
                    for ssn in self.args[i].all_leaves():
                        yield ssn
                else:
                    yield self.args[i]

    def string_below(self, sep=" "):
        """
                The string of terminals (leaves) below the current FunctionNode in the parse tree.

                *sep* is the delimiter between terminals. E.g. sep="," => "the,fuzzy,cat"
        """
        return sep.join(map(str, self.all_leaves()))

    def fix_bound_variables(self, d=1, rename=None):
        """
                Fix the naming scheme of bound variables. This happens if we promote or demote some nodes
                via insert/delete

                *d* - Current depth.

                *rename* - a dictionary to store how we should rename
        """
        if rename is None:
            rename = dict()

        if self.name is not None:
            if self.islambda() and (self.bv_type is not None):
                assert self.args is not None

                newname = self.bv_prefix+str(d)

                # And rename this below
                rename[self.bv_name] = newname
                self.bv_name = newname
            elif self.name in rename:
                self.name = rename[self.name]

        # and recurse
        for k in self.argFunctionNodes():

            #print "\t\tRENAMING", k, k.bv_prefix, rename
            k.fix_bound_variables(d+1, rename)

    ############################################################
    ## Derived functions that build on the above core
    ############################################################

    def contains_function(self, x):
        """
                Checks if the FunctionNode contains x as function below
        """
        for n in self:
            if n.name == x:
                return True
        return False

    def count_nodes(self):
        """
                Returns the subnode count.
        """
        return self.count_subnodes()

    def count_subnodes(self):
        """
                Returns the subnode count.
        """
        c = 0
        for _ in self:
            c = c + 1
        return c

    def depth(self):
        """
                Returns the depth of the tree (how many embeddings below)
        """
        depths = [ a.depth() for a in self.argFunctionNodes() ]
        depths.append(-1) # for no function nodes (+1=0)
        return max(depths)+1

    # get a description of the input and output types
    # if collapse_terminal then we just map non-FunctionNodes to "TERMINAL"
    def type(self):
        """
                The type of a FunctionNode is defined to be its returntype if it's not a lambda,
                or is defined to be the correct (recursive) lambda structure if it is a lambda.
                For instance (lambda x. lambda y . (and (empty? x) y))
                is a (SET (BOOL BOOL)), where in types, (A B) is something that takes an A and returns a B
        """

        if self.name == '': # If we don't have a function call (as in START), just use the type of what's below
            assert len(self.args) == 1, "**** Nameless calls must have exactly 1 arg"
            return self.args[0].type()
        if (not self.islambda()):
            return self.returntype
        else:
            # figure out what kind of lambda
            t = []
            if self.bv_args is not None:
                t = tuple( [self.bv_type,] + copy(self.bv_args) )
            else:
                t = self.bv_type

            return (self.args[0].type(), t)

        # ts = [self.returntype, self.bv_type, self.bv_args]
        # if self.args is not None:
        #       for i in range(len(self.args)):
        #               if isFunctionNode(self.args[i]):
        #                       ts.append(self.args[i].returntype)
        #               else:
        #                       ts.append(self.args[i])
        # return ts

    def is_replicating(self):
        """
                A function node is replicating (by definition) if one of its children is of the same type
        """
        return any([ x.returntype == self.returntype for x in self.argFunctionNodes() ])

    def is_canonical_order(self, symmetric_ops):
        """
                Takes a set of symmetric (commutative) ops (plus, minus, times, etc, not divide)
                and asserts that the LHS ordering is less than the right (to prevent)

                This is useful for removing duplicates of nodes. For instance,

                        AND(X, OR(Y,Z))

                is logically the same as

                        AND(OR(Y,Z), X)

                This function essentially checks if the tree is in sorted (alphbetical)
                order, but only for functions whose name is in symmetric_ops.
        """
        if self.args is None or len(self.args) == 0:
            return True

        if self.name in symmetric_ops:

            # Then we must check children
            if self.args is not None:
                for i in xrange(len(self.args)-1):
                    if self.args[i].name > self.args[i+1].name:
                        return False

        # Now check the children, whether or not we are symmetrical
        return all([x.is_canonical_order(symmetric_ops) for x in self.args if self.args is not None])

    def replace_subnodes(self, find, replace):
        """
                *find*s subnodes and *replace*s it.
                NOTE: NOT THE FASTEST!
                NOTE: Defaultly only makes copies of replace
        """

        # now go through and modify
        for g in filter(lambda x: x==find, self.subnodes() ): # NOTE: must use subnodes since we are modfiying
            g.setto(copy(replace))

    def partial_subtree_root_match(self, y):
        """
                Does *y* match from my root?

                A partial tree here is one with some nonterminals (see random_partial_subtree) that
                are not expanded
        """
        if isFunctionNode(y):
            if y.returntype != self.returntype: return False
            if y.name != self.name: return False

            if y.args is None: return self.args is None
            if len(y.args) != len(self.args): return False

            for a, b in zip(self.args, y.args):
                if isFunctionNode(a):
                    if not a.partial_subtree_root_match(b): return False
                else:
                    if isFunctionNode(b): return False # cannot work!

                    # neither is a function node
                    if a != b: return False

            return True
        else:
            # else y is a string and we match if y is our returntype
            assert isinstance(y,str)
            return y == self.returntype

    def partial_subtree_match(self, y):
        """
                Does *y* match a subtree anywhere?
        """
        for x in self:
            if x.partial_subtree_root_match(y): return True

        return False

    def random_partial_subtree(self, p=0.5):
        """
                Generate a random partial subtree of me. So that

                this:
                        prev_((seven_ if cardinality1_(x) else next_(next_(L_(x)))))
                yeilds:

                        prev_(WORD)
                        prev_(WORD)
                        prev_((seven_ if cardinality1_(x) else WORD))
                        prev_(WORD)
                        prev_((seven_ if BOOL else next_(next_(L_(SET)))))
                        prev_(WORD)
                        prev_((seven_ if cardinality1_(SET) else next_(WORD)))
                        prev_(WORD)
                        prev_((seven_ if BOOL else next_(WORD)))
                        ...

                We do this because there are waay too many unique subtrees to enumerate,
                and this allows a nice variety of structures
                NOTE: Partial here means that we include nonterminals with probability p
        """

        if self.args is None: return copy(self)

        newargs = []
        for a in self.args:
            if isFunctionNode(a):
                if random() < p: newargs.append( a.returntype )
                else:            newargs.append( a.random_partial_subtree(p=p) )
            else:
                newargs.append(a) # string or something else

        ret = self.__copy__(shallow=True) # don't copy kids
        ret.args = newargs

        return ret
