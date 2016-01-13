
"""

Todo: We could probably speed this up by having the conversion of data not happen with every call, but only once, so
that make_data made them in lua format.

"""



WORDS = ['one_', 'two_', 'three_', 'four_', 'five_', 'six_', 'seven_', 'eight_', 'nine_', 'ten_']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a simple grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from LOTlib.Miscellaneous import q
from LOTlib.Grammar import Grammar

grammar = Grammar()

grammar.add_rule('START', '', ['WORD'], 1.0)

grammar.add_rule('BOOL', '(%s and %s)',    ['BOOL', 'BOOL'], 1./3.)
grammar.add_rule('BOOL', '(%s or %s)',     ['BOOL', 'BOOL'], 1./3.)
grammar.add_rule('BOOL', '(not %s)',    ['BOOL'], 1./3.)

grammar.add_rule('BOOL', 'true',    None, 1.0/2.)
grammar.add_rule('BOOL', 'false',   None, 1.0/2.)

# note that this can take basically any types for return values
grammar.add_rule('WORD', 'ifelse',    ['BOOL', 'WORD', 'WORD'], 0.5)
grammar.add_rule('WORD', q('undef'), None, 0.5)

grammar.add_rule('BOOL', '(length(%s)==1)',    ['SET'], 1.0)
grammar.add_rule('BOOL', '(length(%s)==2)',    ['SET'], 1.0)
grammar.add_rule('BOOL', '(length(%s)==3)',    ['SET'], 1.0)

grammar.add_rule('BOOL', '(%s == %s)',    ['WORD', 'WORD'], 1.0)

grammar.add_rule('SET', 'union',     ['SET', 'SET'], 1./3.)
grammar.add_rule('SET', 'intersection',     ['SET', 'SET'], 1./3.)
grammar.add_rule('SET', 'setdifference',     ['SET', 'SET'], 1./3.)
grammar.add_rule('SET', 'select',     ['SET'], 1.0)

grammar.add_rule('SET', 'x',     None, 4.0)

grammar.add_rule('WORD', 'recurse',        ['SET'], 1.0)

grammar.add_rule('WORD', 'next', ['WORD'], 1.0/2.0)
grammar.add_rule('WORD', 'prev', ['WORD'], 1.0/2.0)

# These are quoted (q) since they are strings when evaled
for w in WORDS:
    grammar.add_rule('WORD', q(w), None, 1./len(WORDS))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Base code to be executed in every Lua Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BASE = """
-- Some definitions from http://www.phailed.me/2011/02/common-set-operations-in-lua/
local function find(a, tbl)
	for _,a_ in ipairs(tbl) do if a_==a then return true end end
end

function datan(n)
    local ret = {}
    for i=1,n do
        table.insert(ret, i)
    end
    return ret
end

function union(a, b)
	a = {unpack(a)}
	for _,b_ in ipairs(b) do
		if not find(b_, a) then table.insert(a, b_) end
	end
	return a
end

function intersection(a, b)
	local ret = {}
	for _,b_ in ipairs(b) do
		if find(b_,a) then table.insert(ret, b_) end
	end
	return ret
end

function setdifference(a, b)
	local ret = {}
	for _,a_ in ipairs(a) do
		if not find(a_,b) then table.insert(ret, a_) end
	end
	return ret
end

function select(a)
	return {a[1]}
end

function symmetric(a, b)
	return difference(union(a,b), intersection(a,b))
end

function length(T)
    return #T
end

function ifelse(cond , x , y )
    if cond then return x else return y end
end

-- Define the word list and next, etc.
WORDS = {'one_', 'two_', 'three_', 'four_', 'five_', 'six_', 'seven_', 'eight_', 'nine_', 'ten_'}

function indexof(w)
    -- the one-indexed location of w in WORDS, else -1
    for i, x in ipairs(WORDS) do
        if x == w then
            return i
        end
    end
    return -1
end

function next(w)
    idx = indexof(w)
    if idx < #WORDS then
        return WORDS[idx+1]
    else
        return 'undef'
    end
end

function prev(w)
    idx = indexof(w)
    if idx > 1 then
        return WORDS[idx-1]
    else
        return 'undef'
    end
end

"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define Lua Hypothesis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Examples.Number.Model import NumberExpression
import lupa

class LuaHypothesis(NumberExpression):

    def __init__(self, value=None, base=BASE, **kwargs):

        self.base = base # must be set before initializer

        self.Lua = lupa.LuaRuntime()
        self.Lua.execute(base)

        NumberExpression.__init__(self, grammar, value=value, display="%s", **kwargs)

    def __call__(self, s):
        lset = self.Lua.eval("datan(%s)"%len(s)) # make a set for lua of the same size
        return self.fvalue(lset)

    def compile_function(self):
        # Compile this function. We must execute it in order to define the recursion

        self.Lua.execute("""
function mycall(x)
    recurse_count = 0
    return recurse(x)
end

function recurse(x)
    recurse_count = recurse_count + 1
    if recurse_count < 25 then
        return %s
    end
    return 'undef'
end
""" % str(self.value))

        return self.Lua.eval("function(x) return mycall(x) end")




def make_hypothesis(*args, **kwargs):
    return LuaHypothesis(*args, **kwargs)

from LOTlib.Examples.Number.Model import make_data

if __name__ == "__main__":
    from LOTlib import break_ctrlc
    from LOTlib.Examples.Number.Model import make_data
    data = make_data(300)

    # To see a sampler
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    h0 = LuaHypothesis(base=BASE)
    for h in break_ctrlc(MHSampler(h0, data, steps=1000000)):
        print h.posterior_score, h.prior, h.likelihood, h
