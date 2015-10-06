
# Introduction

LOTlib is a library for inferring compositions of functions from observations of their inputs and outputs. This tutorial will introduce a very simple problem and how it can be solved in LOTlib. 

Suppose that you know basic arithmetic operations (called "primitives") like addition (+), subtraction (-), multiplication (*) and division (/). You observe a number which has been constructed using these operations, and wish to infer which operations were used. We'll assume that you observe the single number *12* and then do Bayesian inference in order to discover which operations occurred. For instance, 12 may be written as (1+1)*6, involving an addition, a multiplication, two uses of 1, and one use of 6. Or it may have been written as 1+11, or 2*3*2, etc. There are lots of other ways.

# Let's define a grammar

The general strategy of LOTlib models is to specify a space of possible compositions using a grammar. The grammar is actually a probabilistic context free grammar (with one small modification described below) that specifies a prior distribution on trees, or equivalently compositional structures like (1+1)*6, 2+2+2+2+2+2, (1+1)+(2*5), etc. If this is unfamiliar, the wiki on [PCFGs](https://help.github.com/articles/markdown-basics/) would be useful to read first. 

However, the best way to understand the grammar is as a way of specifying a program: any expansion of the grammar "renders" into a python program, whose code can then be evaluated. This will be made more concrete later

Here is how we can construct a grammar

```
    from LOTlib.Grammar import Grammar
    
    # Define a grammar object
    # Defaultly this has a start symbol called 'START' but we want to call 
    # it 'EXPR'
    grammar = Grammar(start='EXPR')
    
    # Define some operations
    grammar.add_rule('EXPR', '(%s + %s)', ['EXPR', 'EXPR'], 1.0)
    grammar.add_rule('EXPR', '(%s * %s)', ['EXPR', 'EXPR'], 1.0)
    grammar.add_rule('EXPR', '(float(%s) / float(%s))', ['EXPR', 'EXPR'], 1.0)
    grammar.add_rule('EXPR', '(-%s)', ['EXPR'], 1.0)
    
    # And define some numbers. We'll give them a 1/n^2 probability
    for n in xrange(1,10):
        grammar.add_rule('EXPR', str(n), None, 1.0/n**2)
```
A few things to note here. The grammar rules have the format
```
    grammar.add_rule( <NONTERMINAL>, <FUNCTION>, <ARGUMENTS>, <PROBABILITY>)
```
where <NONTERMINAL> says what nonterminal this rule is expanding. Here there is only one kind of nonterminal, an expression (EXPR). <FUNCTION> here is the function that this rule represents. These are strings that name defined functions in LOTlib, but they can also be strings (as here) where the <ARGUMENTS> get substituted in via string substitution (so for instance, "(%s+%s)" can be viewed as the function `lambda x,y: eval("(%s+%s)"%(x,y)))`. The arguments are a list of the arguments to the function. If the <FUNCTION> is a terminal that does not take arguments (as in the numbers 1..10), the <ARGUMENTS> part of a rule should be None. Note that None is very different from an empty list:
```
    grammar.add_rule('EXPR', 'hello', None, 1.0)
```
renders into the program "hello" but 
```
    grammar.add_rule('EXPR', 'hello', [], 1.0)
``
renders into "hello()". 

We can see some productions from this grammar if we call Grammar.generate. We will also show the probability of this tree according to the grammar, which is computed by renormalizing the <PROBABILITY> values of each rule when expanding each nonterminal:
```
    for _ in xrange(100):
        t = grammar.generate()
        print grammar.log_probability(t), t 
```
As you can see, the longer/bigger trees have lower (more negative) probabilities, implementing essentially a simplicity bias. These PCFG probabilities will often be our prior for Bayesian inference. 

Note that even though each `t` is a tree (a hierarchy of LOTlib.FunctionNodes), it renders nicely above as a string. This is defaultly how to expressions are evaluated in python. But we can see more of the internal structure using `t.fullprint()`, which shows the nonterminals, functions, and arguments at each level:
```
    t = grammar.generate()
    t.fullprint()
``
# Hypotheses

The grammar nicely specifies a space of expressions, but LOTlib needs a "hypothesis" to perform inference. In most cases, a hypothesis will represent a single production from the grammar. In LOTlib, hypotheses must define functions for computing priors, computing the likelihood of data, and implementing proposals in order for MCMC to work. 

Fortunately, for our purposes, there is a simple hypothesis class that it built-in to LOTlib which defaultly implements these. Let's just use it here. 
```
    from math import log
    from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
    
    # define a 
    class MyHypothesis(LOTHypothesis):
        def __init__(self, **kwargs):
            LOTHypothesis.__init__(self, grammar=grammar, args=[], **kwargs)
    
        def compute_single_likelihood(self, datum):
            if self(*datum.input) == datum.output:
                return log((1.0-datum.alpha)/100. + datum.alpha)
            else:
                return log((1.0-datum.alpha)/100.)
            
```
There are a few things going on here. First, we import LOTHypothesis and use that. It defines `compute_prior()` and `compute_likelihood(data)`. We define the initializer `__init__`. We overwrite the LOTHypothesis default and specify that the grammar we want is the one defined above. LOTHypotheses also defaultly take an argument called `x` (more on this later), but for now we want our hypothesis to be a function of no arguments. So we set `args=[]`. 

Essentially, `compute_likelihood` maps `compute_single_likelihood` over a list of data (treating each as IID conditioned on the hypothesis). So when we want to define how the likelihood works, we typically want to overwrite `compute_single_likelihood` as we have above. In this function, we expect an input `datum` with attirbutes `input`, `output`, and `alpha`. The LOTHypothesis `self` can be viewed as a function (here, one with no arguments) and so it can be called on `datum.input`. The likelihood this defines is one in which we generate a random number from 1..100 with probability `1-datum.alpha` and the correct number with probability `datum.alpha`. Thus, when the hypothesis returns the correct value (e.g. `self(*datum.input) == datum.output`) we must add these quantities to get the total probability of producing the data. When it does not, we must return only the former. LOTlib.Hypotheses.Likelihoods defines a number of other standard likelihoods, including the most commonly used  one, `BinaryLikelihood`. 

# Data

Given that our hypothesis wants those kinds of data, we can then create data as follows:

```
    from LOTlib.DataAndObjects import FunctionData
    data = [ FunctionData(input=[], output=12, alpha=0.99) ]
```
Note here that the most natural form of data is a list (where each element, a datum, gets passed to `compute_single_likelihood`). 

# Making hypotheses

We may now use our definition of a hypothesis to make one. If we call the initializer without a `value` keyword, LOTHypothesis just samples it from the given grammar: 
```
    h = MyHypothesis()
    print h.compute_prior(), h.compute_likelihood(data), h
```
Even better, `MyHypothesis` also inherits a `compute_posterior` function:
```
    print h.compute_posterior(data), h.compute_prior(), h.compute_likelihood(data), h
```
For convenience, when `compute_posterior` is called, it sets attributes on `h` for the prior, likelihood, and posterior (score):
```
    h.compute_prior(data)
    print h.posterior_score, h.prior, h.likelihood, h
```

# Running MCMC

We are almost there. We have define a grammar and a hypothesis which uses the grammar to define a prior, and custom code to define a likelihood. LOTlib's main claim to fame is that we can simply import MCMC routines and do inference over the space defined by the grammar. It's very easy:
```
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    
    # define a "starting hypothesis". This one is essentially copied by 
    # all proposers, so the sampler doesn't need to know its type or anything. 
    h0 = MyHypothesis()
    
    # Now use the sampler like an iterator. In MHSampler, compute_posterior gets called
    # so when we have an h, we can get its prior and likelihood
    for h in MHSampler(h0, data, steps=100):
        print h.posterior_score, h.prior, h.likelihood, h 
```
That probably went by pretty fast. Here's another thing we can do:
```
    h0 = MyHypothesis()
    
    from collections import Counter

    count = Counter()
    for h in MHSampler(h0, data, steps=1000):
        count[h] += 1
    
    for h in sorted(count.keys(), key=lambda x: count[x]):
        print count[h], h.posterior_score, h
```
LOTlib hypotheses are required to hash nicely, meaning that they can be saved or put into dictionaries and sets like this. 

# Making our Hypothesis more robust

It's possible that in running the above code, you got a zero division error. Can you see why this can happen?

Fortunately, we can "hack" our hypothesis class to address this by catching the exception. A smart way to do this is to override `__call__` and return an appropriate value when such an error occurs:
```

    class MyHypothesis(LOTHypothesis):
        def __init__(self, **kwargs):
            LOTHypothesis.__init__(self, grammar=grammar, args=[], **kwargs)
            
        def __call__(self, *args):
            try:
                # try to do it from the superclass
                return LOTHypothesis.__call__(self, *args)
            except ZeroDivisionError:
                # and if we get an error, return nan
                return float("nan")
    
        def compute_single_likelihood(self, datum):
            if self(*datum.input) == datum.output:
                return log((1.0-datum.alpha)/100. + datum.alpha)
            else:
                return log((1.0-datum.alpha)/100.)
```

# Getting serious about running

Now with more robust code, we can run the `Counter` code above for longer and get a better picture of the posterior. Often, though, in LOTlib models it helps to start multiple chains. Each chain gets its own hypothesis which is randomly initialized from the grammar. This often helps the hypotheses to "fall" into the right regions of space. Let's import a class for running multiple chains and try it, running even more steps:
```
    from LOTlib.Inference.Samplers.MultipleChainMCMC import MultipleChainMCMC
    from collections import Counter
    
    # now instead of starting from a single h0, we need a function to make the h0. 
    # The constructor for MyHypothesis will do this well!

    count = Counter()
    for h in MultipleChainMCMC(MyHypothesis, data, steps=100000, nchains=10):
        # Note that this yields our sampled h back interwoven between chains
        count[h] += 1
    
    
    for h in sorted(count.keys(), key=lambda x: count[x]):
        print count[h], h.posterior_score, h.prior, h.likelihood, h 
```
If our sampler is working correctly, it should be the case that the time average of the sampler (the `h`es from the for loop) should approximate the posterior distribution (e.g. their re-normalized scores). Let's use this code to see if that's true
```
    # Miscellaneous stores a number of useful functions. Here, we need logsumexp, which will
    # compute the normalizing constant for posterior_scores when they are in log space
    from LOTlib.Miscellaneous import logsumexp 
    from numpy import exp # but things that are handy in numpy are not duplicated (usually)
    
    # get a list of all the hypotheses we found. This is necessary because we need a fixed order,
    # which count.keys() does not guarantee unless we make a new variable. 
    hypotheses = count.keys() 
    
    # first convert posterior_scores to probabilities. To this, we'll use a simple hack of 
    # renormalizing the psoterior_scores that we found. This is a better estimator of each hypothesis'
    # probability than the counts from the sampler
    z = logsumexp([h.posterior_score for h in hypotheses])
    
    posterior_probabilities = [ exp(h.posterior_score - z) for h in hypotheses ]
    
    # and compute the probabilities over the sampler run
    cntz = sum(count.values())    
    sampler_counts          = [ float(count[h])/cntz for h in hypotheses ] 
    
    ## and let's just make a simple plot
    import matplotlib.pyplot as pyplot
    fig = pyplot.figure()
    plt = fig.add_subplot(1,1,1)
    plt.scatter(posterior_probabilities, sampler_counts)
    fig.show()
    
```

Combining our definitions of data and hypotheses, 


## Function with arguments



If we want to create an instance of this hypothesis, 

What LOTlib does is take this prior distribution on compositions and data, and infer the posterior distribution on hypotheses (or trees, or compositions of primitives). There is only a little magic involved in LOTlib---it has 