import collections


class SampleStream(object):
    """
    SampleStream is the basic class that others in this directory inherit from. It allows constructs
    where we process and pass on samples.
    """

    def __init__(self, generator=None):
        self.outputs = [] # what do I do to things?
        self.parent = None # back pointers are needed to call "iter"
        self.generator = generator

    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()

    def process(self, x):
        """ The main processing step that must be overridden. If it returns x, the sample continues in the stream;
            if it return None, the sample is deleted """
        return x

    def process_and_push(self, x):
        """
        Process x and push it to my kids if I should
        """

        v = self.process(x)

        if len(self.outputs) > 0: # If I have children, return the output of my last child
            last = None
            if v is not None:
                for o in self.outputs:
                    last = o.process_and_push(v)

            return last # return our last output
        else:
            # Otherwise return what I did (if I'm the leaf)
            return v

    def __rshift__(self, other):
        """
        Implements the shift operator. So

        A >> B

        calls A.__rshift__(B).

        We have to return a new SampleStream object or else we can't chain >>
        """

        self.outputs.append(other)
        other.parent = self

        return other

    def __iter__(self):
        if self.parent is not None:
            assert self.generator is None, "*** Cannot have a parent and a generator"

            # Just act as though my parent is generating
            # This means I can call "iter" on a long chain of >>, and it will act as though
            # The top generator is being run through
            for x in self.parent:
                yield x
        else:
            # Otherwise, I am the top level. So I should process and then make my kids process too.
            # assert I am the top level
            assert self.generator is not None, "*** If I don't have a parent, I must have a generator"
            with self:
                for x in self.generator:
                    ## Ugh the top level has to yield the result of recursing process_and_push all the way down
                    v = self.process_and_push(x)


                    if v is not None:
                        yield v


    def __enter__(self):
        for a in self.outputs:
            a.__enter__()

    def __exit__(self, t, value, traceback):
        """ I defaultly call all of my children's exits """
        for a in self.outputs:
            a.__exit__(t, value, traceback)
        return False


if __name__ == "__main__":

    from LOTlib import break_ctrlc
    from LOTlib.Examples.Number.Model import make_hypothesis, make_data
    from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
    from LOTlib.SampleStream import *

    sampler = break_ctrlc(MHSampler(make_hypothesis(), make_data()))

    for h in SampleStream(sampler) >> Tee(Skip(2) >> Unique() >> Print(), PosteriorTrace()) >> Z():
        pass







