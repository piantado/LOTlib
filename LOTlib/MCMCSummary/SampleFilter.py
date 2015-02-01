

class SampleFilter(object):
    """
        A sample filter is a class that others in MCMCSummary can inherit that allows them to be put inline
        in generator calls

        for h in sampler:
            ...


        f = SampleFilter()
        for h in f(sampler):
            ...

        These loops internally behave the same, but samples are added to f
    """

    def add(self, x):
        """
        Add a sample x to myself. Must be overwritten by subclasses
        """
        raise NotImplementedError

    def __call__(self, generator):
        for x in generator:
            self.add(x)
            yield x

