

class MCMCSummary:
    """
        This is a superclass for representing summary statistics from various model runs.
    """
    def __init__(self):
        pass

    def add(self, sample):
        raise NotImplementedError

    def __call__(self, gen):
        """
        If we pass this a generator, we add each element and then yield it, allowing us to make a pipeline
        """
        for g in gen:
            self.add(g)
            yield g

