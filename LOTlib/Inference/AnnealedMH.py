"""

    An annealed MH sampler

"""
from math import sin, pi

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Set up some schedule classes for how temp varies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Schedule:
    """
    A class to represent an annealing schedule. Should be subclassed.
    NOTE: "skip" steps are all run at the same temperature!
    """
    def __init__(self):
        raise NotImplementedError

    def __iter__(self):
        return self


class ConstantSchedule(Schedule):
    def __init__(self, k):
        self.k = k

    def next(self):
        return self.k

class InverseSchedule(Schedule):
    """
    Temperature = max/(scale*time)
    """
    def __init__(self, max, scale):
        self.__dict__.update(locals())
        self.ticks = 0

    def next(self):
        self.ticks += 1
        return self.max / (self.scale*self.ticks)

class SinSchedule(Schedule):
    """
    Temperature = min + (max-min)*sin(2*pi*time/period)
    """
    def __init__(self, min, max, period):
        assert max>min
        assert period > 0.
        self.__dict__.update(locals())
        self.ticks = 0 # how many times have we called next?

    def next(self):
        self.ticks += 1
        return self.min + (self.max-self.min) * ( 1. + sin(self.ticks * 2 * pi / self.period))/2.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## An annealed MH sampler
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Inference.MetropolisHastings import MHSampler

class AnnealedMHSampler(MHSampler):
    def __init__(self, h0, data, prior_schedule=None, likelihood_schedule=None, **kwargs):
        MHSampler.__init__(self, h0, data, **kwargs)

        if prior_schedule is None:
            prior_schedule = ConstantSchedule(1.0)
        if likelihood_schedule is None:
            likelihood_schedule = ConstantSchedule(1.0)

        self.prior_schedule = prior_schedule
        self.likelihood_schedule = likelihood_schedule

    def next(self):
        # Just set the temperatures by the schedules
        self.prior_temperature      = self.prior_schedule.next()
        self.likelihood_temperature = self.likelihood_schedule.next()

        return MHSampler.next(self)

