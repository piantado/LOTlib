
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We define two variables, one for how many function calls have been
# used in a single function/hypothesis, and one for how many have been
# run over the entire course of the experiment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LOCAL_PRIMITIVE_OPS = 0
GLOBAL_PRIMITIVE_OPS = 0


def LOTlib_primitive(fn):
    """A decorator for basic primitives that increments our counters."""
    def inside(*args, **kwargs):

        global LOCAL_PRIMITIVE_OPS
        LOCAL_PRIMITIVE_OPS += 1

        global GLOBAL_PRIMITIVE_OPS
        GLOBAL_PRIMITIVE_OPS += 1

        return fn(*args, **kwargs)

    return inside


def None2None(fn):
    """
    A decorator to map anything with "None" as a *list* arg (NOT a keyword arg)
    this will make it return None overall

    If you want to have this not prevent incrementing (via LOTlib_primitive), then
    we need to put it *after* LOTlib_primitive:

    @None2None
    def f(...):

    """
    def inside(*args, **kwargs):
        if any([a is None for a in args]): return None
        return fn(*args, **kwargs)

    return inside
