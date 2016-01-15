
if __name__ == "__main__":

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--steps", dest="STEPS", type="int", default=1000, help="Number of samples to run")
    parser.add_option("--data", dest="DATA", type="int",default=300, help="Amount of data")
    (options, args) = parser.parse_args()

    from time import time

    from LOTlib.Examples.Number.Model import make_data
    data = make_data(options.DATA)

    from Model import LuaHypothesis, BASE

    start = time()
    for _ in xrange(options.STEPS):
        h = LuaHypothesis(base=BASE)
        h.compute_posterior(data)
    print "# Lua time: %s" % (time() - start)

    from LOTlib.Examples.Number.Model import NumberExpression
    from LOTlib.Examples.Number.Model import grammar as number_grammar

    start = time()
    for _ in xrange(options.STEPS):
        h = NumberExpression(grammar=number_grammar)
        h.compute_posterior(data)
    print "# Python time: %s" % (time() - start)