"""
    Run MDS on some hypotheses, taking the proposal probability as the distance metric.

    This is one way to visualize the space.
"""

import numpy
import matplotlib
from sklearn.manifold import MDS
from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal # Our distance metric will be the regeneration proposal distance

def compute_similarity_matrix(hyps):
    """ Take N hyps and return an NxN similarity matrix corresponding to how likely a proposal is to go from hyps[i]->hyps[j] """
    N = len(hyps)
    RP = RegenerationProposal(grammar)
    sim = numpy.zeros((N,N), dtype=numpy.float32)
    posts = numpy.zeros(N) # store the posterior to plot a color

    for i in xrange(N):
        posts[i] = hyps[i].posterior_score
        for j in xrange(i, N):
            # For some architectures, we get a weird precision error with MDS, we must round here and specify float32 above
            sim[i][j] = numpy.round(RP.lp_propose(hyps[i].value, hyps[j].value) + RP.lp_propose(hyps[j].value, hyps[i].value), decimals=5) # must be similarities
            sim[j][i] = sim[i][j] # force symmetric!

    return sim


def compute_MDS_positions(sim, **kwargs):
    """ Run MDS on a similarity matrix. Converts it to distance via negative"""

    m = MDS(dissimilarity="precomputed")

    pos = m.fit_transform(-sim, **kwargs)

    return pos



if __name__ == "__main__":
    """
        Show an example of this on the number model
    """

    import pickle
    from LOTlib.Examples.Number.Model import *
    from LOTlib.Miscellaneous import logsumexp, logplusexp, Infinity
    from LOTlib.MCMCSummary.TopN import TopN

    NDATA = 300 # We keep hypotheses based on their rank here

    data = generate_data(NDATA)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print "# Loading"
    # with open('../Examples/Number/output/out.pkl', 'r') as f:
    with open('../../Examples/Number/mpi-run-mds.pkl', 'r') as f:
    # with open('../Examples/Number/mpi-run.pkl', 'r') as f:
        hyps = tuple(pickle.load(f))
        print "# Loaded %s hypotheses" % len(hyps)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print "# Computing similarity matrix"

    sim = compute_similarity_matrix(hyps)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print "# Computing MDS"

    positions = compute_MDS_positions(sim)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot for different amounts of data

    def code_knower_level(h):
        """ Recode knower-levels to shapes    """
        kl = get_knower_pattern(h)
        if   kl == 'UUUUUUUUU':    return '$\mathbf{U}$' #'*'
        elif kl == '1UUUUUUUU':    return '$\mathbf{1}$' #"."
        elif kl == '12UUUUUUU':    return '$\mathbf{2}$' # "|"
        elif kl == '123UUUUUU':    return '$\mathbf{3}$' #"^"
        elif kl == '123456789':    return '$\mathbf{9}$' # "o"
        else:                      return '.' #"x"

    import matplotlib.cm as cm
    import matplotlib.pyplot as pyplot
    markers = map(code_knower_level, hyps)

    # Make plots for various amounts of data
    for di in [0,50,100,150,200]:
        data = generate_data(di)
        posts = numpy.array([ sum(h.compute_posterior(data)) for h in hyps])
        Z = logsumexp(posts)

        print "# Plotting ", di
        fig = pyplot.figure()
        plt = fig.add_subplot(111)

        # Do a plot of the numbers/digits, colored by posterior
        for i in xrange(len(hyps)):
            plt.scatter(positions[i,0], positions[i,1], edgecolor='none',  marker=markers[i], c=(posts[i] - Z), vmin=-10, vmax=0)
        fig.savefig('plot-%s.pdf'%di)
        fig.clf()

        ## Do a plot of the posterior probability mass
        bins = 100
        xs = numpy.linspace( min(positions[:,0]), max(positions[:,0]), num=bins)
        ys = numpy.linspace( min(positions[:,1]), max(positions[:,1]), num=bins)
        z = numpy.zeros( (bins+1,bins+1) ) - Infinity
        for i in xrange(len(hyps)):
            x = min(numpy.flatnonzero(positions[i,0] <= xs)) # put into bins
            y = min(numpy.flatnonzero(positions[i,1] <= ys))
            z[x][y] = logplusexp(z[x][y], posts[i]-Z)
        plt = fig.add_subplot(111)
        cax = plt.imshow(z)
        fig.colorbar(cax)
        fig.savefig('density-%s.pdf'%di)
        fig.clf()