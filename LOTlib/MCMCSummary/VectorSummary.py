import csv, math
import numpy as np
import pickle
from MCMCSummary import MCMCSummary


class VectorSummary(MCMCSummary):
    """
    Summarize & plot data for MCMC with a VectorHypothesis (e.g. GrammarHypothesis).

    """
    def __init__(self, skip=100, cap=100):
        MCMCSummary.__init__(self, skip=skip, cap=cap)

    def zip_vector(self, idxs):
        """Return a list of time series of samples for specified vector indexes."""
        zipped_vector = zip(*[[s.value[i] for i in idxs] for s in self.samples])
        zipped_vector = [np.array(l) for l in zipped_vector]
        return zipped_vector

    def median_value(self, idxs=None):
        """Return a vector for the median of each value item accross `self.samples`, items in `idxs`."""
        if idxs is None:
            idxs = range(1, self.samples[0].n)
        vector_data = self.zip_vector(range(1, idxs))
        return [np.mean(v) for v in vector_data]

    def mean_value(self, idxs=None):
        """Return a vector for the mean of each value item accross `self.samples`, items in `idxs`."""
        if idxs is None:
            idxs = range(1, self.samples[0].n)
        vector_data = self.zip_vector(idxs)
        return [np.mean(v) for v in vector_data]

    def mean_gh(self, idxs=None):
        value = self.mean_value(idxs)
        gh = self.samples[idxs[-1]].__copy__()
        gh.set_value(value)
        gh.update_posterior()
        return gh

    # --------------------------------------------------------------------------------------------------------
    # Saving methods

    def pickle_cursample(self, filename):
        with open(filename, 'a') as f:
            gh = self.samples[-1]
            pickle.dump(gh.value, f)

    def pickle_MAPsample(self, filename):
        with open(filename, 'a') as f:
            gh = self.get_top_samples(1)[0]
            pickle.dump(gh.value, f)

    def csv_initfiles(self, filename):
        """
        Initialize new csv files.

        """
        with open(filename+'_values_recent.csv', 'a') as w:
            writer = csv.writer(w)
            writer.writerow(['i', 'nt', 'name', 'to', 'p'])
        with open(filename+'_bayes_recent.csv', 'a') as w:
            writer = csv.writer(w)
            writer.writerow(['i', 'Prior', 'Likelihood', 'Posterior Score'])
        with open(filename+'_values_map.csv', 'a') as w:
            writer = csv.writer(w)
            writer.writerow(['i', 'nt', 'name', 'to', 'p'])
        with open(filename+'_bayes_map.csv', 'a') as w:
            writer = csv.writer(w)
            writer.writerow(['i', 'Prior', 'Likelihood', 'Posterior Score'])

    def csv_appendfiles(self, filename, data):
        """
        Append Bayes data to `_bayes` file, values to `_values` file, and MAP hypothesis human
        correlation data to `_data_MAP` file.

        """
        i = self.count
        gh_recent = self.samples[-1]
        gh_map = self.get_top_samples(1)[0]

        with open(filename+'_values_recent.csv', 'a') as w:
            writer = csv.writer(w)
            writer.writerows([[i, r.nt, r.name, str(r.to), gh_recent.value[j]] for j,r in enumerate(gh_recent.rules)])
        with open(filename+'_bayes_recent.csv', 'a') as w:
            writer = csv.writer(w)
            if self.sample_count:
                writer.writerow([i, gh_recent.prior, gh_recent.likelihood, gh_recent.posterior_score])
        with open(filename+'_values_map.csv', 'a') as w:
            writer = csv.writer(w)
            writer.writerows([[i, r.nt, r.name, str(r.to), gh_map.value[j]] for j,r in enumerate(gh_map.rules)])
        with open(filename+'_bayes_map.csv', 'a') as w:
            writer = csv.writer(w)
            if self.sample_count:
                writer.writerow([i, gh_map.prior, gh_map.likelihood, gh_map.posterior_score])

    # --------------------------------------------------------------------------------------------------------
    # Plotting methods

    def plot(self, plot_type='violin'):
        assert plot_type in ('violin', 'values', 'post', 'MLE', 'MAP', 'barplot'), "invalid plot type!"
        if plot_type == 'violin':
            return self.violinplot_value()
        if plot_type == 'values':
            self.lineplot_value()
        if plot_type in ('post', 'MLE', 'MAP'):
            self.lineplot_gh_metric(metric=plot_type)

    def violinplot_value(self):
        """
        TODO: doc?

        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, Slider

        # Numpy array of sampled values for each vector element altered in proposals
        s0 = self.samples[0]
        propose_idxs = s0.propose_idxs

        def draw_violinplot(value):
            """Clear axis & draw a labelled violin plot of the specified data.

            Note:
              * [fixed] If we haven't accepted any proposals yet, all our data is the same and this causes
                a singular matrix 'LinAlgError'

            """
            vector_data = self.zip_vector(propose_idxs)
            data = [vector[0:value] for vector in vector_data]

            ax.clear()
            ax.set_title('Distribution of values over GrammarRules generated by MH')
            try:
                vplot = ax.violinplot(data, points=100, vert=False, widths=0.7,
                                      showmeans=True, showextrema=True, showmedians=True)
            except Exception:     # seems to get LinAlgError, ValueError when we have single-value vectors
                vplot = None
            ax.set_yticks(range(1, len(propose_idxs)+1))
            y_labels = [s0.rules[i].short_str() for i in propose_idxs]
            ax.set_yticklabels(y_labels)

            fig.canvas.draw_idle()
            return vplot

        # Set up initial violinplot
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2, left=0.1)
        violin_stats = draw_violinplot(self.sample_count)

        # Slider updates violinplot as a function of how many samples have been generated
        slider_ax = plt.axes([0.1, 0.1, 0.8, 0.02])
        slider = Slider(slider_ax, "after N samples", valmin=1., valmax=self.sample_count, valinit=1.)
        slider.on_changed(draw_violinplot)

        plt.show()
        return violin_stats

    def lineplot_value(self):
        """
        http://matplotlib.org/examples/pylab_examples/subplots_demo.html

        """
        import matplotlib.pyplot as plt

        # Numpy array of sampled values for each vector element altered in proposals
        s0 = self.samples[0]
        propose_idxs = s0.propose_idxs
        n = len(propose_idxs)
        y_labels = [s0.rules[i].short_str() for i in propose_idxs]
        vector_data = self.zip_vector(propose_idxs)

        # N subplots sharing both x/y axes
        f, axs = plt.subplots(n, sharex=True, sharey=True)
        axs[0].set_title('\tGrammar Priors as a Function of MCMC Samples')
        y_min = math.ceil(min([v for vector in vector_data for v in vector]))
        y_max = math.ceil(max([v for vector in vector_data for v in vector]))
        for i in range(n):
            axs[i].plot(vector_data[i])
            axs[i].set_yticks(np.linspace(y_min, y_max, 5))
            # axs[i].scatter(vector_data[i])
            rule_label = axs[i].twinx()
            rule_label.set_yticks([0.5])
            rule_label.set_yticklabels([y_labels[i]])

        # Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.show()

    def lineplot_gh_metric(self, metric='post'):
        """
        Draw a line plot for the GrammarHypothesis, evaluated by GH.posterior_score, MAP, or MLE.

        """
        import matplotlib.pyplot as plt

        assert metric in ('post', 'MLE', 'MAP'), "invalid plot metric!"
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2, left=0.1)
        ax.set_title('Evaluation for GrammarHypotheses Sampled by MCMC')

        if metric == 'post':
            mcmc_values = [gh.posterior_score for gh in self.samples]
        elif metric == 'MAP':
            mcmc_values = [gh.max_a_posteriori() for gh in self.samples]
        elif metric == 'MLE':
            mcmc_values = [gh.max_like_estimate() for gh in self.samples]
        else:
            mcmc_values = []
        ax.plot(mcmc_values)
        plt.show()


