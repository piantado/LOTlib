import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons, Slider
from MCMCSummary import MCMCSummary


class VectorSummary(MCMCSummary):
    """
    Summarize & plot data for MCMC with a VectorHypothesis (e.g. GrammarHypothesis).

    """
    def __init__(self, skip=100, cap=100):
        MCMCSummary.__init__(self, skip=skip, cap=cap)

    def zip_vector(self, idxs):
        """Return a n-long list - each member is a time series of samples for a single vector item.

        In `self.samples`, we have a list of samples; basically instead of this:
            [sample1, sample2, sample3, ...]

        We want to return this:
            [[s1[0], s2[0], s3[0], ...], [s1[1], s2[1], s3[1], ...], ...]

        """
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
        gh = self.samples[-1].__copy__()
        gh.set_value(value)
        return gh

    # --------------------------------------------------------------------------------------------------------
    # Gridplot
    #
    # This doesn't work yet!  But when it does, it will hopefully be awesome and super-useful!
    #

    def gridplot(self, plot_type='violin'):
        """Plot predictive dist., rule sampling dist, & top hypotheses, all together with 1 slider.

        This slider controls which sample we are on, so we can see how things change over samples with MCMC.

        This should also work as a template so if we want to try different graphs, we can plug them in
          and use the other gridplot graphs to help us interpret & debug.

        Note:
            This has not been tested at all!  (yet)

        TODO:
          * Predictive dist.  --   p(y \in C | Hypothesis Space)
          * Top Hypotheses listed in a subplot (neatly... maybe in a table?)
          * Test everything......
          * TODO set all sub-methods to clear their axes & set title, etc.

        """
        # Numpy array of sampled values for each vector element altered in proposals
        s0 = self.samples[0]
        propose_idxs = s0.propose_idxs

        # Set up initial violinplot
        fig, ax = plt.subplots()

        # GridSpec for all our plotting stuff
        gs = gridspec.GridSpec(7, 7)
        gs.update(left=0.02, right=0.98, bottom=0.15, top=0.02)

        # Create subplots for each of our panels
        ax_bars    = plt.subplot(gs.new_subplotspec((0, 0), rowspan=3, colspan=7))
        ax_samples = plt.subplot(gs.new_subplotspec((3, 0), rowspan=4, colspan=3))
        ax_radio1  = plt.subplot(gs.new_subplotspec((3, 3), rowspan=2, colspan=1))
        ax_radio2  = plt.subplot(gs.new_subplotspec((5, 3), rowspan=2, colspan=1))
        ax_top_h   = plt.subplot(gs.new_subplotspec((3, 4), rowspan=4, colspan=3))

        # Radio button to set plot_type
        def set_plottype(plot_type):
            plot_type = plot_type
            redraw(plot_type)
        radio1 = RadioButtons(ax_radio1, ('violin', 'value', 'post', 'MLE', 'MAP'))
        radio1.on_clicked(set_plottype)

        # Radio button should set gh_key (mean, MLE, MAP, recent)
        def set_ghkey(gh_key):
            gh_key = gh_key
            redraw(self)
        radio2 = RadioButtons(ax_radio2, ('mean', 'MLE', 'MAP', 'recent'))
        radio2.on_clicked(set_ghkey)

        def redraw(idx):
            """Redraw all our plots when we move the slider or click a button."""
            draw_rulesplot(idx)
            draw_predictive(idx)
            draw_tophypo(idx)

            fig.canvas.draw_idle()  # TODO do we need both of these?
            plt.show()



        def draw_rulesplot(idx):
            """Draw violinplot or lineplot of vector samples, or a plot of a GrammarHypothesis metric.

            This does different things depending on what we set `plot_type` as in `self.gridplot`.

            """
            ax_samples.clear()

            def draw_violinplot(idx):
                vector_data = self.zip_vector(propose_idxs)
                data = [vector[0:idx] for vector in vector_data]
                y_labels = [s0.rules[i].short_str() for i in propose_idxs]
                ax_samples.set_title('Distribution of values over GrammarRules generated by MH')
                try:
                    vplot = ax_samples.violinplot(data, points=100, vert=False, widths=0.7,
                                            showmeans=True, showextrema=True, showmedians=True)
                except Exception:     # seems to get LinAlgError, ValueError when we have single-value vectors
                    vplot = None
                ax_samples.set_yticks(range(1, len(propose_idxs)+1))
                ax_samples.set_yticklabels(y_labels)
                return vplot

            def draw_lineplot(idx):
                # TODO: set X-axis here to be self.sample.count
                # TODO: only do data[0:idx]
                vector_data = self.zip_vector(propose_idxs)
                data = [vector[0:idx] for vector in vector_data]
                y_labels = [s0.rules[i].short_str() for i in propose_idxs]

                # TODO: make this divide `ax_samples` into subplots
                # N subplots sharing both x/y axes
                n = len(propose_idxs)
                ## f, AXES = plt.subplots(n, sharex=True, sharey=True)
                # ==> use `ax_samples`
                AXES[0].set_title('\tGrammar Priors as a Function of MCMC Samples')
                y_min = math.ceil(min([v for vector in vector_data for v in vector]))
                y_max = math.ceil(max([v for vector in vector_data for v in vector]))
                for i in range(n):
                    AXES[i].plot(data[i])
                    AXES[i].set_yticks(np.linspace(y_min, y_max, 5))
                    # axs[i].scatter(vector_data[i])
                    rule_label = AXES[i].twinx()
                    rule_label.set_yticks([0.5])
                    rule_label.set_yticklabels([y_labels[i]])

                ### Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
                ## f.subplots_adjust(hspace=0)
                ## plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

            def draw_ghmetric(idx, metric=plot_type):
                # TODO: only do from [0:idx]
                ax_samples.set_title(metric + ' for GrammarHypotheses Sampled by MCMC')
                if metric == 'post':
                    mcmc_values = [gh.posterior_score for gh in self.samples]
                elif metric == 'MAP':
                    mcmc_values = [gh.max_a_posteriori() for gh in self.samples]
                elif metric == 'MLE':
                    mcmc_values = [gh.max_like_estimate() for gh in self.samples]
                else:
                    mcmc_values = []
                ax_samples.plot(mcmc_values)

            if plot_type == 'violin':
                draw_violinplot(idx)
            if plot_type == 'values':
                draw_lineplot(idx)
            if plot_type in ('post', 'MLE', 'MAP'):
                draw_ghmetric(idx)


        def draw_predictive(idx):
            """Draw bar plot for  p(y \in C)."""
            ax_bars.clear()
            ax_bars.set_title('Predictive distribution  --  p(y in C | H)')

            if plot_type == 'recent':
                gh = self.samples[idx]
            if plot_type == 'mean':
                gh = samplemean(self.samples[0:idx])
            if plot_type == 'MAP':
                gh = sampleMAP(self.samples[0:idx])
            if plot_type == 'MLE':
                gh = sampleMLE(self.samples[0:idx])
            else:
                gh = None

            # TODO: compute_likelihood won't work.
            # TODO: gh won't have `domain`
            y_likelihoods = [math.exp(gh.compute_likelihood([i])) for i in range(1, gh.domain+1)]

            # TODO: histogram won't work...
            ax_bars = plt.hist(h_in_domain, bins=domain, range=(1, domain))
            ax_bars.set_yticks(range(0., 1., .2))




        redraw(self.sample_count)

        # Slider updates violinplot as a function of how many samples have been generated
        slider_ax = plt.axes([0.1, 0.1, 0.8, 0.02])
        slider = Slider(slider_ax, "after N samples", valmin=1., valmax=self.sample_count, valinit=1.)
        slider.on_changed(redraw)

        plt.show()

    # --------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------
    # Plotting methods  (this stuff is tested & it works!)
    # --------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------

    def plot(self, plot_type='violin'):
        assert plot_type in ('violin', 'values', 'post', 'MLE', 'MAP', 'barplot'), "invalid plot type!"
        if plot_type == 'violin':
            return self.violinplot_value()
        if plot_type == 'values':
            self.lineplot_value()
        if plot_type in ('post', 'MLE', 'MAP'):
            self.lineplot_gh_metric(metric=plot_type)
        if plot_type == 'barplot':
            self.plot_predictive()

    def violinplot_value(self):
        """
        TODO: doc?

        """
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

    def plot_predictive(self):
        """Visualize p(y \in C) over each y in the domain.

        This will be either the MLE GrammarHypothesis, the MAP, the most recent, or the mean averaged vector.

        This should be like the bar graphs in Josh Tenenbaum / Kevin Murphy's 'Bayesian Concept Learning'.

        Notes:
            * for now, this is only built for NumberGameHypothesis.
            * this should be 3 barplots: bayesian model averaging (weighted likelihood), MLE, & MAP
              ==> can these just be combined on 1 plot?  ==> checkbox instead of radiobutton? (future)

            * For now, just do 'recent' (not mean, MLE, MAP)

        """

        # Update the bar plots when you move the slider
        def draw_barplots(idx, plot_type):
            idxs = range(0, idx)
            if plot_type == 'recent':
                gh = self.samples[idx]
            if plot_type == 'mean':
                gh = self.mean_gh(idxs)     # TODO: does this work?????
            if plot_type == 'MAP':
                gh = self.get_top_samples(n=1, s_idxs=idxs, key=(lambda x: x.posterior_score))[0]
            if plot_type == 'MLE':
                gh = self.get_top_samples(n=1, s_idxs=idxs, key=(lambda x: x.likelihood))[0]

            domain = range(1, gh.hypotheses[0].domain+1)
            p_in_concept = gh.in_concept(domain)

            ax.clear()
            ax.set_title('Distribution of values over GrammarRules generated by MH')
            ax.bar(domain, p_in_concept)
            ax.set_yticks(range(0., 1., .2))
            fig.canvas.draw_idle()
            plt.show()

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2, left=0.1)
        draw_barplots(self.sample_count, plot_type='recent')

        # Slider updates violinplot as a function of how many samples have been generated
        slider_ax = plt.axes([0.1, 0.1, 0.8, 0.02])
        slider = Slider(slider_ax, "after N samples",
                        valmin=1., valmax=self.sample_count, valinit=self.sample_count)
        slider.on_changed(draw_barplots)
        plt.show()



