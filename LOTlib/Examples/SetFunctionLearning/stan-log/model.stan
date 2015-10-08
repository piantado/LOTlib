// Data is assumed to be "grouped", where items in the same group share a posterior
// so this does not need to be recomputed. The groups are stacked together in NYes, NTrials, and Output
// The posterior is computed on each group, and the predictive is done for the entire group.

data {

    int<lower=1> N_HYPOTHESES;                     // # of domain hypotheses
    int<lower=0> N_DATA;                           // total # of data points
    int<lower=0> N_GROUPS;                         // data is assumed to be grouped (where each group has the same posterior

    // rule counts for each hypothesis
    matrix<lower=0>[N_HYPOTHESES, 3] count_START;
    matrix<lower=0>[N_HYPOTHESES, 9] count_PREDICATE;
    matrix<lower=0>[N_HYPOTHESES, 4] count_BOOL;

    vector[N_HYPOTHESES] PriorOffset; // a constant prior penalty for each hypothesis

    int<lower=0>          GroupLength[N_GROUPS];             // the length of each group (used in segments below)
    vector[N_HYPOTHESES]            L[N_GROUPS];     // log likelihood of data.input

    vector[N_HYPOTHESES]       Output[N_DATA];     // each hypothesis' output to each data point

    // this is grouped by group_length chunks
    int<lower=0> NYes[N_DATA];                     // number of yes responses
    int<lower=0> NTrials[N_DATA];                  // total number of human trials per data point
}


parameters {
    // response ll parameter (NOTE: Likelihood should be sensitive to this!)
    real<lower=0,upper=1> alpha;
    real<lower=0,upper=1> baseline;

    // parameters for probabilities
    simplex[3] x_START;
    simplex[9] x_PREDICATE;
    simplex[4] x_BOOL;
}


model {
    vector[N_HYPOTHESES] priors;
    vector[N_HYPOTHESES] posteriors;
    vector[N_HYPOTHESES] w;
    int pos;

    alpha    ~ uniform(0,1); // probability of answering according to rule
    baseline ~ uniform(0,1); // When we don't answer with the rule, what's the probability of answering true
    //print(alpha)

    // Prior on rule probs
    x_START ~ dirichlet(rep_vector(1,3));
    x_PREDICATE ~ dirichlet(rep_vector(1,9));
    x_BOOL ~ dirichlet(rep_vector(1,4));

    // prior for each hypothesis -- here just normalize x, even though it should be normalized per-nonterminal
    priors <- rep_vector(0.0, N_HYPOTHESES) + PriorOffset;
    priors <- priors + count_START * log(x_START);
    priors <- priors + count_PREDICATE * log(x_PREDICATE);
    priors <- priors + count_BOOL * log(x_BOOL);

    // likelihood of each human data point
    pos <- 1;
    for (g in 1:N_GROUPS) {
        posteriors <- L[g] + priors;
        w <- exp(posteriors - log_sum_exp(posteriors));            // normalize to get weights

        // save on computing the posterior here, since it will be the same as above
        // although there may be a way to do it with segments
        for(i in 1:GroupLength[g]){
            NYes[pos] ~ binomial(NTrials[pos],  (1-alpha)*baseline + alpha*dot_product(w, Output[pos]));
            pos <- pos + 1;
        }
    }

}

