// Data is assumed to be "grouped", where items in the same group share a posterior
// so this does not need to be recomputed. The groups are stacked together in NYes, NTrials, and Output
// The posterior is computed on each group, and the predictive is done for the entire group.

data {

    int<lower=1> N_HYPOTHESES;                     // # of domain hypotheses
    int<lower=0> N_DATA;                           // total # of data points
    int<lower=0> N_GROUPS;                         // data is assumed to be grouped (where each group has the same posterior

    // rule counts for each hypothesis
    matrix<lower=0>[N_HYPOTHESES, 7] count_EXPR;
    matrix<lower=0>[N_HYPOTHESES, 5] count_SET;
    matrix<lower=0>[N_HYPOTHESES, 2] count_MATH;

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

    // parameters for probabilities
    simplex[7] x_EXPR;
    simplex[5] x_SET;
    simplex[2] x_MATH;
}


model {
    vector[N_HYPOTHESES] priors;
    vector[N_HYPOTHESES] posteriors;
    vector[N_HYPOTHESES] w;
    int pos;

    alpha ~ uniform(0,1);
    //print(alpha)

    // Prior on rule probs
    x_EXPR ~ dirichlet(rep_vector(1,7));
    x_SET ~ dirichlet(rep_vector(1,5));
    x_MATH ~ dirichlet(rep_vector(1,2));

    // prior for each hypothesis -- here just normalize x, even though it should be normalized per-nonterminal
    priors <- rep_vector(0.0, N_HYPOTHESES) + PriorOffset;
    priors <- priors + count_EXPR * log(x_EXPR);
    priors <- priors + count_SET * log(x_SET);
    priors <- priors + count_MATH * log(x_MATH);

    // likelihood of each human data point
    pos <- 1;
    for (g in 1:N_GROUPS) {
        posteriors <- L[g] + priors;
        w <- exp(posteriors - log_sum_exp(posteriors));            // normalize to get weights

        // save on computing the posterior here, since it will be the same as above
        // although there may be a way to do it with segments
        for(i in 1:GroupLength[g]){
            NYes[pos] ~ binomial(NTrials[pos],  (1-alpha)/2 + alpha*dot_product(w, Output[pos]));
            pos <- pos + 1;
        }
    }

}

