#
# Script to to generate model correlation plots for MAP & h0 of loaded GrammarHypothesis csv.
#
# If model is 'lot' or 'indep', then we also make a violin plot with appropriate x-axis labels.
#
#
# Arguments
# ---------
# 1: model
#   Specify which model we're using.   [ 'mix' | 'indep' | 'lot' ]
# 2: name
#   Specify the filename of the csv we're loading; png & eps files will also be saved using this name. Do
#   not include '.csv' at the end of the filename.  (e.g. 'out/gh_lot100k')
#
#
# Example
# -------
# $ Rscript CSVtoFigures.R lot out/gh_lot100k
#
#

args <- commandArgs(trailingOnly = TRUE)
model <- args[1]        # model  [ 'mix' | 'indep' | 'lot' ]
name <- args[2]         # filename, e.g. 'gh_lot100k'


# ------------------------------------------------------------------------------------------------------------
# Model correlation plots (MAP)

f <- paste(name, "_data_MAP", sep="")
d <- read.csv(paste(f, ".csv", sep=""), header=T)

m <- sort(unique(d$i), decreasing=TRUE)[3]
d <- subset(d, i==m)
postscript(paste(f, ".eps", sep=""), height=4, width=4)

plot(jitter(d$model.p), jitter(d$human.p), pch="+", xlab="Model Predictive Probability", ylab="Human Predictive Probability", ylim=c(0,1), xlim=c(0,1))
#    points(0:1, 0:1, lty="dotted", col="#000099")
rsq <- cor.test(d$model.p, d$human.p)$estimate
text( 0.85, 0.05, substitute(R^2 == r, list(r=round(rsq,3))), col=4)

dev.off()

# ------------------------------------------------------------------------------------------------------------
# Model correlation plots (h0)

f <- paste(name, "_data_h0", sep="")
d <- read.csv(paste(f, ".csv", sep=""), header=T)
postscript(paste(f, ".eps", sep=""), height=4, width=4)

plot(jitter(d$model.p), jitter(d$human.p), pch="+", xlab="Model Predictive Probability", ylab="Human Predictive Probability", ylim=c(0,1), xlim=c(0,1))
# main="Independent Model (Initial Parameters)")
#    abline(1, 0, lty="dotted")
# points(0:1, 0:1, lty="dotted", col="#000099")       # blue line
rsq <- cor.test(d$model.p, d$human.p)$estimate
text( 0.85, 0.05, substitute(R^2 == r, list(r=round(rsq,3))), col=4)

dev.off()

# ------------------------------------------------------------------------------------------------------------

library(ggplot2)

if (model=='mix') {
    f <- paste(name, "_values", sep="")
    d <- read.csv(paste(f, ".csv", sep=""), header=T)
    mth <- subset(d, to=="MATH")
    smlr <- subset(d, to=="INTERVAL")
    q <- mth / (smlr + mth)



#    f <- paste(name, "_values", sep="")
#    d <- read.csv(paste(f, ".csv", sep=""), header=T)
#
#    q <- subset(d, nt=="START")
#    m <- sort(unique(d$i), decreasing=TRUE)[3]
#    q <- subset(q, i>m/2)       # Second half of samples
#    q <- subset(q, i<m)
#    q$x <- paste(q$name, q$to)
#
#    plt <- ggplot(q, aes(x=x, y=p)) +
#            geom_boxplot() +
##            geom_violin(fill = "red", scale="width") +
#            ylab("Probability") + xlab("PCFG Parameter") +
#            theme(axis.text.x = element_text(angle = 90, hjust = 1))
#    ggsave(paste(f, ".eps", sep=""), plt, heigh=4, width=7)

} else if (model=='indep') {
    # --------------------------------------------------------------------------------------------------------
    # Independent probabilities model
    # --------------------------------------------------------------------------------------------------------

    f <- paste(name, "_values", sep="")
    d <- read.csv(paste(f, ".csv", sep=""), header=T)

    q <- subset(d, nt=="EXPR")
    m <- sort(unique(d$i), decreasing=TRUE)[3]
    q <- subset(q, i>m/2)       # Second half of samples
    q <- subset(q, i<m)
    q$x <- paste(q$name, q$to)

    plt <- ggplot(q, aes(x=x, y=p)) +
            # geom_boxplot() +
            geom_violin(fill = "yellow", scale="width") +
            ylab("Probability") + xlab("PCFG Parameter") +
            theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
            scale_x_discrete(
                breaks=c(
                    "ends_in_ ['X', '0']", "ends_in_ ['X', '1']", "ends_in_ ['X', '2']", "ends_in_ ['X', '3']", "ends_in_ ['X', '4']", "ends_in_ ['X', '5']", "ends_in_ ['X', '6']", "ends_in_ ['X', '7']", "ends_in_ ['X', '8']", "ends_in_ ['X', '9']",
                    "times_ ['X', '2']", "times_ ['X', '3']", "times_ ['X', '4']", "times_ ['X', '5']", "times_ ['X', '6']", "times_ ['X', '7']", "times_ ['X', '8']", "times_ ['X', '9']", "times_ ['X', '10']", "times_ ['X', '11']", "times_ ['X', '12']",
                    "ipowf_ ['2', 'X']", "ipowf_ ['3', 'X']", "ipowf_ ['4', 'X']", "ipowf_ ['5', 'X']", "ipowf_ ['6', 'X']", "ipowf_ ['7', 'X']", "ipowf_ ['8', 'X']", "ipowf_ ['9', 'X']", "ipowf_ ['10', 'X']",
                    "ipowf_ ['X', '2']", "ipowf_ ['X', '3']", "isprime_ ['X']", "plus_ ['ODD', '1']", "pow2n_d32_ ['X']", "pow2n_u37_ ['X']"
                    ),
                labels=c(
                    "Ends in 0", "Ends in 1", "Ends in 2", "Ends in 3", "Ends in 4", "Ends in 5", "Ends in 6", "Ends in 7", "Ends in 8", "Ends in 9",
                    "Even Numbers", "Multiples of 3", "Multiples of 4", "Multiples of 5", "Multiples of 6", "Multiples of 7", "Multiples of 8", "Multiples of 9", "Multiples of 10", "Multiples of 11", "Multiples of 12",
                    "Powers of 2", "Powers of 3", "Powers of 4", "Powers of 5", "Powers of 6", "Powers of 7", "Powers of 8", "Powers of 9", "Powers of 10",
                    "Squares", "Cubes", "Primes", "Odd Numbers", "Powers of 2, not 32", "Powers of 2, or 37"),
                limits=c(
                    "ends_in_ ['X', '0']", "ends_in_ ['X', '1']", "ends_in_ ['X', '2']", "ends_in_ ['X', '3']", "ends_in_ ['X', '4']", "ends_in_ ['X', '5']", "ends_in_ ['X', '6']", "ends_in_ ['X', '7']", "ends_in_ ['X', '8']", "ends_in_ ['X', '9']",
                    "ipowf_ ['X', '2']", "ipowf_ ['X', '3']", "isprime_ ['X']", "plus_ ['ODD', '1']",
                    "times_ ['X', '2']", "times_ ['X', '3']", "times_ ['X', '4']", "times_ ['X', '5']", "times_ ['X', '6']", "times_ ['X', '7']", "times_ ['X', '8']", "times_ ['X', '9']", "times_ ['X', '10']", "times_ ['X', '11']", "times_ ['X', '12']",
                    "ipowf_ ['2', 'X']", "ipowf_ ['3', 'X']", "ipowf_ ['4', 'X']", "ipowf_ ['5', 'X']", "ipowf_ ['6', 'X']", "ipowf_ ['7', 'X']", "ipowf_ ['8', 'X']", "ipowf_ ['9', 'X']", "ipowf_ ['10', 'X']",
                    "pow2n_d32_ ['X']", "pow2n_u37_ ['X']"
                    ))
    ggsave(paste(f, ".eps", sep=""), plt, heigh=4, width=7)

} else if (model=='lot') {
    # --------------------------------------------------------------------------------------------------------
    # LOT / compositional model
    # --------------------------------------------------------------------------------------------------------

    f <- paste(name, "_values", sep="")
    d <- read.csv(paste(f, ".csv", sep=""), header=T)

    q <- subset(d, (nt=="EXPR")|(nt=="OPCONST"))
    m <- sort(unique(d$i), decreasing=TRUE)[3]
    q <- subset(q, i>m/2)       # Second half of samples
    q <- subset(q, i<m)
    q$x <- paste(q$name, q$to)

    plt <- ggplot(q, aes(x=x, y=p)) +
             # geom_boxplot() +
             geom_violin(fill="green", scale="width") +
             ylab("Probability") + xlab("PCFG Parameter") +
             theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
             scale_x_discrete(
                 breaks=c(
                     " ['1']", " ['2']", " ['3']", " ['4']", " ['5']", " ['6']", " ['7']", " ['8']", " ['9']", " ['10']",
                     " ['OPCONST']",
                     "ends_in_ ['EXPR', 'EXPR']",
                     "ipowf_ ['EXPR', 'EXPR']",
                     "isprime_ ['EXPR']",
                     "plus_ ['EXPR', 'EXPR']",
                     "times_ ['EXPR', 'EXPR']"
                     ),
                 labels=c(
                     "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                     "Constant",
                     "Ends in",
                     "^",
                     "Prime",
                     "+",
                     "*"),
                 limits=c(
                     "ends_in_ ['EXPR', 'EXPR']",
                     "isprime_ ['EXPR']",
                     "ipowf_ ['EXPR', 'EXPR']",
                     "plus_ ['EXPR', 'EXPR']",
                     "times_ ['EXPR', 'EXPR']",
                     " ['OPCONST']",
                     " ['1']", " ['2']", " ['3']", " ['4']", " ['5']", " ['6']", " ['7']", " ['8']", " ['9']", " ['10']"
                     ))
    ggsave(paste(f, ".eps", sep=""), plt, height=4, width=7)
}












