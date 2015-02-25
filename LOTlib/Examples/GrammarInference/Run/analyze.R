



csvs <- c("lot_5mil_1", "lot_5mil_2", "lot_5mil_3", "lot_5mil_a85", "lot_5mil_a95", "lot_5mil_a95_2")
# csvs <- c("individual_1mil_1", "individual_1mil_2", "individual_1mil_3", "individual_1mil_a85", "individual_1mil_a95", "individual_1mil_a95_2")

for (name in csvs){

    # ---------------------------------------------------------------------------------------------------
    # Model correlation plots (MAP)
    
    f <- paste(name, "_data_MAP", sep="")
    d <- read.csv(paste(f, ".csv", sep=""), header=T)

    m <- sort(unique(d$i), decreasing=TRUE)[3]
    d <- subset(d, i==m)
    postscript(paste(f, ".eps", sep=""), height=4, width=4) 
    
    plot(jitter(d$model.p), jitter(d$human.p), pch="+", xlab="Model Predictive Probability", ylab="Human Predictive Probability", ylim=c(0,1), xlim=c(0,1))
    points(0:1, 0:1, lty="dotted", col="#000099")
    rsq <- cor.test(d$model.p, d$human.p)$estimate
    text( 0.85, 0.05, substitute(R^2 == r, list(r=round(rsq,3))), col=4)

    dev.off()

    # ---------------------------------------------------------------------------------------------------
    # Model correlation plots (h0)
    
    f <- paste(name, "_data_h0", sep="")
    d <- read.csv(paste(f, ".csv", sep=""), header=T)
    postscript(paste(f, ".eps", sep=""), height=4, width=4) 
    
    plot(jitter(d$model.p), jitter(d$human.p), pch="+", xlab="Model Predictive Probability", ylab="Human Predictive Probability", ylim=c(0,1), xlim=c(0,1)) 
        # main="Independent Model (Initial Parameters)")
    # abline(1, 0, lty="dotted")
    points(0:1, 0:1, lty="dotted", col="#000099")       # blue line
    rsq <- cor.test(d$model.p, d$human.p)$estimate
    text( 0.85, 0.05, substitute(R^2 == r, list(r=round(rsq,3))), col=4)

    dev.off()


    # # ---------------------------------------------------------------------------------------------------
    # # ---------------------------------------------------------------------------------------------------
    # # FOR INDEPENDENT PROBABILITIES MODEL
    # # ---------------------------------------------------------------------------------------------------

    # library(ggplot2)

    # f <- paste(name, "_values", sep="")
    # d <- read.csv(paste(f, ".csv", sep=""), header=T)
    # q <- subset(d, nt=="OPCONST")
    # m <- sort(unique(d$i), decreasing=TRUE)[3]
    # q <- subset(q, i>m/2)       # Second half of samples
    # q <- subset(q, i<m)

    # q$to <- as.numeric(gsub("[^0-9]+", "", as.character(q$to)))

    # plt <- ggplot(q, aes(x=as.factor(to), y=p)) +
    #         geom_boxplot() +                     
    #         ylab("Probability") + xlab("PCFG Parameter")
    #         # theme(axis.text.x = element_text(angle = 90, hjust = 1))   

    # ggsave(paste(name, "_opconst.eps", sep=""), plt, heigh=3, width=15)


    # # ---------------------------------------------------------------------------------------------------
    # # Plot the different functions

    # q <- subset(d, nt=="EXPR") 
    # m <- sort(unique(d$i), decreasing=TRUE)[3]
    # q <- subset(q, i>m/2)       # Second half of samples
    # q <- subset(q, i<m)
    # q$x <- paste(q$name, q$to) 

    # plt <- ggplot(q, aes(x=x, y=p)) +
    #         geom_boxplot() +
    #         ylab("Probability") + xlab("PCFG Parameter") +
    #         theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    #         scale_x_discrete(
    #             breaks=c(
    #                 "ends_in_ ['X', '0']", "ends_in_ ['X', '1']", "ends_in_ ['X', '2']", "ends_in_ ['X', '3']", "ends_in_ ['X', '4']", "ends_in_ ['X', '5']", "ends_in_ ['X', '6']", "ends_in_ ['X', '7']", "ends_in_ ['X', '8']", "ends_in_ ['X', '9']", 
    #                 "times_ ['X', '2']", "times_ ['X', '3']", "times_ ['X', '4']", "times_ ['X', '5']", "times_ ['X', '6']", "times_ ['X', '7']", "times_ ['X', '8']", "times_ ['X', '9']", "times_ ['X', '10']", "times_ ['X', '11']", "times_ ['X', '12']", 
    #                 "ipowf_ ['2', 'X']", "ipowf_ ['3', 'X']", "ipowf_ ['4', 'X']", "ipowf_ ['5', 'X']", "ipowf_ ['6', 'X']", "ipowf_ ['7', 'X']", "ipowf_ ['8', 'X']", "ipowf_ ['9', 'X']", "ipowf_ ['10', 'X']", 
    #                 "ipowf_ ['X', '2']", "ipowf_ ['X', '3']", "isprime_ ['X']", "plus_ ['ODD', '1']", "pow2n_d32_ ['X']", "pow2n_u37_ ['X']"
    #                 ), 
    #             labels=c(
    #                 "Ends in 0", "Ends in 1", "Ends in 2", "Ends in 3", "Ends in 4", "Ends in 5", "Ends in 6", "Ends in 7", "Ends in 8", "Ends in 9",
    #                 "Even Numbers", "Multiples of 3", "Multiples of 4", "Multiples of 5", "Multiples of 6", "Multiples of 7", "Multiples of 8", "Multiples of 9", "Multiples of 10", "Multiples of 11", "Multiples of 12", 
    #                 "Powers of 2", "Powers of 3", "Powers of 4", "Powers of 5", "Powers of 6", "Powers of 7", "Powers of 8", "Powers of 9", "Powers of 10",
    #                 "Squares", "Cubes", "Primes", "Odd Numbers", "Powers of 2, not 32", "Powers of 2, or 37"),
    #             limits=c(
    #                 "ends_in_ ['X', '0']", "ends_in_ ['X', '1']", "ends_in_ ['X', '2']", "ends_in_ ['X', '3']", "ends_in_ ['X', '4']", "ends_in_ ['X', '5']", "ends_in_ ['X', '6']", "ends_in_ ['X', '7']", "ends_in_ ['X', '8']", "ends_in_ ['X', '9']", 
    #                 "ipowf_ ['X', '2']", "ipowf_ ['X', '3']", "isprime_ ['X']", "plus_ ['ODD', '1']",
    #                 "times_ ['X', '2']", "times_ ['X', '3']", "times_ ['X', '4']", "times_ ['X', '5']", "times_ ['X', '6']", "times_ ['X', '7']", "times_ ['X', '8']", "times_ ['X', '9']", "times_ ['X', '10']", "times_ ['X', '11']", "times_ ['X', '12']", 
    #                 "ipowf_ ['2', 'X']", "ipowf_ ['3', 'X']", "ipowf_ ['4', 'X']", "ipowf_ ['5', 'X']", "ipowf_ ['6', 'X']", "ipowf_ ['7', 'X']", "ipowf_ ['8', 'X']", "ipowf_ ['9', 'X']", "ipowf_ ['10', 'X']", 
    #                 "pow2n_d32_ ['X']", "pow2n_u37_ ['X']"
    #                 ))

    # ggsave(paste(name, "_expr.eps", sep=""), plt, heigh=4, width=7)


    # ---------------------------------------------------------------------------------------------------
    # FOR LOT MODEL
    # ---------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------      

    library(ggplot2)

    f <- paste(name, "_values", sep="")
    d <- read.csv(paste(f, ".csv", sep=""), header=T)

    q <- subset(d, (nt=="EXPR")|(nt=="OPCONST")) 
    m <- sort(unique(d$i), decreasing=TRUE)[3]
    q <- subset(q, i>m/2)       # Second half of samples
    q <- subset(q, i<m)
    q$x <- paste(q$name, q$to) 

    plt <- ggplot(q, aes(x=x, y=p)) +
            geom_boxplot() +
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

    ggsave(paste(name, "_values.eps", sep=""), plt, heigh=4, width=7)


    # ---------------------------------------------------------------------------------------------------      


}







# d <- read.csv('individual_500000.csv')

# # par(mfrow=c(10,10))
# # for(i in 1:100) {
# #         hist(d[,i])
# # }

# means <- apply(d, 2, mean)
# sds   <- apply(d, 2, sd)
# mins   <- apply(d, 2, min)
# maxes   <- apply(d, 2, max)

# late.means <- apply(d[(3*nrow(d)/4):nrow(d),], 2, mean)


# library(Hmisc)
# plot(means[1:100])
# errbar(1:100, means[1:100], mins[1:100], maxes[1:100])
# points(1:100, late.means[1:100], col=2)



#  sum([1356,1679,1204,1755,1675,1247,1683,1774,1318,1553])/10





###########################################################################
# Posteriors
###########################################################################

d <- read.csv('individual_1000000_1_bayes.csv')
postscript("individual_1000000_1_posterior.eps", height=4, width=4) 

plot(d$Posterior.Score, ylab="Sample #", xlab="Posterior Score", main="Posterior Score for individual model over 1mil iters")

dev.off()





###########################################################################
# Model correlation plots
###########################################################################

d <- read.csv("lot_5mil_3_data_MAP.csv", header=T)

# Everything is sensitive to the height and width, which set the overall scaling. I think this looks pretty good. 
# Check out the options to par(..) in order to adjust the axis labels and title to be a bit closer
postscript("lot_map.eps", height=4, width=4) 

# Just a simple plot for these
plot(d$model.p, d$human.p, pch="+", xlab="Model predicted probability", ylab="Human probability", ylim=c(0,1), xlim=c(0,1), main="LOT Model MAP Predictions (500k iters, .9 alpha)")

rsq <- cor.test(d$model.p, d$human.p)$estimate # compute a correlation
text( 0.85, 0.05, substitute(R^2 == r, list(r=round(rsq,3))), col=4) # and put it in the corner; col=4 is blue

dev.off() # Close postscript output device





###########################################################################
# Plot the CONSTs
###########################################################################

library(ggplot2) # we'll use ggplot to make things easier

d <- read.csv("lot_30k_2_0_values.csv", header=T)

q <- subset(d, nt=="CONST") # Who to plot?

# Here I throw out the first half of samples, only keeping those after convergence (we must check that this is true)
m <- max(q$i)               # Number of samples
q <- subset(q, i>m/2)       # Second half of samples

# Convert "to" to numbers for q, so that it sorts and prints nicely
q$to <- as.numeric(gsub("[^0-9]+", "", as.character(q$to)))

plt <- ggplot(q, aes(x=as.factor(to), y=p)) +
        geom_boxplot() +                        # the beauty of ggplot
        xlab("") + ylab("PCFG Parameter") +
        theme(axis.text.x = element_text(angle = 90, hjust = 1))    # rotate the angles

# Typing "plt" in the terminal will now plot it

ggsave("lot_30k_2_0_CONST.eps", plt, heigh=3, width=15)







###########################################################################
# Plot the different functions
###########################################################################

library(ggplot2) # we'll use ggplot to make things easier

d <- read.csv("lot_30k_2_0_values.csv", header=T)

# Only plot the Expressions
q <- subset(d, nt=="EXPR") 

# Only use second half of samples
m <- max(q$i)
q <- subset(q, i>m/2)

# Make the thing we plot
q$x <- paste(q$name, q$to) 

plt <- ggplot(q, aes(x=x, y=p)) +
        geom_boxplot() +
        xlab("Probability") + ylab("PCFG Parameter") +
        theme(axis.text.x = element_text(angle = 90, hjust = 1)) + # Rotate the angles

       

ggsave("lot_30k_2_0_operators.eps", plt, heigh=4, width=7)





# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------



    xlab("Probability") + ylab("PCFG Parameter") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) + # Rotate the angles



# ends_in_ ['X', '#']   
# ipowf_ ['#', 'X']
# ipowf_ ['X', '2']
# ipowf_ ['X', '3']
# isprime_ ['X'] 
# plus_ ['ODD ','1']
# pow2n_d32_ ['X'] 
# pow2n_u37_ ['X'] 
# times_ ['X', '#']   --> 2-12

scale_x_discrete(
    breaks=c(
        "ends_in_ ['X', '0']", "ends_in_ ['X', '1']", "ends_in_ ['X', '2']", "ends_in_ ['X', '3']", "ends_in_ ['X', '4']", "ends_in_ ['X', '5']", "ends_in_ ['X', '6']", "ends_in_ ['X', '7']", "ends_in_ ['X', '8']", "ends_in_ ['X', '9']", 
        "times_ ['X', '2']", "times_ ['X', '3']", "times_ ['X', '4']", "times_ ['X', '5']", "times_ ['X', '6']", "times_ ['X', '7']", "times_ ['X', '8']", "times_ ['X', '9']", "times_ ['X', '10']", "times_ ['X', '11']", "times_ ['X', '12']", 
        "ipowf_ ['2', 'X']", "ipowf_ ['3', 'X']", "ipowf_ ['4', 'X']", "ipowf_ ['5', 'X']", "ipowf_ ['6', 'X']", "ipowf_ ['7', 'X']", "ipowf_ ['8', 'X']", "ipowf_ ['9', 'X']", "ipowf_ ['10' ,'X']", 
        "ipowf_ ['X', '2']", "ipowf_ ['X', '3']", "isprime_ ['X']", "plus_ ['ODD ','1']", "pow2n_d32_ ['X']", "pow2n_u37_ ['X']"
        ), 
    labels=c(
        "Ends in 0", "Ends in 1", "Ends in 2", "Ends in 3", "Ends in 4", "Ends in 5", "Ends in 6", "Ends in 7", "Ends in 8", "Ends in 9",
        "Even Numbers", "Multiples of 3", "Multiples of 4", "Multiples of 5", "Multiples of 6", "Multiples of 7", "Multiples of 8", "Multiples of 9", "Multiples of 10", "Multiples of 11", "Multiples of 12", 
        "Powers of 2", "Powers of 3", "Powers of 4", "Powers of 5", "Powers of 6", "Powers of 7", "Powers of 8", "Powers of 9", "Powers of 10",
        "Squares", "Cubes", "Primes", "Odd Numbers", "Powers of 2 Except 32", "Powers of 2, or 37"))



# ['OPCONST']
# ends_in_ ['EXP R','EXPR']
# ipowf_ ['EXP R','EXPR']
# isprime_ ['EXP R']
# plus_ ['EXP R','EXPR']
# times_ ['EXP R','EXPR']

scale_x_discrete(
    breaks=c(
        "['OPCONST']"
        "ends_in_ ['EXP R','EXPR']"
        "ipowf_ ['EXP R','EXPR']"
        "isprime_ ['EXP R']"
        "plus_ ['EXP R','EXPR']"
        "times_ ['EXP R','EXPR']"
        ), 
    labels=c(
        "Ends-with (Expr, Expr)",
        "Expr ^ Expr"
        "Prime (Expr)"
        "Expr + Expr"
        "Expr * Expr"))





















