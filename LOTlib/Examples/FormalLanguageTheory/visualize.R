library(ggplot2)
library(plyr)

d <- read.table("G://programs and files//Python//Lib//site-packages//LOTlib//LOTlib//Examples//FormalLanguageTheory//out//regular_finite_vs_infinite//out_False_0718_014626")
names(d) <- c("data.size", "posterior.probability", "posterior.score", "prior", "likelihood", "number.of.generated.strings", "h", "precision", "recall")


# Now compute a summary of the data -- for each data_size, compute an expected length
s <- ddply( d, "data.size", function(dd){ data.frame(expected.number.of.strings=sum(dd$number.of.generated.strings*dd$posterior.probability))}  )

# make the plot
plt <- ggplot(s, aes(x=data.size, y=expected.number.of.strings)) + geom_line()

# show it
plt



###################################################################
## Now let's make a plot of hypotheses vs amount of data
###################################################################

# first toss hypotheses with very low probability so there are not too many

keepers <- unique(subset(d, posterior.probability > 0.05)$h) # keep any hypothesis that gets above 1% posterior prob
d2 <- subset(d, is.element(h, keepers))

d2$h <- as.factor(as.character(d2$h)) # we have to re-set this factor so it doesn't remember all the hyps we removed

# color by hypothesis
plt <- ggplot(d2, aes(x=data.size, y=posterior.probability, color=h)) +  geom_line()


# plot precision and recall
plt <- ggplot(d2, aes(x=data.size, y=precision, color=h)) +  geom_line()

# plot recall and recall
plt <- ggplot(d2, aes(x=data.size, y=recall, color=h)) +  geom_line()


