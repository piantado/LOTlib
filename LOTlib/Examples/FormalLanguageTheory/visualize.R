library(ggplot2)
library(plyr)

d <- read.table("G://programs and files//Python//Lib//site-packages//LOTlib//LOTlib//Examples//FormalLanguageTheory//out//regular_finite_vs_infinite_2//out__0722_224739")
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

d$posterior.score <- d$posterior.score / d$data.size
# first toss hypotheses with very low probability so there are not too many

keepers <- unique(subset(d, posterior.probability > 0.5)$h) # keep any hypothesis that gets above 1% posterior prob
d2 <- subset(d, is.element(h, keepers))

d2$h <- as.factor(as.character(d2$h)) # we have to re-set this factor so it doesn't remember all the hyps we removed

# color by hypothesis
plt <- ggplot(d2, aes(x=data.size, y=posterior.probability, color=h)) +  geom_line()


d <- read.table("G://programs and files//Python//Lib//site-packages//LOTlib//LOTlib//Examples//FormalLanguageTheory//out//SimpleEnglish//out_8_10w_t")
names(d) <- c("data.size", "posterior.probability", "posterior.score", "prior", "likelihood", "number.of.generated.strings", "h", "precision", "recall")

# d$precision <- (d$precision > 0.7) & (d$recall > 0.1)
d$precision <- (d$precision * d$recall) > 0.2972972 * (d$precision + d$recall)

s <- ddply(d, "data.size", function(dd){

    ddply(dd, "precision", function(ddd){
          data.frame(score=sum(ddd$posterior.probability))
    })
})

plt <- ggplot(s, aes(x=data.size, y=score, color=precision, group=precision)) + geom_line()

plt


d <- read.table("G://programs and files//Python//Lib//site-packages//LOTlib//LOTlib//Examples//FormalLanguageTheory//out//SimpleEnglish//rr_w_t")
names(d) <- c("data.size", "posterior.probability", "posterior.score", "prior", "likelihood", "number.of.generated.strings", "h", "precision", "recall")

plt <- ggplot(d, aes(x=steps, y=precision_recall, group=n, color=n)) +  geom_line()
