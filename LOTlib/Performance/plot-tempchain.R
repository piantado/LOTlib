library(ggplot2)
library(stringr)
library(gridExtra) # needed for "unit"

##############################################################
## Evaluation of temperatures
##############################################################

d <- read.table("output/tempchain-aggregate.txt")
names(d)[1:7] <- c("model", "iteration", "nchains", "temperature", "steps", "time", "Z")


d$temperature <- as.factor(d$temperature)
d$nchains <- as.factor(d$nchains)
p <- ggplot(d, aes(x=steps, y=Z, color=temperature)) + 
	stat_summary(fun.y=mean, geom="line", size=1) +
	opts(legend.key.size=unit(3,"lines")) +
	facet_wrap(model ~ nchains, scales="free") 

ggsave("output/tempchain.pdf", width=16, height=12)
# p




