library(ggplot2)
library(stringr)
library(gridExtra) # needed for "unit"

##############################################################
## Evaluation of proposals
##############################################################

d <- read.table("output/proposal-aggregate.txt")
names(d)[1:7] <- c("model", "iteration", "proposal.type", "parameter", "steps", "time", "Z")

d$parameter <- as.factor(d$parameter)
p <- ggplot(d, aes(x=steps, y=Z, color=parameter)) + 
	stat_summary(fun.y=mean, geom="line", size=1) +
	opts(legend.key.size=unit(3,"lines")) +
	facet_wrap( ~ model, scales="free_y") # free_y makes our y axes free
p

ggsave("output/proposal.pdf", width=16, height=12)





