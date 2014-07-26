# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)
library(stringr)
library(gridExtra) # needed for "unit"

d <- read.table("output/inference-aggregate.txt")
# d <- read.table("ei.txt")
names(d)[1:6] <- c("model", "iteration", "method.param", "steps", "time", "Z")

d$method <- gsub("_[A-Z]$", "", d$method.param, perl=TRUE)
d$parameter <- as.factor(str_sub(d$method.param, -1, -1))

p <- ggplot(d, aes(x=steps, y=Z, color=method, linetype=parameter)) + 
	stat_summary(fun.y=mean, geom="line", size=2) +
	opts(legend.key.size=unit(3,"lines")) +
	facet_wrap( ~ model, scales="free")
	
p


##############################################################
## Evaluation of temperatures
##############################################################

d <- read.table("evaluation-inference-aggregate.txt")
# d <- read.table("et.txt")
names(d)[1:6] <- c("iteration", "nchains", "temperature", "steps", "time", "Z")

aggregate( Z ~ nchains + temperature, d, max)


