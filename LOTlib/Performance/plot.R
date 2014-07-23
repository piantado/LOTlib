# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)
library(stringr)

d <- read.table("data/evaluation-inference-aggregate.txt")
# d <- read.table("ei.txt")
names(d)[1:6] <- c("model", "iteration", "method.param", "steps", "time", "Z")

d$method <- gsub("_[A-Z]$", "", d$method.param, perl=TRUE)
d$parameter <- as.factor(str_sub(d$method.param, -1, -1))

p <- ggplot(d, aes(x=steps, y=Z, color=method, linetype=parameter)) + 
	stat_summary(fun.y=mean, geom="line", size=2) 
	
p


##############################################################
## Evaluation of temperatures
##############################################################

# # d <- read.table("evaluation-inference-aggregate.txt")
# d <- read.table("et.txt")
# names(d)[1:6] <- c("iteration", "nchains", "temperature", "steps", "time", "Z")
# 
# aggregate( Z ~ nchains + temperature, d, max)
# 

