# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)


# d <- read.table("evaluation-temperatures-aggregate.txt")
d <- read.table("ei.txt")
names(d)[1:5] <- c("iteration", "method.param", "steps", "time", "Z")

d$method <- gsub("_[A-Z]$", "", d$method.param, perl=TRUE)
# d$parameter <- 

p <- ggplot(d, aes(x=steps, y=Z, color=method, group=method.param)) + 
	stat_summary(fun.y=mean, geom="line", size=2) 
	
p


