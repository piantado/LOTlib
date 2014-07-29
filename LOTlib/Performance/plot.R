# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)
library(stringr)
library(gridExtra) # needed for "unit"

d <- read.table("output-tmp/inference-aggregate.txt")
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

d <- read.table("output-tmp/tempchain-aggregate.txt")
names(d)[1:7] <- c("model", "iteration", "nchains", "temperature", "steps", "time", "Z")

# Pick the median or some time throughout:
a <- aggregate( Z ~ nchains + temperature, d, median)

# A contour plot -- TODO: Fix the scaling
p <- ggplot( a, aes(x=log(nchains), y=log(temperature), z=Z)) +
	stat_contour( aes(color=..level..), binwidth=.5, size=2) 
p

# A simpler tiled plot
a$nchains <- as.factor(a$nchains) # make these factors so they plot on a discrete scale
a$temperature <- as.factor(a$temperature)
p <- ggplot(a, aes(nchains, temperature)) + 
	geom_tile(aes(fill = Z), colour = "white") + 
	scale_fill_gradient(low = "white",high = "steelblue")
p



