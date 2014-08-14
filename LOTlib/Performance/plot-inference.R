# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)
library(stringr)
library(gridExtra) # needed for "unit"

d <- read.table("output/inference-aggregate.txt")
names(d)[1:6] <- c("model", "iteration", "method.param", "steps", "time", "Z")

d$method <- gsub("_[A-Z]$", "", d$method.param, perl=TRUE)
d$parameter <- as.factor(str_sub(d$method.param, -1, -1))

p <- ggplot(d, aes(x=steps, y=Z, color=method, linetype=parameter)) + 
	stat_summary(fun.y=mean, geom="line", size=2) +
	opts(legend.key.size=unit(3,"lines")) +
	facet_wrap( ~ model, scales="free")

ggsave("output/inference.pdf", width=16, height=12)



# Plot the MAP found
# p <- ggplot(d, aes(x=steps, y=V8, color=method, linetype=parameter)) + 
# 	stat_summary(fun.y=mean, geom="line", size=2) +
# 	opts(legend.key.size=unit(3,"lines")) +
# 	facet_wrap( ~ model, scales="free")


# p

# A contour plot -- TODO: Fix the scaling
# a <- aggregate( Z ~ nchains + temperature, d, median)
# p <- ggplot( a, aes(x=log(nchains), y=log(temperature), z=Z)) +
# 	stat_contour( aes(color=..level..), binwidth=.5, size=2) 
# p

# A simpler tiled plot
# p <- ggplot(a, aes(nchains, temperature)) + 
# 	geom_tile(aes(fill = Z), colour = "white") + 
# 	scale_fill_gradient(low = "white",high = "steelblue")
# p



