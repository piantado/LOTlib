# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)
library(stringr)
library(gridExtra) # needed for "unit"

d <- NULL
for(f in list.files("output", pattern="agg*", full.names=TRUE)) {
        d <- rbind(d, read.table(f))
}
names(d)[1:6] <- c("model", "iteration", "nchains", "steps", "time", "Z")


d$nchains <- as.factor(d$nchains)
p <- ggplot(d, aes(x=steps, y=Z, color=nchains)) + 
	stat_summary(fun.y=mean, geom="line", size=1) +
	opts(legend.key.size=unit(3,"lines")) +
	facet_wrap( ~ model, scales="free") 

ggsave("chains-vs-length.pdf", width=16, height=12)
# p




