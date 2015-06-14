# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)
library(stringr)
library(gridExtra) # needed for "unit"

d <- NULL
for(f in list.files("output", pattern="agg*", full.names=TRUE)) {
        d <- rbind(d, read.table(f))
}

names(d) <- c("model", "iteration", "temperature", "steps", "time", "Z", "map", "aprx", "N")

d$temperature <- as.factor(d$temperature)
p <- ggplot(d, aes(x=steps, y=Z, color=temperature)) + 
	stat_summary(fun.y=mean, geom="line", size=1) +
	opts(legend.key.size=unit(3,"lines")) +
	facet_wrap(~ model, scales="free") 

ggsave("temperatures.pdf", width=16, height=12)
# p




