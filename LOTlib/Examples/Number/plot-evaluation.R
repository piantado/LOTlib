# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)


d <- read.table("evaluation-temperatures-aggregate.txt")

p <- ggplot(d, aes(x=V4, y=V6, color=as.character(V3))) + stat_summary(fun.y=mean, geom="line", size=2)

p