# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)


# d <- read.table("evaluation-temperatures-aggregate.txt")
d <- read.table("ei.txt")

p <- ggplot(d, aes(x=V3, y=V5, color=as.character(V2))) + stat_summary(fun.y=mean, geom="line", size=2)

p