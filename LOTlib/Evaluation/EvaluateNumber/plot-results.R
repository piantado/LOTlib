
library(ggplot2)

d <- read.table("evaluation-chains.txt")
names(d) <- c("name", "chain.i", "n", "time", "KL", "pct.found", "log.pm.found", "len.hyp", "len.samples", "sampleZ")
d$type <- substr(as.character(d$name), 0, 6) # the first few chars, ignoring the parameters at the end of name
d$time <- round(d$time/100) * 100
# d <- d[d$time<800,]


a <- aggregate( KL ~ name + time + type, d, mean)

p <- ggplot( data=a, aes(x=time, y=KL, color=name, group=name)) +
	geom_line(size=1.1) +
	scale_x_continuous(limits = c(0,800))
	
p
