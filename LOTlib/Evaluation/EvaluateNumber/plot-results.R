
library(ggplot2)

d <- read.table("evaluation.txt")
a <- aggregate( V5 ~ V1 + V4, d, mean)

p <- ggplot( data=a, aes(x=V4, y=V5, color=V1, group=V1)) +
	geom_line(size=1.5)
	
p
