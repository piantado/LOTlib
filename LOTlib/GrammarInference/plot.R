
library(tidyr)
library(ggplot2)

dwide <- read.table("o.txt", header=T)

d <- gather(dwide, parameter, value, 3:ncol(dwide)) # put a single parameters on each row, like ggplot wants
d$chain <- as.factor(d$chain)

plt <- ggplot(d, aes(x=value, fill=chain)) + 
        geom_histogram(alpha=0.3, position="identity", bins=50) + 
        facet_wrap(~parameter, scales="free") 

plt
