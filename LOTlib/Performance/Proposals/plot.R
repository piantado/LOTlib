# Plots the evaluation for *evaluate-temperatures.py* only (otherwise the column numbers need to be changed, and the path)

library(ggplot2)
library(stringr)
library(gridExtra) # needed for "unit"

d <- NULL
for(f in list.files("output", pattern="agg*", full.names=TRUE)) {
        d <- rbind(d, read.table(f))
}
names(d) <- c("model", "iteration", "parameter", "steps", "time", "Z", "map", "aprx", "N")

# recode to two factors: strength 2/10/1 and code 
d$strength <- ifelse(grepl("2", d$parameter), "2", 
                               ifelse(grepl("10", d$parameter), "10", "1"))

d$code <- gsub("(10.0|2.0)", "L", d$parameter)
d$code <- gsub("1.0", "S", d$code)
d$code <- gsub("[^SL]", "", d$code)



d$parameter <- as.factor(d$parameter)
d$code <- as.factor(d$code)
p <- ggplot(d, aes(x=steps, y=Z, color=code, linetype=strength)) + 
	stat_summary(fun.y=mean, geom="line", size=1) +
	opts(legend.key.size=unit(3,"lines")) +
	facet_wrap( ~ model, scales="free_y") # free_y makes our y axes free
p

ggsave("proposal.pdf", width=16, height=12)





