d <- read.table("o.txt", header=F)
names(d) <- c("prior", "ll", "knower.level", "hypothesis")

data_sizes <- seq(0.0, 500.0, 1.0)

# and compute the normalizers for each data size
Z = rep(0, length(data_sizes))
for(i in 1:length(data_sizes)) {
	d$posterior <- d$prior + d$ll*data_sizes[i]
	m <- max(d$posterior)
	Z[i] <- m + log(sum(exp(d$posterior-m)))  #lse
}

# Now go throug knower levels / behavioral patterns and plot
plot(NULL, xlim=range(data_sizes), ylim=c(0,1), xlab="Amount of data", ylab="Posterior probability")
cols <- c("#029a67", "#3966cd", "#fce000", "#ff6602", "#ca0032", "#999999")
for(b in levels(d$knower.level)) {

	a <- d[d$knower.level == b,]
	
	posteriors <- rep(NA, length(data_sizes)) 
	for(i in 1:length(data_sizes)) {
		posteriors[i] <- sum( exp(a$prior + a$ll*(data_sizes[i]) - Z[i]))
	}
	
	cl <- "#efefef"
	if(b == "1UUUUUUUU") { cl=cols[1] }
	if(b == "12UUUUUUU") { cl=cols[2] }
	if(b == "123UUUUUU") { cl=cols[3] }
	if(b == "1234UUUUU") { cl=cols[4] }
	if(b == "123456789") { cl=cols[5] }
# 	if
	
	lines(data_sizes, posteriors, lwd=2, col=cl)
}
