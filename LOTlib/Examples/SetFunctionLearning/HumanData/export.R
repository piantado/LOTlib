library(Hmisc) 

## NOTES
## This now exports two files, a held out set for test, and a training set. 
 
 
#library(MASS) # for write.matrix

# this outputs a data file for doing mcmc on the pcfg
# TFcnt stores the T and false counts for each data point, under our order of data points. 

# the standard for this data is that we remove 

source("../../../../../Libraries/misc-lib.R")
source("../../../../../Libraries/sort.R")
source("HSFL-Shared.R")

# set our seed to make this replicable
set.seed(3161984)

HELD.OUT <- 0.5 # what proportion is held out? We'll take this at random from 

d <- load.subject.data(print.summary=TRUE, remove.duplicate.concepts=TRUE)

d <- d[! is.na(d$response),]

# get the subject responses to each data point
ag <- aggregate.binomial(d$response, by=list(d$concept, d$list, d$set.number, d$response.number, d$right.answer), names=c("concept", "list", "set.number", "response.number", "right.answer"), stats=F)
 
 
# check that item numbers are right -- that we don't incorrectly map responses across subjects or something
q <- d[d$concept=="hg03" & d$list=="L1",]
print(table(q$right.answer, q$set.number, q$response.number))

 # for exporting small
#ag <- ag[ is.element(ag$concept, c("lg06", "lg07")),]

# so we sort by concept, list.number, item.number, response.number, and that gives us a list of 
# data points, which is the data vector that is handled by scheme. 
ag <- sort(ag, by = ~concept + list + set.number + response.number)

## Find the held out data
if(HELD.OUT > 0 ) {
# 
# ###############!!!!!!!!!!!!!!!!!! NOTE HERE WE HAVE CHANGED THIS TO HOLD OUT LIST 2
# ###############!!!!!!!!!!!!!!!!!! NOTE HERE WE HAVE CHANGED THIS TO HOLD OUT LIST 2
# ###############!!!!!!!!!!!!!!!!!! NOTE HERE WE HAVE CHANGED THIS TO HOLD OUT LIST 2
# ###############!!!!!!!!!!!!!!!!!! NOTE HERE WE HAVE CHANGED THIS TO HOLD OUT LIST 2
# ###############!!!!!!!!!!!!!!!!!! NOTE HERE WE HAVE CHANGED THIS TO HOLD OUT LIST 2
#  ho.ind <- sample.int( nrow(ag), size=HELD.OUT*nrow(ag), replace=FALSE) 
  ho.ind <- (1:nrow(ag))[ag$list == "L2"]
  ho.ag <- ag[ho.ind,]
  ag[ho.ind,]$x <- 0
  ag[ho.ind,]$n <- 0
  
  # save the held out data
  write.table( ho.ag, file="HELD-OUT-ag.txt", sep="\t", row.names=F)
  
  # save a key for whether it was held out. 
  write.table( data.frame( concept=as.character(ag$concept), 
			list=ag$list, 
			set.number=ag$set.number, 
			response.number=ag$response.number, 
			is.held.out=(ag$x==0 & ag$n==0) ## NOTE: This assumes we have nonzero responses everywhere!
			), file="HELD-OUT-key.txt", sep="\t", row.names=F)
}

# we stack all data points by concept, item.number, response.number, to produce one big vector
write.table(data.frame(concept=as.character(ag$concept), 
			list=ag$list, 
			set.number=ag$set.number, 
			response.number=ag$response.number, 
			t=ag$x, 
			f=ag$n-ag$x, 
			right.answer=ifelse(ag$right.answer,1,0) ),
		file="../Model-Vicare/Data-Scheme-HELD-OUT.txt", sep="\t", row.names=F, col.names=F, quote=F)



