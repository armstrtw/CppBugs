require(inline)
require(Rcpp)

get.model <- function(model.file) {
    do.call(paste,as.list(paste(readLines(model.file),"\n")))
}

model <- get.model("linear.model.hpp")
src <- get.model("run.model.cpp")

linear.model <- cxxfunction(signature(XR="numeric", yr="numeric",iterations="integer",burn="integer",adapt="adapt",thin="integer"), body=src, include=model, plugin="RcppArmadillo",verbose=F)


NR <- 1000
NC <- 5
X <- cbind(rep(1,NR),matrix(rnorm(NR*NC),nrow=NR,ncol=NC))
y <- matrix(rnorm(NR),nrow=NR)

res <- linear.model(XR=X,yr=y,iterations=1e5L,burn=1e5L,adapt=2e3L,thin=5L)
print(res)
