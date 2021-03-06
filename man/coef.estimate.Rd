% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/coef.estimate.R
\name{coef.estimate}
\alias{coef.estimate}
\title{Precision Matrix to Multiple Regression}
\usage{
\method{coef}{estimate}(object, node = 1, cred = 0.95, iter = 500, ...)
}
\arguments{
\item{object}{object of class \code{estimate} (analytic = F)}

\item{node}{which node to summarize (i.e., the outcome)}

\item{cred}{credible interval used in the summary output}

\item{iter}{number of samples used in the conversion.}

\item{...}{e.g., \code{digits}}
}
\value{
list of class \code{coef.estimate}:

list \code{inv_2_beta}:
\itemize{
 \item \code{betas} posterior samples for the regression coefficients
 \item \code{sigma} posterior samples for sigma (residual sd)
 \item \code{call} \code{match.call()}
}

data frame \code{summary_inv_2_beta}:
\itemize{
\item summary of regression coefficients
}


\code{call} \code{match.call()}
}
\description{
There is a direct correspondence between the covariance matrix and multiple regression. In the case of GGMs, it is possible
to estimate the edge set with multiple regression (i.e., neighborhood selection). In *BGGM*, the precision matrix is first sampled from, and then
each draws is converted to the corresponding coefficients and error variances. This results in a posterior distribution. This function can be used
to perform Bayesian multiple regression.
}
\examples{
# p = 10
Y <- BGGM::bfi[,1:10]

# sample posterior
fit <- estimate(Y, iter = 5000)

# precision to regression
coefficients(fit, node = 1, cred = 0.95)
}
