% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fitted.R
\name{fitted.estimate}
\alias{fitted.estimate}
\title{Fitted Values for \code{estimate} Objects}
\usage{
\method{fitted}{estimate}(object, iter = 500, cred = 0.95, summary = TRUE, ...)
}
\arguments{
\item{object}{object of class \code{estimate}}

\item{iter}{iters used to compute the residuals}

\item{cred}{credible interval used for summarizing}

\item{summary}{summarize the posterior samples (Default is \code{TRUE}).}

\item{...}{currently ignored}
}
\value{
3D array of dimensions n (observations),
        4 (posterior summary),
        p (number of nodes)
}
\description{
Fitted Values for \code{estimate} Objects
}
\examples{
\donttest{
# data
Y <- subset(tas, gender == "M")[,-ncol(tas)]

# fit model
fit <- estimate(Y)

# fitted values
fitted(fit, iter = 25)
}
}
