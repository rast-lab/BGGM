% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/posterior_predict.R
\name{posterior_predict}
\alias{posterior_predict}
\title{Posterior Predictive Distribution for \code{estimate} Objects}
\usage{
posterior_predict(
  object,
  iter = 500,
  cred = 0.95,
  newdata = NULL,
  summary = TRUE,
  ...
)
}
\arguments{
\item{object}{object of class \code{estimate}}

\item{iter}{iters used to compute the residuals}

\item{cred}{credible interval used for summarizing}

\item{newdata}{an optional data frame for obtaining predictions (e.g., on test data)}

\item{summary}{summarize the posterior samples (Default is \code{TRUE}).
Setting it to \code{FALSE} can be used to then compute performance metrics.}

\item{...}{currently ignored}
}
\value{
\code{summary = TRUE}: 3D array of dimensions n (observations),
        4 (posterior summary),
        p (number of nodes). \code{summary = FALSE}:
        list containing predictions for each variable
}
\description{
Posterior Predictive Distribution for \code{estimate} Objects
}
\examples{
\donttest{
# data
Y <- subset(tas, gender == "M")[,-ncol(tas)]

# fit model
fit <- estimate(Y)

# predict
posterior_predict(fit, iter = 25)
}
}
