% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ggm_compare_ppc.default.R
\name{plot.ggm_compare_ppc}
\alias{plot.ggm_compare_ppc}
\title{Plot \code{ggm_compare_ppc} Objects}
\usage{
\method{plot}{ggm_compare_ppc}(
  x,
  critical = 0.05,
  col_noncritical = "#84e184A0",
  col_critical = "red",
  point_size = 2,
  log = FALSE,
  ...
)
}
\arguments{
\item{x}{object of class \code{ggm_compare_ppc}}

\item{critical}{'significance' level}

\item{col_noncritical}{fill color of the non critical region}

\item{col_critical}{fill color of the critical region (e.g., \code{critical = 0.05})}

\item{point_size}{point size for the observed KL-divergence}

\item{log}{log transformation. useful for small values and skewed predictive distributions}

\item{...}{currently ignored}
}
\value{
one object of class \code{ggplot} when \code{type = "global"}. One object for each pairwise contrast when \code{type = "nodewise"}
}
\description{
Plot \href{https://CRAN.R-project.org/package=ggridges/vignettes/introduction.html}{ggridges} for
the GGM comparison with posterior predictive KL-divergence. The plots contain the predictive distribution, assuming group equality, as well as
the observed KL-divergence. Further, the predictive distributions are conveniently colored to infer whether the null of group equality
should be rejected. This is accomplished by having the critical region, corresponding to a desired 'significance' level, shaded in red.
Thus, if the observed value is in the red region, this suggests the null hypothesis of group equality should be rejected.
}
\note{
This method is Bayesian, as it relies on the posterior predictive distribution. That said, there are clear parallels to frequentist testing-e.g.,
assuming group equality and critical regions. Most importantly, this method CANNOT provide evidence for the null hypothesis. Thus it can only reject
the underlying assumption of group equality..
}
\examples{
# assume group equality
Y1 <- MASS::mvrnorm(500, rep(0, 16), Sigma = diag(16))
Y2 <- MASS::mvrnorm(500, rep(0, 16), Sigma = diag(16))
Y3 <- MASS::mvrnorm(500, rep(0, 16), Sigma = diag(16))

# global
ggm_ppc  <- ggm_compare_ppc(Y1, Y2, Y3, type = "global", iter = 50)

# plot
plot(ggm_ppc)

\donttest{
# nodewise
ggm_ppc  <- ggm_compare_ppc(Y1, Y2, Y3, type = "nodewise", iter = 50)

plot(ggm_ppc, log = TRUE)
}
}
\references{
Williams, D. R., Rast, P., Pericchi, L. R., & Mulder, J. (2019). Comparing Gaussian Graphical
Models with the Posterior Predictive Distribution and Bayesian Model Selection. \href{https://psyarxiv.com/yt386}{pre print}
}
