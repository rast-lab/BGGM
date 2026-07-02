library(testthat)
library(BGGM)

# ============================================
# Tests for the C++ search() / bic_fast() / hft_algorithm() functions
# that back ggm_search(). These target the specific behaviors involved in
# fixing the historical "creeping BIC" issue: a variable-shadowing bug that
# froze the edge-flip proposal set after the first accepted move, and a
# missing birth-death Hastings correction for the add/remove proposal-size
# asymmetry. See NEWS.md / commit history for the full writeup.
# ============================================

set.seed(2026)

make_test_ggm <- function(p = 6, n = 300) {
  Theta <- diag(p)
  edges <- which(upper.tri(Theta))
  n_edges <- max(1, round(length(edges) * 0.3))
  chosen <- sample(edges, n_edges)
  vals <- runif(n_edges, 0.15, 0.3) * sample(c(-1, 1), n_edges, replace = TRUE)
  Theta[chosen] <- vals
  Theta <- Theta + t(Theta) - diag(diag(Theta))
  diag(Theta) <- diag(Theta) + max(0, -min(eigen(Theta, only.values = TRUE)$values)) + 0.5
  Sigma <- solve(Theta)
  Y <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  list(S = cor(Y), n = n, p = p)
}

run_search <- function(dat, iter, probabilistic, adj_start = NULL, stop_early = NULL) {
  p <- dat$p
  if (is.null(adj_start)) adj_start <- diag(p)
  if (is.null(stop_early)) stop_early <- iter
  old <- BGGM:::hft_algorithm(Sigma = dat$S, adj = adj_start, tol = 1e-10, max_iter = 100)
  bic_old <- BGGM:::bic_fast(Theta = old$Theta, S = dat$S, n = dat$n, prior_prob = 0.3)
  .Call('_BGGM_search', PACKAGE = 'BGGM',
        S = dat$S, iter = iter, old_bic = bic_old, start_adj = adj_start,
        n = dat$n, gamma = 0.3, stop_early = stop_early, progress = FALSE,
        probabilistic = probabilistic)
}

test_that("bic_fast responds to prior_prob only when the graph has off-diagonal edges", {
  dat <- make_test_ggm()
  adj_dense <- matrix(1, dat$p, dat$p)
  adj_sparse <- diag(dat$p)

  fit_dense <- BGGM:::hft_algorithm(Sigma = dat$S, adj = adj_dense, tol = 1e-10, max_iter = 100)
  fit_sparse <- BGGM:::hft_algorithm(Sigma = dat$S, adj = adj_sparse, tol = 1e-10, max_iter = 100)

  bic_dense_lowprior <- BGGM:::bic_fast(fit_dense$Theta, dat$S, dat$n, prior_prob = 0.1)
  bic_dense_hiprior  <- BGGM:::bic_fast(fit_dense$Theta, dat$S, dat$n, prior_prob = 0.9)

  # bic_fast's edge penalty term depends on prior_prob, so it must change
  # the BIC value for a graph that has off-diagonal edges
  expect_false(isTRUE(all.equal(bic_dense_lowprior, bic_dense_hiprior)))

  # the empty graph has no off-diagonal nonzero entries, so the penalty
  # term (which multiplies by nonzero-edge count) is a no-op either way
  bic_sparse_lowprior <- BGGM:::bic_fast(fit_sparse$Theta, dat$S, dat$n, prior_prob = 0.1)
  bic_sparse_hiprior  <- BGGM:::bic_fast(fit_sparse$Theta, dat$S, dat$n, prior_prob = 0.9)
  expect_equal(bic_sparse_lowprior, bic_sparse_hiprior)
})

test_that("greedy search (probabilistic = FALSE) has a monotone non-increasing BIC trajectory", {
  dat <- make_test_ggm()
  fit <- run_search(dat, iter = 300, probabilistic = FALSE)

  bics <- fit$bics
  bics <- bics[bics != 0 | seq_along(bics) == 1]
  # greedy only ever accepts a move that improves (lowers) BIC
  expect_true(all(diff(bics) <= 1e-8))
})

test_that("probabilistic search accepts a healthy fraction of proposals (regression: proposal-set shadowing bug)", {
  dat <- make_test_ggm()
  fit <- run_search(dat, iter = 1000, probabilistic = TRUE)

  # with the historical shadowing bug, the add/remove proposal set froze
  # after the first accepted move (it never picked up the graph's current
  # state), which strongly caps how many further moves could ever be
  # accepted; a correctly functioning sampler clears this easily for a
  # p = 6 graph (15 possible edges)
  expect_gt(fit$acc, 100)
})

test_that("probabilistic search does not show systematic BIC drift over a long run (regression: creeping BIC)", {
  dat <- make_test_ggm()
  fit <- run_search(dat, iter = 3000, probabilistic = TRUE)

  bics <- fit$bics
  bics <- bics[bics != 0 | seq_along(bics) == 1]

  # compare mean BIC across the first vs. last third of the run; a
  # systematically drifting/creeping chain shows a clear trend across
  # thirds, a stationary chain fluctuates around a stable level. Tolerance
  # is generous on purpose (coarse, non-flaky check, not a precise
  # convergence diagnostic) -- the drift observed with the uncorrected
  # Hastings ratio was an order of magnitude larger than this threshold.
  n_b <- length(bics)
  third <- n_b %/% 3
  first_third <- mean(bics[1:third])
  last_third  <- mean(bics[(n_b - third + 1):n_b])

  expect_lt(abs(last_third - first_third), 0.15 * abs(first_third))
})

test_that("probabilistic search always runs the full iter count (stop_early is ignored)", {
  dat <- make_test_ggm()
  fit <- run_search(dat, iter = 500, probabilistic = TRUE, stop_early = 1)

  # every slot should have been written to -- no trailing zero padding
  # from an early stop, i.e. the chain ran all 500 iterations despite
  # stop_early being set to an aggressively low value
  expect_equal(sum(fit$bics != 0), 500)
})

test_that("greedy search honors stop_early", {
  dat <- make_test_ggm()
  fit <- run_search(dat, iter = 5000, probabilistic = FALSE, stop_early = 5)

  n_ran <- sum(fit$bics != 0)
  expect_lt(n_ran, 5000)
})
