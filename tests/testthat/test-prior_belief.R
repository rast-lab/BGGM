library(testthat)
library(BGGM)

# Test data setup
set.seed(123)
n <- 100
p <- 4
test_data <- matrix(rnorm(n * p), ncol = p)
colnames(test_data) <- paste0("V", 1:p)

# ============================================
# Tests for prior_belief_ggm
# ============================================
# Note: prior_ggm is a matrix of prior odds for including edges

test_that("prior_belief_ggm works with basic parameters", {
  # Create a prior odds matrix
  # Values > 1 favor including the edge, 1 = equal odds
  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- prior_ggm[2, 1] <- 10  # Strong prior for this edge
  prior_ggm[2, 3] <- prior_ggm[3, 2] <- 5   # Moderate prior for this edge

  result <- prior_belief_ggm(
    Y = test_data,
    prior_ggm = prior_ggm,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "prior_ggm"))
  expect_true(is.list(result))
})

test_that("prior_belief_ggm returns adjacency matrix", {
  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- prior_ggm[2, 1] <- 10

  result <- prior_belief_ggm(
    Y = test_data,
    prior_ggm = prior_ggm,
    progress = FALSE
  )

  # Should contain adjacency matrix
  expect_true("adj" %in% names(result))
  expect_true(is.matrix(result$adj))
})

test_that("prior_belief_ggm returns posterior probabilities", {
  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- prior_ggm[2, 1] <- 10

  result <- prior_belief_ggm(
    Y = test_data,
    prior_ggm = prior_ggm,
    progress = FALSE
  )

  # Should contain posterior probability
  expect_true("post_prob" %in% names(result))
})

test_that("prior_belief_ggm accepts post_odds_cut parameter", {
  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- prior_ggm[2, 1] <- 10

  result <- prior_belief_ggm(
    Y = test_data,
    prior_ggm = prior_ggm,
    post_odds_cut = 10,
    progress = FALSE
  )

  expect_s3_class(result, "prior_ggm")
})

test_that("prior_belief_ggm adjacency matrix is symmetric", {
  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- prior_ggm[2, 1] <- 10

  result <- prior_belief_ggm(
    Y = test_data,
    prior_ggm = prior_ggm,
    progress = FALSE
  )

  expect_equal(result$adj, t(result$adj))
})

test_that("prior_belief_ggm adjacency has correct dimensions", {
  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- prior_ggm[2, 1] <- 10

  result <- prior_belief_ggm(
    Y = test_data,
    prior_ggm = prior_ggm,
    progress = FALSE
  )

  expect_equal(dim(result$adj), c(p, p))
})

test_that("prior_belief_ggm adjacency contains only 0 and 1", {
  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- prior_ggm[2, 1] <- 10

  result <- prior_belief_ggm(
    Y = test_data,
    prior_ggm = prior_ggm,
    progress = FALSE
  )

  expect_true(all(result$adj %in% c(0, 1)))
})

test_that("prior_belief_ggm works with equal prior odds", {
  # All prior odds equal (uninformative)
  prior_ggm <- matrix(1, p, p)

  result <- prior_belief_ggm(
    Y = test_data,
    prior_ggm = prior_ggm,
    progress = FALSE
  )

  expect_s3_class(result, "prior_ggm")
})

test_that("prior_belief_ggm works with strong prior odds", {
  # Very strong prior for all edges
  prior_ggm <- matrix(100, p, p)
  diag(prior_ggm) <- 1

  result <- prior_belief_ggm(
    Y = test_data,
    prior_ggm = prior_ggm,
    progress = FALSE
  )

  expect_s3_class(result, "prior_ggm")
})

test_that("prior_belief_ggm works with larger sample size", {
  set.seed(456)
  large_data <- matrix(rnorm(200 * p), ncol = p)
  colnames(large_data) <- paste0("V", 1:p)

  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- prior_ggm[2, 1] <- 10

  result <- prior_belief_ggm(
    Y = large_data,
    prior_ggm = prior_ggm,
    progress = FALSE
  )

  expect_s3_class(result, "prior_ggm")
})
