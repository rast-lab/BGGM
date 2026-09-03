library(testthat)
library(BGGM)

# helper: recompute the correct prior/posterior densities the way select() should
# (defined at the top so every test below can use it).
.se_dens <- function(fit) {
  samp_idx  <- 51:fit$iter
  post_sd   <- apply(fit$post_samp$fisher_z[,, samp_idx], 1:2, sd)
  post_mean <- apply(fit$post_samp$fisher_z[,, samp_idx], 1:2, mean)
  post_dens <- dnorm(0, post_mean, post_sd)
  prior_sd  <- apply(fit$prior_samp$fisher_z[,, samp_idx], 1:2, sd)
  # correct: average only the off-diagonal (edge) prior SDs
  prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))
  list(post_mean = post_mean, post_sd = post_sd,
       BF_10 = prior_dens / post_dens)
}

test_that("select.explore returns expected structure for two-sided alternative", {
  Y <- matrix(rnorm(100), ncol = 4)
  fit <- explore(Y, progress = FALSE)
  result <- select.explore(fit, alternative = "two.sided")
  
  expect_s3_class(result, "select.explore")
  expect_named(result, c("pcor_mat_zero", "pcor_mat", "pcor_sd_fisher", "Adj_10", "Adj_01", "BF_10", "BF_01", "BF_cut", "method", "alternative", "call", "type", "formula", "analytic", "object"))
  expect_true(is.matrix(result$pcor_mat_zero))
  expect_true(is.matrix(result$pcor_mat))
  expect_true(is.matrix(result$Adj_10))
  expect_true(is.matrix(result$Adj_01))
})

test_that("select.explore returns expected structure for greater alternative", {
  Y <- matrix(rnorm(100), ncol = 4)
  fit <- explore(Y, progress = FALSE)
  result <- select.explore(fit, alternative = "greater")
  
  expect_s3_class(result, "select.explore")
  expect_named(result, c("pcor_mat_zero", "pcor_mat", "pcor_sd_fisher", "Adj_20", "Adj_02", "BF_20", "BF_02", "BF_cut", "method", "alternative", "call", "type", "formula", "analytic", "object"))
  expect_true(is.matrix(result$pcor_mat_zero))
  expect_true(is.matrix(result$pcor_mat))
  expect_true(is.matrix(result$Adj_20))
  expect_true(is.matrix(result$Adj_02))
})

test_that("select.explore returns expected structure for exhaustive alternative", {
  Y <- matrix(rnorm(100), ncol = 4)
  fit <- explore(Y, progress = FALSE)
  result <- select.explore(fit, alternative = "exhaustive")

  expect_s3_class(result, "select.explore")
  expect_named(result, c("post_prob", "neg_mat", "pos_mat", "null_mat", "alternative", "pcor_mat", "pcor_sd_fisher", "call", "prob", "method", "type", "formula", "analytic", "object"))
  expect_true(is.data.frame(result$post_prob))
  expect_true(is.matrix(result$neg_mat))
  expect_true(is.matrix(result$pos_mat))
  expect_true(is.matrix(result$null_mat))
})

test_that("select.explore returns expected structure for less alternative", {
  Y <- matrix(rnorm(100), ncol = 4)
  fit <- explore(Y, progress = FALSE)
  result <- select.explore(fit, alternative = "less")

  expect_s3_class(result, "select.explore")
  expect_true("Adj_20" %in% names(result))
  expect_true("Adj_02" %in% names(result))
  expect_true("BF_20" %in% names(result))
})

test_that("select.explore two.sided adjacency matrices are binary", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  expect_true(all(sel$Adj_10 %in% c(0, 1)))
  expect_true(all(sel$Adj_01 %in% c(0, 1)))
})

test_that("select.explore two.sided pcor_mat is symmetric", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  expect_true(isSymmetric(sel$pcor_mat))
})

test_that("select.explore exhaustive probabilities sum to 1", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")

  prob_sums <- sel$post_prob$prob_zero + sel$post_prob$prob_greater + sel$post_prob$prob_less
  expect_true(all(abs(prob_sums - 1) < 0.01))
})

test_that("select.explore respects BF_cut parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)

  sel_3 <- select(fit, BF_cut = 3)
  sel_10 <- select(fit, BF_cut = 10)

  # Higher BF_cut should have fewer or equal selected edges
  expect_true(sum(sel_10$Adj_10) <= sum(sel_3$Adj_10))
})

test_that("select.explore print method works for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  output <- capture.output(print(sel))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
  expect_true(any(grepl("two.sided", output)))
})

test_that("select.explore print method works for greater", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "greater")

  output <- capture.output(print(sel))

  expect_true(any(grepl("greater", output)))
})

test_that("select.explore print method works for less", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "less")

  output <- capture.output(print(sel))

  expect_true(any(grepl("less", output)))
})

test_that("select.explore print method works for exhaustive", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")

  output <- capture.output(print(sel))

  expect_true(any(grepl("exhaustive", output)))
})

test_that("summary.select.explore returns correct class", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  expect_s3_class(summ, "BGGM")
  expect_s3_class(summ, "summary.select.explore")
})

test_that("summary.select.explore two.sided has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")
  summ <- summary(sel)

  expected_cols <- c("Relation", "Post.mean", "Post.sd.fisher", "Pr.H0", "Pr.H1")
  expect_equal(colnames(summ$summary), expected_cols)
})

test_that("summary.select.explore greater has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "greater")
  summ <- summary(sel)

  expect_true("Pr.H0" %in% colnames(summ$summary))
  expect_true("Pr.H1" %in% colnames(summ$summary))
})

test_that("summary.select.explore exhaustive has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")
  summ <- summary(sel)

  expect_true("Pr.H0" %in% colnames(summ$summary))
  expect_true("Pr.H1" %in% colnames(summ$summary))
  expect_true("Pr.H2" %in% colnames(summ$summary))
})

test_that("summary.select.explore print method works", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  output <- capture.output(print(summ))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("plot.summary.select.explore returns ggplot", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  plt <- plot(summ)

  expect_true(inherits(plt, "ggplot"))
})

test_that("plot.summary.select.explore respects size parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  plt <- plot(summ, size = 3)

  expect_true(inherits(plt, "ggplot"))
})

test_that("plot.summary.select.explore respects color parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  plt <- plot(summ, color = "red")

  expect_true(inherits(plt, "ggplot"))
})

test_that("select.explore with col_names = FALSE", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel, col_names = FALSE)

  # Should use numeric indices
  expect_true(any(grepl("^[0-9]+--[0-9]+$", summ$summary$Relation)))
})

test_that("select.explore without column names", {
  set.seed(123)
  Y <- as.matrix(BGGM::bfi[1:100, 1:5])
  colnames(Y) <- NULL

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")

  # Should work with numeric edge names
  expect_true(any(grepl("^[0-9]+--[0-9]+$", sel$post_prob$edge)))
})

# ---- Regression tests: pin internal consistency of BF_cut behavior ----

test_that("BF_10 and BF_01 are reciprocals for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  off_diag <- upper.tri(sel$BF_10)
  expect_true(all(abs(sel$BF_10[off_diag] * sel$BF_01[off_diag] - 1) < 1e-10))
})

test_that("Adj_10 is 1 exactly where BF_10 > BF_cut for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  BF_cut <- 3
  sel <- select(fit, BF_cut = BF_cut, alternative = "two.sided")

  off_diag <- upper.tri(sel$BF_10)
  expect_equal(sel$Adj_10[off_diag], as.numeric(sel$BF_10[off_diag] > BF_cut))
})

test_that("Adj_01 is 1 exactly where BF_10 < 1/BF_cut for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  BF_cut <- 3
  sel <- select(fit, BF_cut = BF_cut, alternative = "two.sided")

  off_diag <- upper.tri(sel$BF_10)
  expect_equal(sel$Adj_01[off_diag], as.numeric(sel$BF_10[off_diag] < 1 / BF_cut))
})

test_that("pcor_mat_zero is zero where Adj_10 is 0 for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  expect_true(all(sel$pcor_mat_zero[sel$Adj_10 == 0] == 0))
})

test_that("pcor_mat_zero is nonzero where Adj_10 is 1 for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  if (any(sel$Adj_10 == 1)) {
    expect_true(all(sel$pcor_mat_zero[sel$Adj_10 == 1] != 0))
  }
})

test_that("exhaustive prob field equals BF_cut / (BF_cut + 1)", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  BF_cut <- 3
  sel <- select(fit, BF_cut = BF_cut, alternative = "exhaustive")

  expect_equal(sel$prob, BF_cut / (BF_cut + 1))
})

test_that("Adj_20 is 1 exactly where BF_20 > BF_cut for greater", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  BF_cut <- 3
  sel <- select(fit, BF_cut = BF_cut, alternative = "greater")

  off_diag <- upper.tri(sel$BF_20)
  expect_equal(sel$Adj_20[off_diag], as.numeric(sel$BF_20[off_diag] > BF_cut))
})

# ---- Exhaustive BF_cut semantics: BF_cut means a Bayes factor against the
#      complement, NOT a posterior-probability threshold. Under equal 1/3
#      priors the prior odds of H_k vs its complement are 1:2, so
#      BF_{k,!k} = 2 * P(H_k|Y) / (1 - P(H_k|Y)).

test_that("prior-odds conversion: BF 3 <-> P 0.60, BF 6 <-> P 0.75 (equal thirds)", {
  # a posterior probability of 0.60 corresponds to BF = 3 against the complement
  expect_equal(2 * 0.60 / (1 - 0.60), 3)
  # 0.75 corresponds to BF = 6, NOT 3 (the old, incorrect BF_cut/(BF_cut+1) rule)
  expect_equal(2 * 0.75 / (1 - 0.75), 6)
})

test_that("exhaustive BF_cut selection thresholds the BF against the complement", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  BF_cut <- 3
  sel <- select(fit, BF_cut = BF_cut, alternative = "exhaustive")

  pn <- sel$post_prob$prob_zero
  pg <- sel$post_prob$prob_greater
  pl <- sel$post_prob$prob_less

  # decisions must follow BF-vs-complement > BF_cut, i.e. P > BF_cut/(BF_cut+2)
  expect_equal(sel$null_mat[upper.tri(sel$null_mat)], as.numeric(2 * pn / (1 - pn) > BF_cut))
  expect_equal(sel$pos_mat[upper.tri(sel$pos_mat)],   as.numeric(2 * pg / (1 - pg) > BF_cut))
  expect_equal(sel$neg_mat[upper.tri(sel$neg_mat)],   as.numeric(2 * pl / (1 - pl) > BF_cut))

  # and equivalently the 0.60 posterior-probability boundary (NOT 0.75)
  expect_equal(sel$null_mat[upper.tri(sel$null_mat)], as.numeric(pn > BF_cut / (BF_cut + 2)))
})

test_that("exhaustive BF_cut selection is NOT the old 0.75 posterior-prob rule", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, BF_cut = 3, alternative = "exhaustive")

  pn <- sel$post_prob$prob_zero
  old_rule <- as.numeric(pn > 3 / (3 + 1))   # the previous hyp_prob = 0.75 threshold
  new_rule <- sel$null_mat[upper.tri(sel$null_mat)]
  # the two rules must actually differ on this data (guards the change is real)
  expect_false(isTRUE(all.equal(new_rule, old_rule)))
})

test_that("exhaustive BF_cut ignores prior.prob.H0 (equal 1/3 priors)", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  # prior.prob.H0 is a BMA argument; it must not affect the BF_cut result
  a <- select(fit, BF_cut = 3, alternative = "exhaustive", prior.prob.H0 = 0.5)
  b <- select(fit, BF_cut = 3, alternative = "exhaustive", prior.prob.H0 = 0.9)
  expect_equal(a$post_prob, b$post_prob)
  expect_equal(a$null_mat, b$null_mat)
})

# ---- BMA tests: will fail until method = "BMA" is implemented ----

test_that("BMA two.sided returns correct structure", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  expect_s3_class(sel, "select.explore")
  expect_true(all(c("pcor_mat_zero", "pcor_mat", "Adj_10", "Adj_01",
                    "BF_10", "BF_01", "prior.prob.H0", "method") %in% names(sel)))
  expect_true(is.na(sel$BF_cut))
  expect_equal(sel$method, "BMA")
  expect_equal(sel$prior.prob.H0, 0.5)
})

test_that("BMA two.sided Adj_10 is binary", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  expect_true(all(sel$Adj_10 %in% c(0, 1)))
  expect_true(all(sel$Adj_01 %in% c(0, 1)))
})

test_that("BMA two.sided Adj_10 is consistent with pcor_mat_zero != 0", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  off_diag <- upper.tri(sel$Adj_10)
  expect_equal(sel$Adj_10[off_diag], as.numeric(sel$pcor_mat_zero[off_diag] != 0))
})

test_that("BMA two.sided pcor_mat_zero is symmetric", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  expect_true(isSymmetric(sel$pcor_mat_zero))
})

test_that("BMA two.sided prior.prob.H0 = 0 selects all edges", {
  set.seed(123)
  Y <- BGGM::bfi[1:500, 1:5]
  fit <- explore(Y, iter = 500, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 0)

  off_diag <- upper.tri(sel$Adj_10)
  expect_true(all(sel$Adj_10[off_diag] == 1))
})

test_that("BMA two.sided prior.prob.H0 = 1 selects no edges", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 1)

  expect_true(all(sel$Adj_10 == 0))
})

test_that("BMA two.sided stores prior.prob.H0", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 0.75)

  expect_equal(sel$prior.prob.H0, 0.75)
})

test_that("higher prior.prob.H0 selects fewer or equal edges in BMA two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)

  sel_low  <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 0.1)
  sel_high <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 0.9)

  expect_true(sum(sel_high$Adj_10) <= sum(sel_low$Adj_10))
})

test_that("BMA greater returns correct structure", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "greater")

  expect_s3_class(sel, "select.explore")
  expect_true(all(c("Adj_20", "Adj_02", "BF_20", "BF_02") %in% names(sel)))
  expect_equal(sel$method, "BMA")
})

test_that("BMA greater pcor_mat_zero has no negative values", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "greater")

  expect_true(all(sel$pcor_mat_zero >= 0))
})

test_that("BMA less returns correct structure", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "less")

  expect_s3_class(sel, "select.explore")
  expect_true(all(c("Adj_20", "Adj_02", "BF_20", "BF_02") %in% names(sel)))
  expect_equal(sel$method, "BMA")
})

test_that("BMA less pcor_mat_zero has no positive values", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "less")

  expect_true(all(sel$pcor_mat_zero <= 0))
})

test_that("BMA exhaustive returns correct structure", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "exhaustive")

  expect_s3_class(sel, "select.explore")
  expect_true(all(c("post_prob", "neg_mat", "pos_mat", "null_mat",
                    "prior.prob.H0", "method") %in% names(sel)))
  expect_true(is.data.frame(sel$post_prob))
  expect_true(is.matrix(sel$null_mat))
  expect_equal(sel$method, "BMA")
  expect_equal(sel$prior.prob.H0, 0.5)
})

test_that("BMA exhaustive probabilities sum to 1", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "exhaustive")

  s <- sel$post_prob$prob_zero + sel$post_prob$prob_greater + sel$post_prob$prob_less
  expect_true(all(abs(s - 1) < 1e-10))
})

test_that("BMA exhaustive assigns every edge to exactly one state", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "exhaustive")

  od <- upper.tri(sel$null_mat)
  total <- sel$null_mat[od] + sel$pos_mat[od] + sel$neg_mat[od]
  expect_true(all(total == 1))
  expect_true(all(diag(sel$null_mat) == 0))
})

test_that("BMA exhaustive follows Eq. 9 with prior.prob.H0 weighting", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  pH0 <- 0.5
  sel <- select(fit, method = "BMA", alternative = "exhaustive", prior.prob.H0 = pH0)

  d <- .se_dens(fit)
  BF_0u <- 1 / d$BF_10
  BF_1u <- (1 - pnorm(0, d$post_mean, d$post_sd)) * 2
  BF_2u <- pnorm(0, d$post_mean, d$post_sd) * 2
  pH1 <- pH2 <- (1 - pH0) / 2
  denom <- pH0 * BF_0u + pH1 * BF_1u + pH2 * BF_2u

  expect_equal(sel$post_prob$prob_zero,    (pH0 * BF_0u / denom)[upper.tri(BF_0u)])
  expect_equal(sel$post_prob$prob_greater, (pH1 * BF_1u / denom)[upper.tri(BF_1u)])
})

test_that("BMA exhaustive: higher prior.prob.H0 assigns at least as many edges to null", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)

  sel_lo <- select(fit, method = "BMA", alternative = "exhaustive", prior.prob.H0 = 0.05)
  sel_hi <- select(fit, method = "BMA", alternative = "exhaustive", prior.prob.H0 = 0.95)

  expect_true(sum(sel_hi$null_mat) >= sum(sel_lo$null_mat))
})

test_that("BMA exhaustive print and summary work", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "exhaustive")

  output <- capture.output(print(sel))
  expect_true(any(grepl("exhaustive", output)))

  summ <- summary(sel)
  expect_true(all(c("Pr.H0", "Pr.H1", "Pr.H2") %in% colnames(summ$summary)))
})

test_that("BMA print method works without error for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  output <- capture.output(print(sel))
  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
  expect_false(any(grepl("Bayes Factor: NA", output)))
})

# ---- Regression tests: prior density must not depend on a hardcoded
#      dimension, and must exclude the (~0) matrix diagonal. These pin the
#      one true way to compute prior_dens across all alternatives. Run at
#      p = 5 (not 3) so the old `upper.tri(diag(3))` shortcut would be caught.

test_that("greater BF_20 uses full-dimension prior mask, not diag(3), at p != 3", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]           # p = 5
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "greater")

  d <- .se_dens(fit)
  BF_20_expected <- d$BF_10 * ((1 - pnorm(0, d$post_mean, d$post_sd)) * 2)
  diag(BF_20_expected) <- 0

  od <- upper.tri(sel$BF_20)
  expect_equal(sel$BF_20[od], BF_20_expected[od])
})

test_that("less BF_20 excludes the ~0 prior_sd diagonal from prior_dens", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "less")

  d <- .se_dens(fit)
  BF_20_expected <- d$BF_10 * (pnorm(0, d$post_mean, d$post_sd) * 2)
  diag(BF_20_expected) <- 0

  od <- upper.tri(sel$BF_20)
  expect_equal(sel$BF_20[od], BF_20_expected[od])
})

test_that("exhaustive posterior probs follow Eq. 9 (Bayes factors vs H_u)", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")

  # Eq. 9, Williams & Mulder (2019): the null, positive, and negative Bayes
  # factors are ALL referenced to the unrestricted model H_u. The directional
  # terms are 2*Pr(rho>0|Y) and 2*Pr(rho<0|Y) -- NOT multiplied by the
  # two-sided BF_10 (which would double-count the two-sided evidence and give
  # P(H0) = 1/(1 + 2 BF_10^2) instead of the correct 1/(1 + 2 BF_10)).
  # method = "BF_cut" uses equal 1/3 priors, which cancel.
  d <- .se_dens(fit)
  BF_0u <- 1 / d$BF_10                                    # = post_dens / prior_dens
  BF_1u <- (1 - pnorm(0, d$post_mean, d$post_sd)) * 2
  BF_2u <- pnorm(0, d$post_mean, d$post_sd) * 2
  denom <- BF_0u + BF_1u + BF_2u

  expect_equal(sel$post_prob$prob_zero,    (BF_0u / denom)[upper.tri(BF_0u)])
  expect_equal(sel$post_prob$prob_greater, (BF_1u / denom)[upper.tri(BF_1u)])
  expect_equal(sel$post_prob$prob_less,    (BF_2u / denom)[upper.tri(BF_2u)])
})

test_that("exhaustive prob_zero is NOT the double-counted (vs-H0) form", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")

  # the pre-fix (buggy) formula multiplied the directional BFs by BF_10
  d <- .se_dens(fit)
  BF_null    <- 1 / d$BF_10
  BF_greater <- d$BF_10 * ((1 - pnorm(0, d$post_mean, d$post_sd)) * 2)
  BF_less    <- d$BF_10 * (pnorm(0, d$post_mean, d$post_sd) * 2)
  buggy_null <- (BF_null / (BF_null + BF_greater + BF_less))[upper.tri(BF_null)]

  # they must actually differ on this data (guards the regression is meaningful)
  expect_false(isTRUE(all.equal(sel$post_prob$prob_zero, buggy_null)))
})

test_that("summary.select.explore works for less alternative (no row-count crash)", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "less")

  summ <- summary(sel)
  expect_equal(nrow(summ$summary), choose(ncol(sel$pcor_mat), 2))
  expect_equal(colnames(summ$summary),
               c("Relation", "Post.mean", "Post.sd.fisher", "Pr.H0", "Pr.H1"))
  # Relation labels must be populated, not empty
  expect_true(all(nzchar(as.character(summ$summary$Relation))))
})
