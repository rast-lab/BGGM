#' @title Graph selection for \code{explore} Objects
#'
#' @description Provides the selected graph based on the Bayes factor
#' \insertCite{Williams2019_bf}{BGGM}.
#'
#' @name select.explore
#'
#' @param object An object of class \code{explore.default}
#'
#' @param method Character string specifying the edge selection method.
#'        Options include:
#'
#'        \itemize{
#'        \item \code{"BF_cut"}: Select edges based on a Bayes factor threshold.
#'        This is the original approach described in
#'        \insertCite{Williams2019_bf}{BGGM}.
#'
#'        \item \code{"BMA"}: Bayesian model averaging based on posterior model
#'        probabilities. For each edge, posterior draws are generated from a
#'        mixture distribution placing mass at zero under the null model and
#'        using posterior draws from the alternative model otherwise.
#'        Reported edges are based on the posterior median of these draws.
#'        }
#'
#' @param BF_cut Numeric. Bayes factor threshold for including an edge when
#'        \code{method = "BF_cut"} (defaults to 3). 
#'
#' @param prior.prob.H0 Numeric between 0 and 1. Prior probability assigned
#'        to the null hypothesis for each edge when
#'        \code{method = "BMA"} (defaults to \code{0.5}).
#'
#' @param alternative A character string specifying the alternative hypothesis. It
#'                    must be one of "two.sided" (default), "greater", "less",
#'                    or "exhaustive". See note for further details.
#'
#' @param ... Currently ignored.
#'
#' @references
#' \insertAllCited{}
#'
#' @details Exhaustive provides the posterior hypothesis probabilities for
#' a positive, negative, or null relation \insertCite{@see Table 3 in @Williams2019_bf}{BGGM}.
#'
#' \code{method = "BF_cut"} performs edge selection using Bayes factor
#' thresholding. For \code{alternative = "exhaustive"} the threshold is
#' applied to the Bayes factor of each hypothesis (null, positive, negative)
#' against its complement, under equal prior hypothesis probabilities of \code{1/3};
#' because the prior odds against the complement are then \code{1:2}, a cutoff
#' of \code{BF_cut = 3} corresponds to a posterior hypothesis probability of \code{0.6}.
#'
#' \code{method = "BMA"} performs Bayesian model averaging by generating
#' posterior draws from a spike-and-slab style mixture distribution for each
#' edge. The spike corresponds to the null hypothesis (exactly zero partial
#' correlation), whereas the slab corresponds to posterior draws under the
#' alternative hypothesis. Posterior model probabilities are computed from the
#' Bayes factors and \code{prior.prob.H0}. The selected network is based on
#' the posterior median of the resulting draws.
#'
#' @importFrom stats median
#' @importFrom truncnorm rtruncnorm
#'
#' @note Care must be taken with the options \code{alternative = "less"} and
#'       \code{alternative = "greater"}. This is because the full parameter space is not included,
#'       such, for  \code{alternative = "greater"}, there can be evidence for the "null" when
#'       the relation is negative. This inference is correct: the null model better predicted
#'       the data than the positive model. But note this is relative and does \strong{not}
#'       provide absolute evidence for the null hypothesis.
#'
#' @return The returned object of class \code{select.explore} contains a lot of information that
#'         is used for printing and plotting the results. For users of \strong{BGGM}, the following
#'         are the useful objects:
#'
#'
#' \code{alternative = "two.sided"}
#'
#'  \itemize{
#'
#'  \item \code{pcor_mat_zero} Selected partial correlation matrix (weighted adjacency).
#'
#'  \item \code{pcor_mat} Partial correlation matrix (posterior mean).
#'
#'  \item \code{Adj_10} Adjacency matrix for the selected edges.
#'
#'  \item \code{Adj_01} Adjacency matrix for which there was
#'                      evidence for the null hypothesis.
#'  }
#'
#' \code{alternative = "greater"} and \code{"less"}
#'
#'  \itemize{
#'
#'  \item \code{pcor_mat_zero} Selected partial correlation matrix (weighted adjacency).
#'
#'  \item \code{pcor_mat} Partial correlation matrix (posterior mean).
#'
#'  \item \code{Adj_20} Adjacency matrix for the selected edges.
#'
#'  \item \code{Adj_02} Adjacency matrix for which there was
#'                      evidence for the null hypothesis (see note).
#'  }
#'
#' \code{alternative = "exhaustive"}
#'
#' \itemize{
#'
#' \item \code{post_prob} A data frame that included the posterior hypothesis probabilities.
#'
#' \item \code{neg_mat} Adjacency matrix for which there was evidence for negative edges.
#'
#' \item \code{pos_mat} Adjacency matrix for which there was evidence for positive edges.
#'
#' \item \code{neg_mat} Adjacency matrix for which there was
#'                      evidence for the null hypothesis (see note).
#'
#'  \item \code{pcor_mat} Partial correlation matrix (posterior mean). The weighted adjacency
#'  matrices can be computed by multiplying \code{pcor_mat} with an adjacency matrix.
#'
#' }
#'
#' @seealso \code{\link{explore}} and \code{\link{ggm_compare_explore}} for several examples.
#'
#' @examples
#'
#' \donttest{
#' #################
#' ### example 1 ###
#' #################
#'
#' #  data
#' Y <- bfi[,1:10]
#'
#' # fit model
#' fit <- explore(Y, progress = FALSE)
#'
#' # edge set (Bayes factor threshold)
#' E <- select(fit,
#'             alternative = "exhaustive")
#'
#' # edge set (Bayesian model averaging), with prior P(H0) = 0.5
#' E <- select(fit,
#'             method = "BMA",
#'             alternative = "exhaustive")
#'
#' }
#' @export
select.explore <- function(object,
                           method = c("BF_cut", "BMA"),
                           BF_cut = 3,
                           prior.prob.H0 = 0.5,
                           alternative = "two.sided",
                           ...){

  method     <- match.arg(method)
  x          <- object
  hyp_prob   <- BF_cut / (BF_cut + 1)
  post_samp  <- x$post_samp
  prior_samp <- x$prior_samp
  samp_idx   <- 51:x$iter

  if (method == "BF_cut") {

    if (alternative == "two.sided") {

      post_sd    <- apply(post_samp$fisher_z[,, samp_idx], 1:2, sd)
      post_mean  <- apply(post_samp$fisher_z[,, samp_idx], 1:2, mean)
      post_dens  <- dnorm(0, post_mean, post_sd)
      prior_sd   <- apply(prior_samp$fisher_z[,, samp_idx], 1:2, sd)
      prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))

      BF_10_mat <- prior_dens / post_dens
      BF_01_mat <- 1 / BF_10_mat
      diag(BF_01_mat) <- 0
      diag(BF_10_mat) <- 0

      Adj_10 <- ifelse(BF_10_mat > BF_cut, 1, 0)
      Adj_01 <- ifelse(BF_10_mat < 1 / BF_cut, 1, 0)
      diag(Adj_01) <- 0
      diag(Adj_10) <- 0

      returned_object <- list(
        pcor_mat_zero  = tanh(post_mean) * Adj_10,
        pcor_mat       = round(tanh(post_mean), 3),
        pcor_sd_fisher = round(post_sd, 3),
        Adj_10         = Adj_10,
        Adj_01         = Adj_01,
        BF_10          = BF_10_mat,
        BF_01          = BF_01_mat,
        BF_cut         = BF_cut,
        method         = method,
        alternative    = alternative,
        call           = match.call(),
        type           = x$type,
        formula        = x$formula,
        analytic       = x$analytic,
        object         = object
      )

    } else if (alternative == "greater") {

      post_sd    <- apply(post_samp$fisher_z[,, samp_idx], 1:2, sd)
      post_mean  <- apply(post_samp$fisher_z[,, samp_idx], 1:2, mean)
      post_dens  <- dnorm(0, post_mean, post_sd)
      prior_sd   <- apply(prior_samp$fisher_z[,, samp_idx], 1:2, sd)
      prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))

      BF_10_mat <- prior_dens / post_dens
      BF_20_mat <- BF_10_mat * ((1 - pnorm(0, post_mean, post_sd)) * 2)
      BF_02_mat <- 1 / BF_20_mat
      diag(BF_02_mat) <- 0
      diag(BF_20_mat) <- 0

      Adj_20 <- ifelse(BF_20_mat > BF_cut, 1, 0)
      Adj_02 <- ifelse(BF_02_mat > BF_cut, 1, 0)
      diag(Adj_02) <- 0
      diag(Adj_20) <- 0

      returned_object <- list(
        pcor_mat_zero  = tanh(post_mean) * Adj_20,
        pcor_mat       = round(tanh(post_mean), 3),
        pcor_sd_fisher = round(post_sd, 3),
        Adj_20         = Adj_20,
        Adj_02         = Adj_02,
        BF_20          = BF_20_mat,
        BF_02          = BF_02_mat,
        BF_cut         = BF_cut,
        method         = method,
        alternative    = alternative,
        call           = match.call(),
        type           = x$type,
        formula        = x$formula,
        analytic       = x$analytic,
        object         = object
      )

    } else if (alternative == "less") {

      post_sd    <- apply(post_samp$fisher_z[,, samp_idx], 1:2, sd)
      post_mean  <- apply(post_samp$fisher_z[,, samp_idx], 1:2, mean)
      post_dens  <- dnorm(0, post_mean, post_sd)
      prior_sd   <- apply(prior_samp$fisher_z[,, samp_idx], 1:2, sd)
      prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))

      BF_10_mat <- prior_dens / post_dens
      BF_20_mat <- BF_10_mat * (pnorm(0, post_mean, post_sd) * 2)
      BF_02_mat <- 1 / BF_20_mat
      diag(BF_02_mat) <- 0
      diag(BF_20_mat) <- 0

      Adj_20 <- ifelse(BF_20_mat > BF_cut, 1, 0)
      Adj_02 <- ifelse(BF_02_mat > BF_cut, 1, 0)
      diag(Adj_02) <- 0
      diag(Adj_20) <- 0

      returned_object <- list(
        pcor_mat_zero  = tanh(post_mean) * Adj_20,
        pcor_mat       = round(tanh(post_mean), 3),
        pcor_sd_fisher = round(post_sd, 3),
        Adj_20         = Adj_20,
        Adj_02         = Adj_02,
        BF_20          = BF_20_mat,
        BF_02          = BF_02_mat,
        BF_cut         = BF_cut,
        method         = method,
        alternative    = alternative,
        call           = match.call(),
        type           = x$type,
        formula        = x$formula,
        analytic       = x$analytic,
        object         = object
      )

    } else if (alternative == "exhaustive") {

      cn  <- colnames(x$Y)
      p   <- ncol(x$pcor_mat)
      I_p <- diag(p)

      if (is.null(cn)) {
        mat_names <- sapply(1:p, function(z) paste(1:p, z, sep = "--"))[upper.tri(I_p)]
      } else {
        mat_names <- sapply(cn, function(z) paste(cn, z, sep = "--"))[upper.tri(I_p)]
      }

      post_sd    <- apply(post_samp$fisher_z[,, samp_idx], 1:2, sd)
      post_mean  <- apply(post_samp$fisher_z[,, samp_idx], 1:2, mean)
      post_dens  <- dnorm(0, post_mean, post_sd)
      prior_sd   <- apply(prior_samp$fisher_z[,, samp_idx], 1:2, sd)
      prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))

      # Posterior hypothesis probabilities via Eq. 9 of Williams & Mulder
      # (2019). All three Bayes factors are referenced to the unrestricted
      # model H_u: BF_0u is the Savage-Dickey null-vs-unrestricted ratio
      # (Eq. 6), while BF_1u / BF_2u are the one-sided-vs-unrestricted ratios
      # (Eq. 8) -- these must NOT be multiplied by the two-sided BF_10, which
      # would put them on the vs-H0 baseline and double-count the two-sided
      # evidence. method = "BF_cut" assigns equal prior probabilities (1/3)
      # to each hypothesis, which cancel in the normalisation below.
      BF_0u <- post_dens / prior_dens
      BF_1u <- (1 - pnorm(0, post_mean, post_sd)) * 2
      BF_2u <- pnorm(0, post_mean, post_sd) * 2

      denom        <- BF_0u + BF_1u + BF_2u
      prob_null    <- BF_0u / denom
      prob_greater <- BF_1u / denom
      prob_less    <- BF_2u / denom

      prob_dat <- data.frame(
        edge         = mat_names,
        prob_zero    = prob_null[upper.tri(prob_null)],
        prob_greater = prob_greater[upper.tri(prob_greater)],
        prob_less    = prob_less[upper.tri(prob_less)]
      )
      row.names(prob_dat) <- c()

      # Selection thresholds the Bayes factor of each hypothesis against its
      # COMPLEMENT against BF_cut, so BF_cut literally means "Bayes factor >
      # BF_cut" (as the argument name implies). Under the exhaustive test's
      # equal 1/3 priors, the prior odds of H_k versus its complement are 1:2,
      # so BF_{k,!k} = 2 * P(H_k|Y) / (1 - P(H_k|Y)). Note this is NOT the
      # posterior-probability threshold BF_cut/(BF_cut+1): e.g. BF_cut = 3
      # corresponds to P(H_k|Y) > 0.6, not 0.75.
      BF_null_comp <- 2 * prob_null    / (1 - prob_null)
      BF_pos_comp  <- 2 * prob_greater / (1 - prob_greater)
      BF_neg_comp  <- 2 * prob_less    / (1 - prob_less)

      null_mat <- ifelse(BF_null_comp > BF_cut, 1, 0)
      pos_mat  <- ifelse(BF_pos_comp  > BF_cut, 1, 0)
      neg_mat  <- ifelse(BF_neg_comp  > BF_cut, 1, 0)

      returned_object <- list(
        post_prob      = prob_dat,
        neg_mat        = neg_mat,
        pos_mat        = pos_mat,
        null_mat       = null_mat,
        alternative    = alternative,
        pcor_mat       = round(tanh(post_mean), 3),
        pcor_sd_fisher = round(post_sd, 3),
        call           = match.call(),
        prob           = hyp_prob,
        method         = method,
        type           = x$type,
        formula        = x$formula,
        analytic       = x$analytic,
        object         = object
      )

    } else {
      stop("alternative not supported. see documentation")
    }

  } else {
    # BMA

    P        <- object$p
    indices  <- which(lower.tri(diag(P), diag = FALSE), arr.ind = TRUE)
    num_pcor <- P * (P - 1) / 2

    post_sd   <- apply(post_samp$fisher_z[,, samp_idx], 1:2, sd)
    post_mean <- apply(post_samp$fisher_z[,, samp_idx], 1:2, mean)
    post_dens <- dnorm(0, post_mean, post_sd)
    prior_sd  <- apply(prior_samp$fisher_z[,, samp_idx], 1:2, sd)

    .bma_matrix <- function(excl_vec, incl_vec, draw_fn) {
      bma_draws <- do.call(cbind, lapply(seq_len(num_pcor), function(e) {
        d        <- sample(c(0, 1), size = x$iter,
                           prob = c(excl_vec[e], incl_vec[e]), replace = TRUE)
        incl_pos <- which(d == 1)
        if (length(incl_pos) > 0) d[incl_pos] <- draw_fn(e, incl_pos)
        d
      }))
      medians <- apply(bma_draws, 2, median)
      m <- matrix(0, P, P)
      for (i in seq_len(nrow(indices))) {
        m[indices[i, 1], indices[i, 2]] <- medians[i]
        m[indices[i, 2], indices[i, 1]] <- medians[i]
      }
      m
    }

    if (alternative == "two.sided") {

      prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))
      BF_10_mat  <- prior_dens / post_dens
      BF_01_mat  <- 1 / BF_10_mat
      diag(BF_01_mat) <- 0
      diag(BF_10_mat) <- 0

      edge_excl  <- (BF_01_mat * prior.prob.H0) /
                    (BF_01_mat * prior.prob.H0 + (1 - prior.prob.H0))
      excl_vec   <- edge_excl[lower.tri(diag(P))]
      incl_vec   <- 1 - excl_vec

      bma_matrix <- .bma_matrix(excl_vec, incl_vec, function(e, incl_pos) {
        object$post_samp$pcors[
          indices[e, 1], indices[e, 2],
          sample(samp_idx, size = length(incl_pos), replace = TRUE)
        ]
      })

      Adj_10 <- ifelse(bma_matrix != 0, 1, 0)
      Adj_01 <- ifelse(bma_matrix == 0, 1, 0)
      diag(Adj_01) <- 0
      diag(Adj_10) <- 0

      returned_object <- list(
        pcor_mat_zero  = bma_matrix,
        pcor_mat       = round(tanh(post_mean), 3),
        pcor_sd_fisher = round(post_sd, 3),
        Adj_10         = Adj_10,
        Adj_01         = Adj_01,
        BF_10          = BF_10_mat,
        BF_01          = BF_01_mat,
        BF_cut         = NA,
        prior.prob.H0  = prior.prob.H0,
        method         = method,
        alternative    = alternative,
        call           = match.call(),
        type           = x$type,
        formula        = x$formula,
        analytic       = x$analytic,
        object         = object
      )

    } else if (alternative == "greater") {

      prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))
      BF_10_mat  <- prior_dens / post_dens
      BF_20_mat  <- BF_10_mat * ((1 - pnorm(0, post_mean, post_sd)) * 2)
      BF_02_mat  <- 1 / BF_20_mat
      diag(BF_02_mat) <- 0
      diag(BF_20_mat) <- 0

      edge_excl <- (BF_02_mat * prior.prob.H0) /
                   (BF_02_mat * prior.prob.H0 + (1 - prior.prob.H0))
      excl_vec  <- edge_excl[lower.tri(diag(P))]
      incl_vec  <- 1 - excl_vec

      bma_matrix <- .bma_matrix(excl_vec, incl_vec, function(e, incl_pos) {
        pos_idx <- which(object$post_samp$pcors[indices[e, 1], indices[e, 2], samp_idx] > 0)
        if (length(pos_idx) > 0) {
          object$post_samp$pcors[
            indices[e, 1], indices[e, 2],
            sample(samp_idx[pos_idx], size = length(incl_pos), replace = TRUE)
          ]
        } else {
          tanh(truncnorm::rtruncnorm(length(incl_pos),
                          mean = post_mean[indices[e, 1], indices[e, 2]],
                          sd   = post_sd[indices[e, 1], indices[e, 2]],
                          a    = 0))
        }
      })

      Adj_20 <- ifelse(bma_matrix != 0, 1, 0)
      Adj_02 <- ifelse(bma_matrix == 0, 1, 0)
      diag(Adj_02) <- 0
      diag(Adj_20) <- 0

      returned_object <- list(
        pcor_mat_zero  = bma_matrix,
        pcor_mat       = round(tanh(post_mean), 3),
        pcor_sd_fisher = round(post_sd, 3),
        Adj_20         = Adj_20,
        Adj_02         = Adj_02,
        BF_20          = BF_20_mat,
        BF_02          = BF_02_mat,
        BF_cut         = NA,
        prior.prob.H0  = prior.prob.H0,
        method         = method,
        alternative    = alternative,
        call           = match.call(),
        type           = x$type,
        formula        = x$formula,
        analytic       = x$analytic,
        object         = object
      )

    } else if (alternative == "less") {

      prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))
      BF_10_mat  <- prior_dens / post_dens
      BF_20_mat  <- BF_10_mat * (pnorm(0, post_mean, post_sd) * 2)
      BF_02_mat  <- 1 / BF_20_mat
      diag(BF_02_mat) <- 0
      diag(BF_20_mat) <- 0

      edge_excl <- (BF_02_mat * prior.prob.H0) /
                   (BF_02_mat * prior.prob.H0 + (1 - prior.prob.H0))
      excl_vec  <- edge_excl[lower.tri(diag(P))]
      incl_vec  <- 1 - excl_vec

      bma_matrix <- .bma_matrix(excl_vec, incl_vec, function(e, incl_pos) {
        neg_idx <- which(object$post_samp$pcors[indices[e, 1], indices[e, 2], samp_idx] < 0)
        if (length(neg_idx) > 0) {
          object$post_samp$pcors[
            indices[e, 1], indices[e, 2],
            sample(samp_idx[neg_idx], size = length(incl_pos), replace = TRUE)
          ]
        } else {
          tanh(truncnorm::rtruncnorm(length(incl_pos),
                          mean = post_mean[indices[e, 1], indices[e, 2]],
                          sd   = post_sd[indices[e, 1], indices[e, 2]],
                          b    = 0))
        }
      })

      Adj_20 <- ifelse(bma_matrix != 0, 1, 0)
      Adj_02 <- ifelse(bma_matrix == 0, 1, 0)
      diag(Adj_02) <- 0
      diag(Adj_20) <- 0

      returned_object <- list(
        pcor_mat_zero  = bma_matrix,
        pcor_mat       = round(tanh(post_mean), 3),
        pcor_sd_fisher = round(post_sd, 3),
        Adj_20         = Adj_20,
        Adj_02         = Adj_02,
        BF_20          = BF_20_mat,
        BF_02          = BF_02_mat,
        BF_cut         = NA,
        prior.prob.H0  = prior.prob.H0,
        method         = method,
        alternative    = alternative,
        call           = match.call(),
        type           = x$type,
        formula        = x$formula,
        analytic       = x$analytic,
        object         = object
      )

    } else if (alternative == "exhaustive") {

      cn  <- colnames(x$Y)
      p   <- ncol(x$pcor_mat)
      I_p <- diag(p)

      if (is.null(cn)) {
        mat_names <- sapply(1:p, function(z) paste(1:p, z, sep = "--"))[upper.tri(I_p)]
      } else {
        mat_names <- sapply(cn, function(z) paste(cn, z, sep = "--"))[upper.tri(I_p)]
      }

      prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))

      # Posterior hypothesis probabilities via Eq. 9 of Williams & Mulder
      # (2019), with the three Bayes factors referenced to the unrestricted
      # model H_u (see the method = "BF_cut" branch above for the baseline
      # rationale). Unlike "BF_cut" (equal 1/3 priors), the null hypothesis
      # is given prior probability prior.prob.H0 and the two directional
      # hypotheses split the remainder equally.
      BF_0u <- post_dens / prior_dens
      BF_1u <- (1 - pnorm(0, post_mean, post_sd)) * 2
      BF_2u <- pnorm(0, post_mean, post_sd) * 2

      prior_H0 <- prior.prob.H0
      prior_H1 <- prior_H2 <- (1 - prior.prob.H0) / 2

      denom        <- prior_H0 * BF_0u + prior_H1 * BF_1u + prior_H2 * BF_2u
      prob_null    <- prior_H0 * BF_0u / denom
      prob_greater <- prior_H1 * BF_1u / denom
      prob_less    <- prior_H2 * BF_2u / denom

      prob_dat <- data.frame(
        edge         = mat_names,
        prob_zero    = prob_null[upper.tri(prob_null)],
        prob_greater = prob_greater[upper.tri(prob_greater)],
        prob_less    = prob_less[upper.tri(prob_less)]
      )
      row.names(prob_dat) <- c()

      # Hard assignment: each edge is placed in the state (null, positive, or
      # negative) with the largest posterior probability, so every edge
      # belongs to exactly one state -- unlike "BF_cut", where an edge may
      # belong to none when no probability exceeds the threshold.
      largest        <- pmax(prob_null, prob_greater, prob_less)
      null_mat       <- 1 * (prob_null    == largest)
      pos_mat        <- 1 * (prob_greater == largest)
      neg_mat        <- 1 * (prob_less    == largest)
      diag(null_mat) <- 0
      diag(pos_mat)  <- 0
      diag(neg_mat)  <- 0

      returned_object <- list(
        post_prob      = prob_dat,
        neg_mat        = neg_mat,
        pos_mat        = pos_mat,
        null_mat       = null_mat,
        alternative    = alternative,
        pcor_mat       = round(tanh(post_mean), 3),
        pcor_sd_fisher = round(post_sd, 3),
        call           = match.call(),
        prob           = NA,
        prior.prob.H0  = prior.prob.H0,
        method         = method,
        type           = x$type,
        formula        = x$formula,
        analytic       = x$analytic,
        object         = object
      )

    } else {
      stop("alternative not supported. see documentation")
    }
  }

  class(returned_object) <- c("BGGM", "select.explore", "explore", "select")
  returned_object
}




print_select_explore <- function(x,
                                 ...){

  p <- ncol(x$pcor_mat_zero)
  cat("BGGM: Bayesian Gaussian Graphical Models \n")
  cat("--- \n")
  cat("Type:", x$type, "\n")
  cat("Analytic:", x$analytic, "\n")
  cat("Formula:", paste(as.character(x$formula), collapse = " "), "\n")
  cat("Alternative:", x$alternative, "\n")
  cat("Method:", if (is.null(x$method)) "BF_cut" else x$method, "\n")
  if (x$alternative == "two.sided") {
    if (!is.null(x$method) && x$method == "BMA") {
      cat("Prior P(H0):", x$prior.prob.H0, "\n")
    } else {
      cat("Bayes Factor:", x$BF_cut, "\n")
    }
  }
  cat("--- \n")
  cat("Call:\n")
  print(x$call)
  cat("--- \n")
  cat("Hypotheses: \n")

  if(x$alternative == "two.sided"){

    cat("H0: rho = 0\nH1: rho != 0", "\n")
    cat("--- \n")
    colnames(x$Adj_10) <- 1:p
    row.names(x$Adj_10) <- 1:p
    colnames( x$pcor_mat_zero) <- 1:p
    row.names(x$pcor_mat_zero) <- 1:p
    cat("Partial Correlations:\n\n")
    print(round(x$pcor_mat_zero, 2))
    cat("--- \n")
    cat("Adjacency:\n\n")
    print(x$Adj_10)
    cat("--- \n")
  } else if (x$alternative == "greater"){

    cat("H0: rho = 0\nH1: rho > 0", "\n")
    cat("--- \n")
    colnames(x$Adj_20) <- 1:p
    row.names(x$Adj_20) <- 1:p
    colnames( x$pcor_mat_zero) <- 1:p
    row.names(x$pcor_mat_zero) <- 1:p
    cat("Partial Correlations:\n\n")
    print(round(x$pcor_mat_zero, 2))
    cat("--- \n")
    cat("Adjacency:\n\n")
    print(x$Adj_20)
    cat("--- \n")

  } else if (x$alternative == "less"){

    cat("H0: rho = 0\nH1: rho < 0", "\n")
    cat("--- \n")
    colnames(x$Adj_20) <- 1:p
    row.names(x$Adj_20) <- 1:p
    colnames( x$pcor_mat_zero) <- 1:p
    row.names(x$pcor_mat_zero) <- 1:p
    cat("Partial Correlations:\n\n")
    print(round(x$pcor_mat_zero, 2))
    cat("--- \n")
    cat("Adjacency:\n\n")
    print(x$Adj_20)
    cat("--- \n")
  } else {

    cat("H0: rho = 0\nH1: rho > 0\nH2: rho < 0", "\n")
    cat("--- \n")
    cat("Summary:\n\n")
    dat <- x$post_prob
    dat$prob_zero <- round(dat$prob_zero, 3)
    dat$prob_greater <- round(dat$prob_greater, 3)
    dat$prob_less <- round(dat$prob_less, 3)
    colnames(dat) <- c("Relation", "Pr.H0", "Pr.H1", "Pr.H2")
    print(dat, row.names = FALSE, right = FALSE)
    cat("--- \n")
  }
}



#' @title   Summary Method for \code{select.explore} Objects
#'
#' @name summary.select.explore
#'
#' @param object object of class \code{select.explore}.
#'
#' @param col_names Logical.
#'
#' @param ... Currently ignored.
#'
#' @examples
#' \donttest{
#' #  data
#' Y <- bfi[,1:10]
#'
#' # fit model
#' fit <- explore(Y, iter = 250,
#'                progress = FALSE)
#'
#' # edge set
#' E <- select(fit,
#'             alternative = "exhaustive")
#'
#' summary(E)
#'
#' }
#' @return a data frame including the posterior mean, standard deviation,
#' and posterior hypothesis probabilities for each relation.
#' @export
summary.select.explore <- function(object,
                                   col_names = TRUE,
                                   ...){

  x <- object

  p <- ncol(x$pcor_mat)

  I_p <- diag(p)

  # column names
  cn <-  colnames(object$object$Y)


  if(!isTRUE(col_names) | is.null(cn)){

    mat_names <- sapply(1:p , function(x) paste(1:p, x, sep = "--"))[upper.tri(I_p)]

  } else {


    mat_names <-  sapply(cn , function(x) paste(cn, x, sep = "--"))[upper.tri(I_p)]

  }



  if(x$alternative == "two.sided"){

    post_mean <- x$pcor_mat[upper.tri(x$pcor_mat)]
    post_sd <-  x$pcor_sd_fisher[upper.tri(x$pcor_sd_fisher)]
    prob_H1 <- x$BF_10[upper.tri(x$BF_10)] / (x$BF_10[upper.tri(x$BF_10)] + 1)
    prob_H0 <- 1 - prob_H1
    summ <-  data.frame(
      Relation = mat_names,
      Post.mean = post_mean,
      Post.sd.fisher = post_sd,
      Pr.H0 = round(prob_H0, 3),
      Pr.H1 = round(prob_H1, 3)
    )

  } else if (x$alternative == "greater" | x$alternative == "less"){

    post_mean <- x$pcor_mat[upper.tri(x$pcor_mat)]
    post_sd <-  x$pcor_sd_fisher[upper.tri(x$pcor_sd_fisher)]
    prob_H1 <- x$BF_20[upper.tri(x$BF_20)] / (x$BF_20[upper.tri(x$BF_20)] + 1)
    prob_H0 <- 1 - prob_H1
    summ <-  data.frame(
      Relation = mat_names,
      Post.mean = post_mean,
      Post.sd.fisher = post_sd,
      Pr.H0 = round(prob_H0, 3),
      Pr.H1 = round(prob_H1, 3)
    )

  } else {

    summ <- cbind.data.frame( x$post_prob[,1],
                              x$pcor_mat[upper.tri(x$pcor_mat)],
                              x$pcor_sd_fisher[upper.tri(x$pcor_sd_fisher)],
                              round(x$post_prob[,2:4], 3))

    colnames(summ) <- c("Relation",
                        "Post.mean",
                        "Post.sd.fisher",
                        "Pr.H0",
                        "Pr.H1",
                        "Pr.H2")


  }

  returned_object <- list(summary = summ, object = object)

  class(returned_object) <- c("BGGM", "summary.select.explore",
                              "explore", "select.explore",
                              "summary")
  returned_object


}



print_summary_select_explore <- function(x,...){

  cat("BGGM: Bayesian Gaussian Graphical Models \n")
  cat("--- \n")
  cat("Type:", x$object$type, "\n")
  cat("Alternative:", x$object$alternative, "\n")
  cat("--- \n")
  cat("Call:\n")
  print(x$object$call)
  cat("--- \n")
  cat("Hypotheses: \n")

  if(x$object$alternative == "two.sided"){

    cat("H0: rho = 0\nH1: rho != 0", "\n")

  } else if (x$object$alternative == "greater"){

    cat("H0: rho = 0\nH1: rho > 0", "\n")

  } else if (x$object$alternative == "less"){

    cat("H0: rho = 0\nH1: rho < 0", "\n")

  } else {

    cat("H0: rho = 0\nH1: rho > 0\nH2: rho < 0", "\n")

  }

  cat("--- \n\n")

  print(x$summary, right = FALSE, row.names = FALSE)


}


#' @title Plot \code{summary.select.explore} Objects
#'
#' @name plot.summary.select.explore
#'
#' @description Visualize the posterior hypothesis probabilities.
#'
#' @param x An object of class \code{summary.select.explore}
#'
#' @param size Numeric. The size for the points (defaults to 2).
#'
#' @param color Character string. The Color for the points
#'
#' @param ... Currently ignored
#'
#' @return A \code{ggplot} object
#'
#' @examples
#' \donttest{
#' #  data
#' Y <- bfi[,1:10]
#'
#' # fit model
#' fit <- explore(Y, iter = 250,
#'                progress = FALSE)
#'
#' # edge set
#' E <- select(fit,
#'             alternative = "exhaustive")
#'
#' plot(summary(E))
#'
#' }
#' @export
plot.summary.select.explore <- function(x,
                                        size = 2,
                                        color = "black",
                                        ...){


  dat_temp <- x$summary[order(x$summary$Pr.H1,
                              decreasing = F), ]

  dat_temp$Relation <-
    factor(dat_temp$Relation,
           levels = dat_temp$Relation,
           labels = dat_temp$Relation)


  ggplot(dat_temp,
         aes(x = Relation,
             y = Pr.H1)) +
    geom_point(size = size, color = color) +

    theme(axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1
    ))

}
