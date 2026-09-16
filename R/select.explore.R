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
#' thresholding. For \code{alternative = "exhaustive"}, \code{BF_cut} is
#' translated into a cutoff for the posterior hypothesis probabilities: a
#' hypothesis (null, positive, negative) is selected when its posterior odds
#' against the other two hypotheses combined exceed \code{BF_cut}, i.e. when
#' its posterior probability exceeds \code{BF_cut / (BF_cut + 1)} (0.75 for
#' \code{BF_cut = 3}). The three hypotheses have equal prior probabilities
#' (\code{1/3}), so the prior odds against the complement are \code{1:2} and this
#' cutoff corresponds to a Bayes factor of \code{2 * BF_cut} against the complement.
#' An edge can be assigned to none of the three hypotheses.
#'
#' \code{method = "BMA"} performs Bayesian model averaging by generating
#' posterior draws from a spike-and-slab style mixture distribution for each
#' edge. The spike corresponds to the null hypothesis (exactly zero partial
#' correlation), whereas the slab corresponds to posterior draws under the
#' alternative hypothesis. Posterior model probabilities are computed from the
#' Bayes factors and \code{prior.prob.H0}. The selected network is based on
#' the posterior median of the resulting draws. For
#' \code{alternative = "exhaustive"} the mixture has three states -- a spike at
#' zero (\eqn{H_0}), a positive slab (\eqn{H_+}), and a negative slab
#' (\eqn{H_-}) -- mixed by the posterior hypothesis probabilities. The
#' model-averaged partial correlations are returned in \code{pcor_mat_zero}, and
#' \code{pos_mat}/\code{neg_mat}/\code{null_mat} classify each edge by the sign
#' of that model-averaged median.
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
#' \item \code{post_prob} A data frame of the posterior hypothesis probabilities
#'                        \eqn{P(H_0 \mid Y)}, \eqn{P(H_+ \mid Y)}, and
#'                        \eqn{P(H_- \mid Y)} for each relation (a null, positive,
#'                        or negative partial correlation).
#'
#'  For \code{method = "BF_cut"} the following are hard hypothesis assignments;
#'  for \code{method = "BMA"} they classify the sign of the model-averaged
#'  posterior median (\code{pcor_mat_zero}), not the most probable hypothesis:
#'
#' \item \code{pos_mat} Adjacency matrix for positive edges.
#'
#' \item \code{neg_mat} Adjacency matrix for negative edges.
#'
#' \item \code{null_mat} Adjacency matrix for null edges (see note).
#'
#'  \item \code{pcor_mat} Partial correlation matrix (posterior mean). The weighted adjacency
#'  matrices can be computed by multiplying \code{pcor_mat} with an adjacency matrix.
#'
#'  \item \code{pcor_mat_zero} (\code{method = "BMA"} only) Model-averaged
#'  partial-correlation matrix. For each edge this is the posterior median of the
#'  three-state mixture over the null, positive, and negative hypotheses.
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
  post_samp  <- x$post_samp
  prior_samp <- x$prior_samp
  # post_samp arrays have iter + 50 slices; the first 50 are burn-in
  samp_idx   <- 51:(x$iter + 50)

  # Posterior mean/sd of the Fisher-z partial correlations and the prior
  # density at zero (Savage-Dickey). Shared by all branches below.
  post_sd    <- apply(post_samp$fisher_z[,, samp_idx], 1:2, sd)
  post_mean  <- apply(post_samp$fisher_z[,, samp_idx], 1:2, mean)
  post_dens  <- dnorm(0, post_mean, post_sd)
  prior_sd   <- apply(prior_samp$fisher_z[,, samp_idx], 1:2, sd)
  prior_dens <- dnorm(0, 0, mean(prior_sd[upper.tri(prior_sd)]))

  # prior.prob.H0 only affects method = "BMA". If the user explicitly set it
  # while using method = "BF_cut", it has no effect on the result -- selection
  # is driven entirely by the BF_cut threshold -- so remind them rather than
  # let it pass silently.
  if (method == "BF_cut" && "prior.prob.H0" %in% names(match.call())) {
    warning(
      paste0(
        "'prior.prob.H0' is ignored when method = \"BF_cut\": edge selection ",
        "is based on the 'BF_cut' Bayes-factor threshold (BF_cut = ", BF_cut,
        "). Use method = \"BMA\" for 'prior.prob.H0' to take effect."
      ),
      call. = FALSE
    )
  }

  if (method == "BF_cut") {

    if (alternative == "two.sided") {

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

      # diagonal: post_sd = 0 gives Inf/NaN; not an edge
      diag(prob_null) <- diag(prob_greater) <- diag(prob_less) <- 0

      prob_dat <- data.frame(
        edge         = mat_names,
        prob_zero    = prob_null[upper.tri(prob_null)],
        prob_greater = prob_greater[upper.tri(prob_greater)],
        prob_less    = prob_less[upper.tri(prob_less)]
      )
      row.names(prob_dat) <- c()

      # Selection: a hypothesis is selected when its posterior odds against
      # the other two hypotheses combined exceed BF_cut, i.e. when
      # P(H_k|Y) > BF_cut / (BF_cut + 1) (0.75 for BF_cut = 3). With equal
      # prior probabilities (1/3) the prior odds against the complement are
      # 1:2, so this corresponds to a Bayes factor of 2 * BF_cut against the
      # complement.
      hyp_prob <- BF_cut / (BF_cut + 1)

      null_mat <- ifelse(prob_null    > hyp_prob, 1, 0)
      pos_mat  <- ifelse(prob_greater > hyp_prob, 1, 0)
      neg_mat  <- ifelse(prob_less    > hyp_prob, 1, 0)

      returned_object <- list(
        post_prob      = prob_dat,
        neg_mat        = neg_mat,
        pos_mat        = pos_mat,
        null_mat       = null_mat,
        alternative    = alternative,
        pcor_mat       = round(tanh(post_mean), 3),
        pcor_sd_fisher = round(post_sd, 3),
        call           = match.call(),
        # posterior-probability threshold used for selection
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

    # Restricted-posterior slab draws for edge `e`: n draws from the positive
    # (> 0) or negative (< 0) part of the posterior partial correlation. When
    # the sampled posterior has no draw on the required side, fall back to a
    # truncated-normal approximation on the Fisher-z scale. Shared by the
    # "greater", "less", and "exhaustive" BMA branches so their slab logic
    # cannot drift apart.
    .draw_positive_slab <- function(e, n) {
      pos_idx <- which(object$post_samp$pcors[indices[e, 1], indices[e, 2], samp_idx] > 0)
      if (length(pos_idx) > 0) {
        object$post_samp$pcors[
          indices[e, 1], indices[e, 2],
          sample(samp_idx[pos_idx], size = n, replace = TRUE)
        ]
      } else {
        tanh(truncnorm::rtruncnorm(n,
                        mean = post_mean[indices[e, 1], indices[e, 2]],
                        sd   = post_sd[indices[e, 1], indices[e, 2]],
                        a    = 0))
      }
    }

    .draw_negative_slab <- function(e, n) {
      neg_idx <- which(object$post_samp$pcors[indices[e, 1], indices[e, 2], samp_idx] < 0)
      if (length(neg_idx) > 0) {
        object$post_samp$pcors[
          indices[e, 1], indices[e, 2],
          sample(samp_idx[neg_idx], size = n, replace = TRUE)
        ]
      } else {
        tanh(truncnorm::rtruncnorm(n,
                        mean = post_mean[indices[e, 1], indices[e, 2]],
                        sd   = post_sd[indices[e, 1], indices[e, 2]],
                        b    = 0))
      }
    }

    # Three-state spike-and-slab BMA: the exhaustive analogue of .bma_matrix.
    # For each edge, draw a hypothesis state H0 / H+ / H- with probabilities
    # (p0, pplus, pminus) for each of x$iter draws, then set the draw to an
    # exact 0 (H0), a positive slab draw (H+), or a negative slab draw (H-).
    # The per-edge posterior median of this mixture is the model-averaged
    # estimate -- it integrates over all three hypotheses rather than picking
    # the single most probable one.
    .bma_matrix_3state <- function(p0_vec, pplus_vec, pminus_vec) {
      bma_draws <- do.call(cbind, lapply(seq_len(num_pcor), function(e) {
        state <- sample(c(0L, 1L, 2L), size = x$iter, replace = TRUE,
                        prob = c(p0_vec[e], pplus_vec[e], pminus_vec[e]))
        vals  <- numeric(x$iter)
        pos   <- which(state == 1L)
        neg   <- which(state == 2L)
        if (length(pos) > 0) vals[pos] <- .draw_positive_slab(e, length(pos))
        if (length(neg) > 0) vals[neg] <- .draw_negative_slab(e, length(neg))
        vals
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
        .draw_positive_slab(e, length(incl_pos))
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
        .draw_negative_slab(e, length(incl_pos))
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

      # Genuine three-state Bayesian model averaging: draw a spike-and-slab
      # mixture with the null spike at 0, a positive slab, and a negative slab,
      # mixed by the posterior hypothesis probabilities (prob_null,
      # prob_greater, prob_less). pcor_mat_zero is the per-edge posterior
      # median of that mixture -- the model-averaged estimate, the exhaustive
      # analogue of the two-state BMA output.
      p0_vec     <- prob_null[lower.tri(diag(P))]
      pplus_vec  <- prob_greater[lower.tri(diag(P))]
      pminus_vec <- prob_less[lower.tri(diag(P))]

      bma_matrix <- .bma_matrix_3state(p0_vec, pplus_vec, pminus_vec)

      # Classify each edge by the SIGN of the model-averaged median, not by the
      # single most probable hypothesis. null_mat == 1 means that the
      # model-averaged posterior median is 0. This can occur even when H0 is not
      # the most probable hypothesis, because neither the positive nor the
      # negative side contains more than half of the posterior mixture
      # probability mass (the median is set by probability mass, not by the
      # magnitudes of the slab draws). The three matrices are mutually exclusive
      # by construction (a real number is > 0, < 0, or == 0), so this also
      # removes the tie ambiguity of an argmax over equal probabilities.
      pos_mat        <- 1 * (bma_matrix > 0)
      neg_mat        <- 1 * (bma_matrix < 0)
      null_mat       <- 1 * (bma_matrix == 0)
      diag(null_mat) <- 0
      diag(pos_mat)  <- 0
      diag(neg_mat)  <- 0

      returned_object <- list(
        post_prob      = prob_dat,
        pcor_mat_zero  = bma_matrix,
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
