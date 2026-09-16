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
#'        probabilities. For each edge, the posterior is a mixture
#'        distribution placing mass at zero under the null model and using the
#'        posterior under the alternative model otherwise. Reported edges are
#'        based on the median of this mixture.
#'        }
#'
#' @param BF_cut Numeric. Bayes factor threshold for including an edge when
#'        \code{method = "BF_cut"} (defaults to 3). 
#'
#' @param prior.prob.H0 Numeric between 0 and 1. Prior probability assigned
#'        to the null hypothesis for each edge (defaults to \code{0.5}). With
#'        \code{method = "BMA"} it is used for model averaging and the edge
#'        inclusion probabilities. With \code{method = "BF_cut"} it only affects
#'        the reported edge inclusion probabilities (\code{incl_prob}), not the
#'        selected graph, and it is not used for \code{alternative = "exhaustive"}.
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
#' \code{method = "BMA"} performs Bayesian model averaging using a
#' spike-and-slab style mixture distribution for each edge. The spike
#' corresponds to the null hypothesis (exactly zero partial correlation),
#' whereas the slab corresponds to the posterior under the alternative
#' hypothesis, approximated by a normal distribution for the Fisher-z
#' transformed partial correlation (truncated to the positive or negative
#' half-line for one-sided hypotheses). Posterior model probabilities are
#' computed from the Bayes factors and \code{prior.prob.H0}. The selected
#' network is based on the median of this mixture, which is computed exactly
#' (no simulation), so the result is deterministic. For
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
#'
#'  \item \code{incl_prob} Matrix of posterior edge inclusion probabilities,
#'  \eqn{P(H_1 \mid Y)}, based on \code{BF_10} and prior inclusion probability
#'  \code{1 - prior.prob.H0}.
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
#'
#'  \item \code{incl_prob} Matrix of posterior probabilities of the
#'  one-sided hypothesis against the null, based on \code{BF_20} and prior
#'  probability \code{1 - prior.prob.H0}.
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
#' \item \code{incl_prob} Matrix of posterior edge inclusion probabilities,
#'   \eqn{1 - P(H_0 \mid Y)}.
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
  # post_samp arrays have iter + 50 slices; the first 50 are burn-in
  samp_idx   <- 51:(x$iter + 50)

  # Posterior mean/sd of the Fisher-z partial correlations and the prior
  # density at zero (Savage-Dickey). Shared by all branches below.
  # from the draws when stored; otherwise from the running summaries
  # (explore(..., store_post_draws = FALSE))
  if (!is.null(post_samp$fisher_z)) {
    post_sd    <- apply(post_samp$fisher_z[,, samp_idx], 1:2, sd)
    post_mean  <- apply(post_samp$fisher_z[,, samp_idx], 1:2, mean)
  } else {
    post_sd    <- post_samp$z_sd
    post_mean  <- post_samp$z_mean
  }
  post_dens  <- dnorm(0, post_mean, post_sd)
  prior_dens <- dnorm(0, 0, .prior_sd_z(x))

  # With method = "BF_cut", prior.prob.H0 does not affect edge selection
  # (which is driven by BF_cut). For "two.sided", "greater" and "less" it is
  # still used for the reported edge inclusion probabilities (incl_prob); for
  # "exhaustive" (fixed equal 1/3 priors) it is not used at all.
  if (method == "BF_cut" && "prior.prob.H0" %in% names(match.call())) {
    if (alternative == "exhaustive") {
      warning(
        paste0(
          "'prior.prob.H0' is ignored when method = \"BF_cut\" and ",
          "alternative = \"exhaustive\": the three hypotheses have equal prior ",
          "probabilities. Use method = \"BMA\" for 'prior.prob.H0' to take effect."
        ),
        call. = FALSE
      )
    } else {
      message(
        "'prior.prob.H0' only affects the edge inclusion probabilities ",
        "('incl_prob') when method = \"BF_cut\"; edge selection is based on ",
        "BF_cut = ", BF_cut, "."
      )
    }
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
        incl_prob      = .incl_prob(BF_10_mat, prior.prob.H0),
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
        incl_prob      = .incl_prob(BF_20_mat, prior.prob.H0),
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
        incl_prob      = .incl_prob(BF_20_mat, prior.prob.H0),
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
        incl_prob      = .incl_prob_exhaustive(prob_null),
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

    P <- object$p

    # Model-averaged partial correlations: the median of the spike-and-slab
    # mixture with a spike at 0 (mass p0), a positive slab (mass pplus) and a
    # negative slab (mass pminus). The slabs are the posterior of the Fisher-z
    # partial correlation, approximated by N(post_mean, post_sd^2), truncated
    # to z > 0 and z < 0. The median is computed exactly under this
    # approximation (no simulation) and transformed back with tanh:
    #   pminus > 0.5         -> median in the negative slab
    #   pminus + p0 >= 0.5   -> median is 0
    #   otherwise            -> median in the positive slab
    # For the two-sided test the (untruncated) slab with mass 1 - p0 equals a
    # positive and a negative truncated slab with masses (1 - p0) * P(z > 0)
    # and (1 - p0) * P(z < 0). Quantiles are computed on the log scale for
    # numerical stability.
    .bma_median <- function(p0, pplus, pminus) {
      log_lo <- pnorm(0, post_mean, post_sd, log.p = TRUE)
      log_up <- pnorm(0, post_mean, post_sd, lower.tail = FALSE, log.p = TRUE)
      off <- row(post_mean) != col(post_mean)
      neg <- off & ((pminus > 0.5) %in% TRUE)
      pos <- off & !neg & ((pminus + p0 < 0.5) %in% TRUE)
      z   <- matrix(0, P, P)
      z[neg] <- qnorm(log_lo[neg] + log(0.5 / pminus[neg]),
                      post_mean[neg], post_sd[neg], log.p = TRUE)
      q <- (0.5 - pminus[pos] - p0[pos]) / pplus[pos]
      z[pos] <- qnorm(log_up[pos] + log1p(-q),
                      post_mean[pos], post_sd[pos],
                      lower.tail = FALSE, log.p = TRUE)
      tanh(z)
    }

    if (alternative == "two.sided") {

      BF_10_mat  <- prior_dens / post_dens
      BF_01_mat  <- 1 / BF_10_mat
      diag(BF_01_mat) <- 0
      diag(BF_10_mat) <- 0

      edge_excl  <- (BF_01_mat * prior.prob.H0) /
                    (BF_01_mat * prior.prob.H0 + (1 - prior.prob.H0))
      p_lo       <- pnorm(0, post_mean, post_sd)
      bma_matrix <- .bma_median(p0     = edge_excl,
                                pplus  = (1 - edge_excl) * (1 - p_lo),
                                pminus = (1 - edge_excl) * p_lo)

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
        incl_prob      = .incl_prob(BF_10_mat, prior.prob.H0),
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
      bma_matrix <- .bma_median(p0 = edge_excl, pplus = 1 - edge_excl,
                                pminus = 0 * edge_excl)

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
        incl_prob      = .incl_prob(BF_20_mat, prior.prob.H0),
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
      bma_matrix <- .bma_median(p0 = edge_excl, pplus = 0 * edge_excl,
                                pminus = 1 - edge_excl)

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
        incl_prob      = .incl_prob(BF_20_mat, prior.prob.H0),
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

      # Three-state Bayesian model averaging: pcor_mat_zero is the median of
      # the mixture of the null spike at 0, the positive slab and the negative
      # slab, mixed by the posterior hypothesis probabilities.
      bma_matrix <- .bma_median(p0 = prob_null, pplus = prob_greater,
                                pminus = prob_less)

      # Classify each edge by the SIGN of the model-averaged median, not by the
      # single most probable hypothesis. null_mat == 1 means that the
      # model-averaged posterior median is 0. This can occur even when H0 is not
      # the most probable hypothesis, because neither the positive nor the
      # negative side contains more than half of the posterior mixture
      # probability mass (the median is set by probability mass, not by the
      # magnitudes of the slab values). The three matrices are mutually exclusive
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
        incl_prob      = .incl_prob_exhaustive(prob_null),
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

# Prior sd of the Fisher-z partial correlations: analytic (prior_sd_z, set by
# explore()); for objects created by older versions, from the prior draws.
.prior_sd_z <- function(x) {
  if (!is.null(x$prior_sd_z)) return(x$prior_sd_z)
  prior_sd <- apply(x$prior_samp$fisher_z[,, 51:(x$iter + 50)], 1:2, sd)
  mean(prior_sd[upper.tri(prior_sd)])
}

# Posterior edge inclusion probabilities from a Bayes factor against H0,
# P(H1 | Y) = q * BF / (q * BF + 1 - q), with q = 1 - prior.prob.H0.
# Written so that BF = Inf gives 1. Diagonal set to 0.
.incl_prob <- function(BF, prior.prob.H0) {
  pip <- 1 / (1 + prior.prob.H0 / ((1 - prior.prob.H0) * BF))
  diag(pip) <- 0
  pip
}

# Exhaustive test: inclusion probability is P(H+ | Y) + P(H- | Y).
.incl_prob_exhaustive <- function(prob_null) {
  pip <- 1 - prob_null
  diag(pip) <- 0
  pip
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
    prob_H1 <- if (!is.null(x$incl_prob)) x$incl_prob[upper.tri(x$incl_prob)] else
      x$BF_10[upper.tri(x$BF_10)] / (x$BF_10[upper.tri(x$BF_10)] + 1)
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
    prob_H1 <- if (!is.null(x$incl_prob)) x$incl_prob[upper.tri(x$incl_prob)] else
      x$BF_20[upper.tri(x$BF_20)] / (x$BF_20[upper.tri(x$BF_20)] + 1)
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
