#' GGM: Missing Data
#'
#' Estimation and exploratory hypothesis testing with missing data.
#'
#' @param x An object of class \code{mid} \code{\link[mice]{mice}}.
#'
#' @param method Character string. Which method should be used (default set to \code{estimate})? The current
#'               options are \code{"estimate"} and \code{"explore"}.
#'
#' @param iter  Number of iterations for each imputed dataset (posterior samples; defaults to 2000).
#'
#' @param ... Additional arguments passed to either
#'            \code{\link{estimate}} or \code{\link{explore}}.
#'
#' @return An object of class \code{estimate} or \code{explore}.
#' @export
#'
#' @note Currently, \strong{BGGM} is compatible with the package \code{\link[mice]{mice}} for handling
#'       the missing data. This is accomplished by fitting a model for each imputed dataset
#'       (i.e., more than one to account for uncertainty in the imputation step) and then pooling
#'       the estimates.
#'
#'       In a future version, an additional option will be added that allows for
#'       imputing the missing values during model fitting. This option will be incorporated directly into
#'       the \code{\link{estimate}} or \code{\link{explore}} functions, such that \code{bggm_missing} will
#'       always support missing data with \code{\link[mice]{mice}}.
#'
#'
#' \strong{Support}:
#'
#'  There is limited support for missing data. As of version \code{2.0.0}, it is possible to
#'  determine the graphical structure with either  \code{\link{estimate}} or \code{\link{explore}}, in addition
#'  to plotting the graph with \code{\link{plot.select}}. All data types \emph{are} currently supported.
#'
#' \strong{Memory Warning}:
#'  A model is fitted for each imputed dataset. This results in a potentially large object.
#'
#' @examples
#' \donttest{
#' # note: iter = 250 for demonstrative purposes
#'
#' # need this package
#' library(mice, warn.conflicts = FALSE)
#'
#' # data
#' Y <- ptsd[,1:5]
#'
#' # matrix for indices
#' mat <- matrix(0, nrow = 221, ncol = 5)
#'
#' # indices
#' indices <- which(mat == 0, arr.ind = TRUE)
#'
#' # Introduce 50 NAs
#' Y[indices[sample(1:nrow(indices), 50),]] <- NA
#'
#' # impute
#' x <- mice(Y, m = 5, print = FALSE)
#'
#' #########################
#' #######   copula    #####
#' #########################
#' # rank based parital correlations
#'
#' # estimate the model 
#' fit_est <-  bggm_missing(x,
#'                          method = "estimate",
#'                          type =  "mixed",
#'                          iter = 250,
#'                          progress = FALSE,
#'                          seed = 1234)
#'
#' # select edge set
#' E <- select(fit_est)
#'
#' # plot E
#' plt_E <- plot(E)$plt
#'
#' plt_E
#'}
bggm_missing <- function(x, iter = 2000,
                         method = "estimate", ...){

  # check for mice
  if(!requireNamespace("mice", quietly = TRUE)) {
    stop("Please install the '", "mice", "' package.")
  }
  # check for abind
  if(!requireNamespace("abind", quietly = TRUE)) {
    stop("Please install the '", "abind", "' package.")
  }

  # combine data in long format
  data_sets <- mice::complete(x, action = "long")

  # number of data sets
  n_data_sets <- length(unique(data_sets$.imp))

  # Remove .id by name because its position changed in mice >= 3.17.0
  # (.imp is removed later).
  Y <- data_sets[, !(names(data_sets) %in% ".id"), drop = FALSE]

  if(method == "explore"){

    # fit the models
    # the posterior draws of the imputed data sets are combined below, so
    # they must be stored
    dots <- list(...)
    if (isFALSE(dots$store_post_draws)) {
      warning("'store_post_draws = FALSE' is ignored by bggm_missing(): ",
              "the posterior draws of the imputed data sets are combined.",
              call. = FALSE)
    }
    dots$store_post_draws <- NULL

    fits <- lapply(1:n_data_sets, function(x)
      do.call(explore, c(list(Y = as.matrix(subset(Y, .imp == x)[,!(names(Y) %in% ".imp")]),
                              iter = iter,
                              impute = FALSE,
                              store_post_draws = TRUE),
                         dots)))

  }

  # estimate models
  if(method == "estimate"){

    # fit the models
    fits <- lapply(1:n_data_sets, function(x) estimate(as.matrix(subset(Y, .imp == x)[,!(names(Y) %in% ".imp")]),
                                                       iter = iter,
                                                       impute = FALSE, ...))

  }

  # pool the posterior draws of the imputed data sets
  fit <- combine_imputed_fits(fits)

  fit
  }

# Pool the posterior draws of models fitted to imputed data sets. The
# pooled object keeps the stored burn-in draws of the first fit (if the fits
# store burn-in draws, see post_draw_idx()), followed by the post-burn-in
# draws of all fits, and has iter = number of pooled post-burn-in draws.
# Posterior summaries (pcor_mat and, for explore, the running summaries) are
# recomputed from the pooled draws.
combine_imputed_fits <- function(fits) {

  fit  <- fits[[1]]
  iter <- fit$iter
  keep <- post_draw_idx(fit)
  burn <- seq_len(min(keep) - 1)          # stored burn-in slices (may be empty)

  pool <- function(name) {
    arrs <- lapply(fits, function(f) f$post_samp[[name]][, , post_draw_idx(f), drop = FALSE])
    if (length(burn) > 0)
      arrs <- c(list(fit$post_samp[[name]][, , burn, drop = FALSE]), arrs)
    abind::abind(arrs, along = 3)
  }

  fit$post_samp$pcors    <- pool("pcors")
  fit$post_samp$fisher_z <- pool("fisher_z")
  if (!is.null(fit$post_samp$beta)) {
    fit$post_samp$beta <- pool("beta")
  }
  if (!is.null(fit$post_samp$thresh)) {
    # thresholds: draws in the first dimension
    arrs <- lapply(fits, function(f) f$post_samp$thresh[post_draw_idx(f), , , drop = FALSE])
    if (length(burn) > 0)
      arrs <- c(list(fit$post_samp$thresh[burn, , , drop = FALSE]), arrs)
    fit$post_samp$thresh <- abind::abind(arrs, along = 1)
  }

  fit$iter <- iter * length(fits)
  idx      <- post_draw_idx(fit)

  pcor_mat <- apply(fit$post_samp$pcors[, , idx, drop = FALSE], 1:2, mean)
  fit$pcor_mat <- pcor_mat
  fit$post_samp$pcor_mat <- pcor_mat

  if (!is.null(fit$post_samp$z_mean)) {
    fit$post_samp$pcor_sd <- apply(fit$post_samp$pcors[, , idx, drop = FALSE], 1:2, sd)
    fit$post_samp$z_mean  <- apply(fit$post_samp$fisher_z[, , idx, drop = FALSE], 1:2, mean)
    fit$post_samp$z_sd    <- apply(fit$post_samp$fisher_z[, , idx, drop = FALSE], 1:2, sd)
  }

  fit
}
