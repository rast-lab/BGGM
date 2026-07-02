##' Perform Bayesian Graph Search and Optional Model Averaging
##'
##' The `ggm_search` function performs a Metropolis-Hastings graph search to
##' identify high-probability graph structures. At each iteration, one edge
##' is randomly proposed to flip (add or remove, chosen by an independent
##' coin flip), and the proposal is accepted or rejected using a proper MH
##' acceptance ratio (including a birth-death Hastings correction for the
##' proposal-size asymmetry between the add and remove moves), so the walk
##' can move to a worse-BIC graph and is not just a hill-climb. It also
##' computes an optional Bayesian Model Averaged (BMA) solution: the
##' distinct graphs visited after burn-in are reweighted by their
##' BIC-approximated posterior model probabilities.
##'
##' Set \code{probabilistic = FALSE} to instead run a deterministic greedy
##' hill-climb (accept a proposed edge flip only if it improves BIC). This
##' is much faster per iteration but explores far less of graph space in
##' practice — in testing it can get stuck after accepting only a handful
##' of moves, well before \code{iter} is reached, and \code{stop_early}
##' exists to end the search once that happens.
##'
##' This function is ideal for exploring the graph space and obtaining an initial
##' estimate of the graph structure or adjacency matrix.
##'
##' To refine the results or compute posterior distributions of graph parameters
##' (e.g., partial correlations), use the \code{\link{bma_posterior}} function,
##' which builds on the output of `ggm_search` to account for parameter uncertainty.
##'
##' @return A list containing the MAP graph structure, BMA solution (if specified),
##'         and BIC-approximated posterior probabilities of the graphs visited
##'         during the search.
##'
##' @seealso \code{\link{bma_posterior}}
##'
##' @param x Data, either raw data or covariance matrix
##' @param n For x = covariance matrix, provide number of observations
##' @param method mc3 defaults to MH sampling
##' @param prior_prob Prior prbability of sparseness.
##' @param iter Number of iterations
##' @param burn_in Number of initial iterations discarded before computing
##'   the BMA solution. Defaults to \code{iter / 2}. Only meaningful when
##'   \code{probabilistic = TRUE} (a greedy hill-climb has no transient to
##'   burn in — its BIC trajectory is monotone non-increasing).
##' @param stop_early Default to 1000. Only used when \code{probabilistic =
##'   FALSE}: stop the greedy search early if proposals keep being rejected
##'   (stopping by default after 1000 consecutive rejections). Ignored under
##'   probabilistic search, which always runs the full \code{iter} iterations
##'   since rejections are an expected, normal part of MH sampling.
##' @param bma_mean Compute Bayesian Model Averaged solution
##' @param seed Set seed. Current default is to set R's random seed.
##' @param progress Show progress bar, defaults to TRUE
##' @param probabilistic Defaults to TRUE: a genuine Metropolis-Hastings
##'   sampler over graph structures (see Details). Set to FALSE for the
##'   deterministic greedy hill-climb instead.
##' @param ... Not currently in use
##' @author Donny Williams and Philippe Rast
ggm_search <- function(x, n = NULL,
                       method = "mc3",
                       prior_prob = 0.3,
                       iter = 5000,
                       burn_in = NULL,
                       stop_early = 1000,
                       bma_mean = TRUE,
                       seed = NULL,
                       progress = TRUE,
                       probabilistic = TRUE, ...){

  set.seed(seed)
  ## Random seed unless user provided
  if(!is.null(seed) ) {
    set.seed(seed)
  }

  if (base::isSymmetric(as.matrix(x))) {
    S <- x
  } else {
    S <- cor(x)
    n <- nrow(x)
  }

  p <- ncol(S)

  if(method == "mc3"){

    if(is.null(stop_early)){

      stop_early <- iter

    }
    
    pcors <- -cov2cor( solve(S) ) + diag(2, p)

    # test full vs missing one edge
    BF_01 <- exp(-0.5 *  (tstat(r = pcors, n = n, p - 2)^2 - log(n)))
    BF_10 <- 1/BF_01
 
    ## Create starting adjacency matrix, based on the BF_10
    adj_start <- ifelse(BF_10 > 1, 1, 0)


    old <- hft_algorithm(Sigma = S,
                         adj = adj_start,
                         tol = 1e-10,
                         max_iter = 100)

    bic_old <- bic_fast(Theta = old$Theta,
                        S = S,
                        n = n,
                        prior_prob = prior_prob)

    if(isTRUE(progress)){
      message(paste0("BGGM: Sampling Graphs"))
    }

    fit <- .Call('_BGGM_search',
                 PACKAGE = 'BGGM',
                  S = S,
                  iter = iter,
                  old_bic = bic_old,
                  start_adj = adj_start,
                  n = n,
                  gamma = prior_prob,
                  stop_early = stop_early,
                  progress = progress,
                  probabilistic = probabilistic)

    if(isTRUE(progress)){
      message("BGGM: Finished")
    }

    # accepted
    acc <- fit$acc

    # first matrix (starting values)
    fit$adj[,,1] <- adj_start

    ## Add a burnin unless defined by user
    if(is.null(burn_in)) {
      burn_in <- round(iter/2)
    }

    # approximate marginal likelihood
    approx_marg_ll <- fit$bics

    # starting bic
    approx_marg_ll[1] <- bic_old

    if(!is.null(stop_early)){
      approx_marg_ll <- approx_marg_ll[which(approx_marg_ll != 0)]
      fit$adj <- fit$adj[,, which(approx_marg_ll != 0)]
    }

    adj_path <- fit$adj

    # Find position of smallest bic
    selected <-  which.min(approx_marg_ll)

    ## acc are accepted proposals
    if(acc == 0){
      adj <- fit$adj[,,1]
    } else {
      adj <-  fit$adj[,,selected]
    }

    ## BFs vs Most Probably Model (mpm)
    ## Comute delta of all models compared to best model
    delta <- approx_marg_ll - min(approx_marg_ll)

    ## Convert differences in marginal log-likelihoods into posterior probabilities for each model
    probs <- exp(-0.5 * delta) / sum( exp(-0.5 * delta) )

    ## MPM:
    Theta_map <- hft_algorithm(
      Sigma = S,
      adj = adj,
      tol = 1e-10,
      max_iter = 1000
    )

    ## Partial Correlation for the MPM model
    pcor_adj <- -cov2cor(Theta_map$Theta) + diag(2, p)

    ## Discard the pre-burn-in transient. Only meaningful for the
    ## probabilistic sampler -- a greedy hill-climb's BIC trajectory is
    ## monotone non-increasing and typically gets stuck after very few
    ## accepted moves, so discarding its early samples would throw away the
    ## only useful ones. MAP selection above intentionally used the full
    ## trajectory (a good graph found early shouldn't be discarded); this
    ## trims what's used for BMA below *and* what's returned/stored, so
    ## bma_posterior() (which re-derives graph counts from the returned
    ## approx_marg_ll/adj_path) stays consistent with probs computed here.
    if (probabilistic && burn_in > 0 && burn_in < length(approx_marg_ll)) {
      bma_range <- (burn_in + 1):length(approx_marg_ll)
      approx_marg_ll <- approx_marg_ll[bma_range]
      adj_path <- adj_path[,, bma_range]
    }

  }

  if(bma_mean & acc > 0){

    graph_ids <-  which(duplicated(approx_marg_ll) == 0)[-1]

    delta <- (approx_marg_ll[graph_ids] - min(approx_marg_ll[graph_ids])) * (6 / (2*sqrt(2*n)))

    probs <- exp(- 0.5 * delta) / sum(exp(- 0.5 * delta))

    graphs <- adj_path[,,graph_ids]

    Theta_bma <- lapply(1:length(probs), function(x){

      hft_algorithm(Sigma = S,
                    adj = graphs[,,x],
                    tol =1e-10,
                    max_iter = 1000)$Theta * probs[x]
      })

    Theta_bma <- Reduce("+", Theta_bma)
    pcor_bma <- -cov2cor(Theta_bma) + diag(2, p)

  } else {

    Theta_bma <- NULL
    pcor_bma <- NULL

  }

  returned_object <- list(pcor_adj = pcor_adj,
                          Theta_map = Theta_map,
                          Theta_bma = Theta_bma,
                          pcor_bma = pcor_bma,
                          adj = adj,
                          adj_start = adj_start,
                          probs = probs,
                          approx_marg_ll = approx_marg_ll,
                          selected = selected,
                          BF_start = BF_10,
                          adj_path = adj_path,
                          acc = acc,
                          S = S,
                          n = n)

  #rm(.Random.seed, envir=.GlobalEnv)

  class(returned_object) <- c("BGGM",
                              "ggm_search")

  return( returned_object )
}


print_ggm_search <- function(x, ...){

  cat("BGGM: Bayesian Gaussian Graphical Models \n")
  cat("--- \n")

  if(x$acc == 0){
    mat <- x$pcor_adj
    p <- ncol(mat)

    if(is.null( colnames(x$S))){
      colnames(mat) <- 1:p
      row.names(mat) <- 1:p

    } else {
      colnames(mat) <- colnames(x$S)
      row.names(mat) <- colnames(x$S)
    }

    cat("Most Probable Graph:\n\n")
    print(round(mat, 3))

  } else {
    mat <- x$pcor_bma
    p <- ncol(mat)

    if(is.null( colnames(x$S))){
      colnames(mat) <- 1:p
      row.names(mat) <- 1:p

    } else {
      colnames(mat) <- colnames(x$S)
      row.names(mat) <- colnames(x$S)
    }
    cat("Bayesian Model Averaged Graph:\n\n")
    print(round(mat, 3))

  }
}



