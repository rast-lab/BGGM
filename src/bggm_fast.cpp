// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include <progress.hpp>
#include <progress_bar.hpp>
#include <truncnorm.h>
#include <RcppArmadilloExtensions/sample.h>
#include <cmath>
#include <algorithm>

// [[Rcpp::depends(RcppArmadillo, RcppDist, RcppProgress)]]

// mean of 3d array
// [[Rcpp::export]]
arma::mat mean_array(arma::cube x){
  return mean(x, 2);
}

// --- Matrix utilities -----------------------------------------------------
// These helpers keep the Wishart scale matrices numerically stable while
// retaining as much of the original structure as possible.

arma::mat symmetrize_matrix(const arma::mat& x) {
  arma::mat sym_x = 0.5 * (x + x.t());
  sym_x.for_each([](arma::mat::elem_type& val) {
    if (!std::isfinite(val)) {
      val = 0.0;
    }
  });
  return sym_x;
}

arma::mat stabilize_pd(const arma::mat& x, double jitter = 1e-8) {
  arma::mat sym_x = symmetrize_matrix(x);

  arma::vec eigval;
  arma::mat eigvec;
  if (!arma::eig_sym(eigval, eigvec, sym_x)) {
    return jitter * arma::eye(sym_x.n_rows, sym_x.n_cols);
  }

  arma::vec clamped = eigval;
  for (arma::uword i = 0; i < clamped.n_elem; ++i) {
    if (!std::isfinite(clamped(i)) || clamped(i) < jitter) {
      clamped(i) = jitter;
    }
  }

  arma::mat rebuilt = eigvec * arma::diagmat(clamped) * eigvec.t();
  return arma::symmatu(rebuilt);
}

arma::mat safe_inv_sympd(const arma::mat& x, double jitter = 1e-8) {
  arma::mat stabilized = stabilize_pd(x, jitter);
  arma::mat inv_mat;
  double current_jitter = jitter;

  for (int attempt = 0; attempt < 5; ++attempt) {
    if (!arma::inv_sympd(inv_mat, stabilized)) {
      inv_mat = arma::inv(stabilized);
    }

    inv_mat = arma::symmatu(inv_mat);

    arma::mat chol;
    if (arma::chol(chol, inv_mat)) {
      return inv_mat;
    }

    current_jitter *= 10.0;
    stabilized = stabilize_pd(x, current_jitter);
  }

  // Fall back to a diagonal matrix if repeated attempts fail.
  return current_jitter * arma::eye(x.n_rows, x.n_cols);
}

arma::vec safe_solve_sympd_vec(const arma::mat& A,
                               const arma::vec& b,
                               double jitter = 1e-8) {
  arma::vec x;
  if (arma::solve(x, A, b, arma::solve_opts::likely_sympd)) {
    return x;
  }

  arma::mat stabilized = stabilize_pd(A, jitter);
  if (arma::solve(x, stabilized, b, arma::solve_opts::likely_sympd)) {
    return x;
  }

  arma::mat inv_A = safe_inv_sympd(stabilized, jitter);
  return inv_A * b;
}

arma::mat safe_solve_sympd_mat(const arma::mat& A,
                               const arma::mat& B,
                               double jitter = 1e-8) {
  arma::mat X;
  if (arma::solve(X, A, B, arma::solve_opts::likely_sympd)) {
    return X;
  }

  arma::mat stabilized = stabilize_pd(A, jitter);
  if (arma::solve(X, stabilized, B, arma::solve_opts::likely_sympd)) {
    return X;
  }

  arma::mat inv_A = safe_inv_sympd(stabilized, jitter);
  return inv_A * B;
}

double clamp01(double x) {
  if (x < 0.0) {
    return 0.0;
  }
  if (x > 1.0) {
    return 1.0;
  }
  return x;
}

double sample_truncated_normal(double mean,
                               double sd,
                               double lower,
                               double upper) {
  const double tol = 1e-12;
  double safe_sd = std::max(sd, 1e-8);

  double p_lower = clamp01(R::pnorm(lower, mean, safe_sd, TRUE, FALSE));
  double p_upper = clamp01(R::pnorm(upper, mean, safe_sd, TRUE, FALSE));

  double width = p_upper - p_lower;
  if (width < tol) {
    if (std::isfinite(lower) && std::isfinite(upper)) {
      return 0.5 * (lower + upper);
    }
    if (std::isfinite(lower) && !std::isfinite(upper)) {
      return lower;
    }
    if (!std::isfinite(lower) && std::isfinite(upper)) {
      return upper;
    }
    return mean;
  }

  double u = R::runif(p_lower, p_upper);
  return R::qnorm(u, mean, safe_sd, TRUE, FALSE);
}

// R quantile type = 1
// Utility for computing empirical cutpoints that match R's type = 1 quantile
// definition. Used when initializing thresholds from observed ordinal data.
// [[Rcpp::export]]
double quantile_type_1(arma::vec x, double prob){

  arma::mat sort_x = sort(x.elem(find_finite(x)));

  int n = sort_x.n_rows;

  float nppm =  n * prob;

  float j = floor(nppm);

  //float h = 0;

  float qs = 0;

  // if(nppm > j){
  //
  //   float h = 1;
  //
  // } else {
  //
  //   float h = 0;
  //
  // }

  arma::mat x_1(2, 1);
  arma::mat x_n(2, 1);

  x_1.col(0).row(0) = sort_x(0);
  x_1.col(0).row(1) = sort_x(0);

  x_n.col(0).row(0) = sort_x(n - 1);
  x_n.col(0).row(1) = sort_x(n - 1);

  arma::mat join_x = join_vert(x_1, sort_x, x_n);

  if(nppm > j){

    qs = join_x(j + 3);

  } else {

    qs = join_x(j + 2);

  }

  return qs;


// --- Latent ordinal helpers -------------------------------------------------

double truncated_normal_sample(double lower,
                               double upper,
                               double mean,
                               double sd) {
  if(!(std::isfinite(sd)) || sd <= 0.0) {
    sd = 1e-8;
  }

  double cdf_lower = R::pnorm(lower, mean, sd, TRUE, FALSE);
  double cdf_upper = R::pnorm(upper, mean, sd, TRUE, FALSE);

  if(!std::isfinite(cdf_lower)) {
    cdf_lower = (lower < mean) ? 0.0 : 1.0;
  }
  if(!std::isfinite(cdf_upper)) {
    cdf_upper = (upper < mean) ? 0.0 : 1.0;
  }

  cdf_lower = std::min(std::max(cdf_lower, 0.0), 1.0);
  cdf_upper = std::min(std::max(cdf_upper, 0.0), 1.0);

  if(cdf_upper <= cdf_lower + 1e-12) {
    if(std::isfinite(lower) && std::isfinite(upper)) {
      return 0.5 * (lower + upper);
    }
    if(!std::isfinite(lower) && std::isfinite(upper)) {
      return upper - std::abs(sd);
    }
    if(std::isfinite(lower) && !std::isfinite(upper)) {
      return lower + std::abs(sd);
    }
    return mean;
  }

  double u = R::runif(cdf_lower, cdf_upper);
  double draw = R::qnorm(u, mean, sd, TRUE, FALSE);

  if(!std::isfinite(draw)) {
    if(std::isfinite(lower) && std::isfinite(upper)) {
      draw = 0.5 * (lower + upper);
    } else if(!std::isfinite(lower) && std::isfinite(upper)) {
      draw = upper - std::abs(sd);
    } else if(std::isfinite(lower) && !std::isfinite(upper)) {
      draw = lower + std::abs(sd);
    } else {
      draw = mean;
    }
  }

  return draw;
}

arma::rowvec initialize_thresholds(const arma::vec& y_col, int K) {
  arma::rowvec thresholds(K + 1, arma::fill::zeros);
  thresholds(0) = -arma::datum::inf;
  thresholds(K) = arma::datum::inf;

  arma::uvec finite_idx = arma::find_finite(y_col);
  arma::vec finite_y = y_col.elem(finite_idx);

  for(int level = 1; level < K; ++level) {
    arma::uvec idx_leq = arma::find(finite_y <= level);
    double prob = 0.0;
    if(finite_y.n_elem > 0) {
      prob = static_cast<double>(idx_leq.n_elem) /
        static_cast<double>(finite_y.n_elem);
    }
    double cut = R::qnorm(prob, 0.0, 1.0, TRUE, FALSE);
    if(!std::isfinite(cut)) {
      cut = static_cast<double>(level) - 0.5;
    }
    thresholds(level) = cut;
  }

  return thresholds;
}

void conditional_gaussian(const arma::mat& cov,
                          const arma::mat& latent,
                          const arma::mat& mean,
                          int index,
                          arma::vec& cond_mean,
                          double& cond_var) {
  const int k = cov.n_cols;
  const int n = latent.n_rows;

  cond_mean.set_size(n);

  if(k == 1) {
    cond_mean = mean.col(index);
    cond_var = cov(index, index);
    if(!std::isfinite(cond_var) || cond_var <= 0.0) {
      cond_var = 1.0;
    }
    return;
  }

  arma::uvec others = arma::regspace<arma::uvec>(0, k - 1);
  others.shed_row(index);

  arma::mat cov_oo = cov.submat(others, others);
  arma::vec cov_io = cov.submat(index, others).t();

  arma::vec beta;
  bool solved = arma::solve(beta, cov_oo, cov_io, arma::solve_opts::fast);
  if(!solved) {
    arma::mat cov_inv = safe_inv_sympd(cov_oo);
    beta = cov_inv * cov_io;
  }

  cond_var = cov(index, index) - arma::dot(cov_io, beta);
  if(!std::isfinite(cond_var) || cond_var <= 0.0) {
    cond_var = std::max(1e-8, cov(index, index) - arma::dot(cov_io, beta));
    if(!std::isfinite(cond_var) || cond_var <= 0.0) {
      cond_var = 1e-8;
    }
  }

  arma::mat residual = latent.cols(others) - mean.cols(others);
  cond_mean = mean.col(index) + residual * beta;
}

double safe_uniform_between(double lower, double upper) {
  if(!std::isfinite(lower) && !std::isfinite(upper)) {
    return 0.0;
  }
  if(!std::isfinite(lower)) {
    if(!std::isfinite(upper)) {
      return 0.0;
    }
    return upper - 1.0;
  }
  if(!std::isfinite(upper)) {
    return lower + 1.0;
  }
  if(upper <= lower + 1e-12) {
    return lower;
  }
  return R::runif(lower, upper);
}

void update_thresholds_uniform(const arma::vec& y_col,
                               const arma::vec& latent_col,
                               arma::rowvec& thresholds_row,
                               int K) {
  for(int level = 1; level < K; ++level) {
    double lower = thresholds_row(level - 1);
    double upper = thresholds_row(level + 1);

    arma::uvec idx_current = arma::find(y_col == level);
    if(idx_current.n_elem > 0) {
      double max_current = latent_col.elem(idx_current).max();
      if(max_current > lower) {
        lower = max_current;
      }
    }

    arma::uvec idx_next = arma::find(y_col == level + 1);
    if(idx_next.n_elem > 0) {
      double min_next = latent_col.elem(idx_next).min();
      if(min_next < upper) {
        upper = min_next;
      }
    }

    if(lower >= upper) {
      thresholds_row(level) = lower;
    } else {
      thresholds_row(level) = safe_uniform_between(lower, upper);
    }
  }
}

void center_latent_column(arma::vec& latent_col,
                          arma::rowvec& thresholds_row,
                          int K) {
  double shift = arma::mean(latent_col);
  latent_col -= shift;

  if(std::abs(shift) < 1e-12) {
    return;
  }

  for(int level = 1; level < K; ++level) {
    double value = thresholds_row(level);
    if(std::isfinite(value)) {
      thresholds_row(level) = value - shift;
    }
  }
}


// [[Rcpp::export]]
arma::mat Sigma_i_not_i(arma::mat x, int index) {
  arma::mat sub_x = x.row(index);
  sub_x.shed_col(index);
  return(sub_x);
}

// [[Rcpp::export]]
arma::vec select_col(arma::mat x, int index){
  arma::vec z = x.col(index);
  return(z);
}

// [[Rcpp::export]]
arma::mat select_row(arma::mat x, int index){
  arma::mat z = x.row(index);
  return(z);
}

// [[Rcpp::export]]
arma::mat remove_row(arma::mat x, int which){
  x.shed_row(which);
  return(x);
}

// [[Rcpp::export]]
arma::mat remove_col(arma::mat x, int index){
  x.shed_col(index);
  return(x);
}


// Hoff, P. D. (2009). A first course in Bayesian statistical
// methods (Vol. 580). New York: Springer.
// pp 105-123

// note: `internal` is a simplified version of missing_gaussian
// that seemed to be faster when used within the Gibbs sampler

// [[Rcpp::export]]
Rcpp::List internal_missing_gaussian(arma::mat Y,
                            arma::mat Y_missing,
                            arma::mat Sigma,
                            int iter_missing) {
  int p = Y.n_cols;
  int n = Y.n_rows;

  arma::uvec index = find(Y_missing == 1);

  int n_na = index.n_elem;

  arma::mat ppc_missing(iter_missing, n_na, arma::fill::zeros);

  for(int s = 0; s < iter_missing; ++s){

    for(int j = 0; j < p; ++j){

      arma::vec Y_j = Y_missing.col(j);

      double check_na = sum(Y_j);

      if(check_na == 0){
        continue;
      }

      arma::uvec  index_j = find(Y_missing.col(j) == 1);
      int  n_missing = index_j.n_elem;

      arma::mat beta_j = Sigma_i_not_i(Sigma, j) * inv(remove_row(remove_col(Sigma, j), j));
      arma::mat  sd_j = sqrt(select_row(Sigma, j).col(j) - Sigma_i_not_i(Sigma, j) *
      inv(remove_row(remove_col(Sigma, j), j)) * Sigma_i_not_i(Sigma, j).t());
      arma::vec pred = remove_col(Y,j) * beta_j.t();
      arma::vec pred_miss = pred(index_j);

      for(int i = 0; i < n_missing; ++i){
        arma::vec ppc_i = Rcpp::rnorm(1,  pred(index_j[i]), arma::as_scalar(sd_j));
        Y.col(j).row(index_j[i]) = arma::as_scalar(ppc_i);
      }
    }

    arma::mat S_Y = Y.t() * Y;
    arma::mat Theta = wishrnd(safe_inv_sympd(S_Y),   (n - 1));
    Sigma = inv(Theta);
    ppc_missing.row(s) = Y.elem(index).t();
  }
  Rcpp::List ret;
  ret["Y"] = Y;
  ret["ppc_missing"] = ppc_missing;
  return  ret;
}



// Hoff, P. D. (2009). A first course in Bayesian statistical
// methods (Vol. 580). New York: Springer.
// pp 105-123

// [[Rcpp::export]]
Rcpp::List missing_gaussian(arma::mat Y,
                            arma::mat Y_missing,
                            arma::mat Sigma,
                            int iter_missing,
                            bool progress_impute,
                            bool store_all,
                            float lambda) {


  // progress
  Progress  pr(iter_missing, progress_impute);

  int p = Y.n_cols;
  int n = Y.n_rows;

  arma::uvec index = find(Y_missing == 1);

  int n_na = index.n_elem;

  arma::mat I_p(p, p, arma::fill::eye);

  // store posterior predictive distribution for missing values
  arma::mat ppd_missing(iter_missing, n_na, arma::fill::zeros);

  // store all imputed data sets
  arma::cube Y_all(n, p, iter_missing, arma::fill::zeros);

  for(int s = 0; s < iter_missing; ++s){

    pr.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }

    for(int j = 0; j < p; ++j){

      arma::vec Y_j = Y_missing.col(j);

      double check_na = sum(Y_j);

      if(check_na == 0){
        continue;
      }

      arma::uvec  index_j = find(Y_missing.col(j) == 1);

      int n_missing = index_j.n_elem;

      arma::mat beta_j = Sigma_i_not_i(Sigma, j) * inv(remove_row(remove_col(Sigma, j), j));

      arma::mat  sd_j = sqrt(select_row(Sigma, j).col(j) - Sigma_i_not_i(Sigma, j) *
        inv(remove_row(remove_col(Sigma, j), j)) * Sigma_i_not_i(Sigma, j).t());

      arma::vec pred = remove_col(Y,j) * beta_j.t();

      arma::vec pred_miss = pred(index_j);

      for(int i = 0; i < n_missing; ++i){

        arma::vec ppd_i = Rcpp::rnorm(1,  pred(index_j[i]),
                                      arma::as_scalar(sd_j));

        Y.col(j).row(index_j[i]) = arma::as_scalar(ppd_i);

      }

    }

    arma::mat S_Y = Y.t() * Y;
    arma::mat Theta = wishrnd(safe_inv_sympd(S_Y + I_p * lambda), n + lambda);
    Sigma = inv(Theta);

    if(store_all){
      Y_all.slice(s) = Y;
    }
  }

  Rcpp::List ret;
  ret["Y_all"] = Y_all;
  return ret;
}




// matrix F continous sampler
// Williams, D. R., & Mulder, J. (2019). Bayesian hypothesis testing for Gaussian
// graphical models:  Conditional independence and order constraints.

// [[Rcpp::export]]
Rcpp::List Theta_continuous(arma::mat Y,
                            int iter,
                            float delta,
                            float epsilon,
                            int prior_only,
                            int explore,
                            arma::mat start,
                            bool progress,
                            bool impute,
                            arma::mat Y_missing) {



  // note p changed to k to be consistent
  //with the multivariate regression samplers
  Progress  p(iter, progress);

  // number of rows
  float n = Y.n_rows;

  int k = 1;

  // sample prior
  if(prior_only == 1){

    if(explore == 1){

      k = 3;

      } else {

      k = Y.n_cols;
        }

      } else {

    // number of columns
    k = Y.n_cols;

        }

  // k by k identity mat
  arma::mat  I_k(k, k, arma::fill::eye);

  int nu = 1/ epsilon;

  // // #nu in Mulder & Pericchi (2018) formula (30) line 1.
  int nuMP = delta + k - 1 ;

  // #delta in Mulder & Pericchi (2018) formula (30) line 1.
  int deltaMP = nu - k + 1 ;

  // Psi update
  arma::cube Psi(k, k, 1, arma::fill::zeros);

  arma::mat B(epsilon * I_k);
  arma::mat BMP(inv(B));
  arma::mat BMPinv(inv(BMP));

  // precison matrix
  arma::cube Theta(k, k, 1, arma::fill::zeros);
  arma::cube Theta_mcmc(k, k, iter, arma::fill::zeros);

  // partial correlations
  arma::mat pcors(k,k);
  arma::cube pcors_mcmc(k, k, iter, arma::fill::zeros);

  arma::cube Sigma(k, k, 1, arma::fill::zeros);

  // starting value
  Psi.slice(0).fill(arma::fill::eye);

  Theta.slice(0) = start;

  arma::mat S_Y(Y.t() * Y);

  arma::uvec index = find(Y_missing == 1);

  int n_na = index.n_elem;

  arma::mat ppd_missing(iter, n_na, arma::fill::zeros);

  float iter_missing = 1;

  for(int  s = 0; s < iter; ++s){

    p.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }

    if(prior_only == 1){

      Psi.slice(0) = wishrnd(I_k * epsilon, nu);

      // sample Theta
      Sigma.slice(0) =   wishrnd(safe_inv_sympd(Psi.slice(0)),   k - 1 + delta);

      // Sigma
      Theta.slice(0) = inv(Sigma.slice(0));


    } else {

      Psi.slice(0) = wishrnd(safe_inv_sympd(BMPinv + Theta.slice(0)), nuMP + deltaMP + k - 1);

      // sample Theta
      Theta.slice(0) =   wishrnd(safe_inv_sympd(Psi.slice(0) + S_Y),  (deltaMP + k - 1) + (n - 1));

    }

    // partial correlations
    pcors = diagmat(1 / sqrt(Theta.slice(0).diag())) *
      Theta.slice(0) *
      diagmat(1 / sqrt(Theta.slice(0).diag()));

    // store posterior samples
    pcors_mcmc.slice(s) =  -(pcors - I_k);


    if(impute){

      Rcpp::List ppd_impute = internal_missing_gaussian(Y, Y_missing,
                                               inv(Theta.slice(0)),
                                               iter_missing);
      // imputed Y
      arma::mat Y = ppd_impute["Y"];

      // scatter matrix
      S_Y = Y.t() * Y;

      // store missing values
      ppd_missing.row(s) = Y.elem(index).t();
    }

  }

  arma::cube fisher_z = atanh(pcors_mcmc);

  arma::mat  pcor_mat = mean(pcors_mcmc.tail_slices(iter - 50), 2);

  arma::mat  ppd_mean = mean(ppd_missing, 0).t();

  Rcpp::List ret;
  ret["pcors"] = pcors_mcmc;
  ret["pcor_mat"] =  pcor_mat;
  ret["fisher_z"] = fisher_z;
  ret["ppd_mean"] = ppd_mean;
  return ret;
}


// [[Rcpp::export]]
Rcpp::List sample_prior(arma::mat Y,
                            int iter,
                            float delta,
                            float epsilon,
                            int prior_only,
                            int explore,
                            bool progress) {

  // note p changed to k to be consistent
  //with the multivariate regression samplers

  Progress  p(iter, progress);

  // number of rows
  float n = Y.n_rows;

  int k = 1;

  // sample prior
  if(prior_only == 1){

    if(explore == 1){

      k = 3;

    } else {

      k = Y.n_cols;

    }

  } else {

    // number of columns
    k = Y.n_cols;

  }

  // k by k identity mat
  arma::mat  I_k(k, k, arma::fill::eye);

  int nu = 1 / epsilon;
  // // #nu in Mulder & Pericchi (2018) formula (30) line 1.
  int nuMP = delta + k - 1 ;
  //
  // // #delta in Mulder & Pericchi (2018) formula (30) line 1.
  int deltaMP = nu - k + 1 ;

  // Psi update
  arma::cube Psi(k, k, 1, arma::fill::zeros);

  arma::mat B(epsilon * I_k);
  arma::mat BMP(inv(B));
  arma::mat BMPinv(inv(BMP));

  // precison matrix
  arma::cube Theta(k, k, 1, arma::fill::zeros);
  arma::cube Theta_mcmc(k, k, iter, arma::fill::zeros);

  // partial correlations
  arma::mat pcors(k,k);
  arma::cube pcors_mcmc(k, k, iter, arma::fill::zeros);

  // correlations
  arma::mat  cors(k,k);
  arma::cube cors_mcmc(k, k, iter, arma::fill::zeros);

  // covariance matrix
  arma::cube Sigma_mcmc(k, k, iter, arma::fill::zeros);
  arma::cube Sigma(k, k, 1, arma::fill::zeros);

  // starting value
  Sigma.slice(0).fill(arma::fill::eye);
  Psi.slice(0).fill(arma::fill::eye);
  Theta.slice(0).fill(arma::fill::eye);

  arma::mat S_Y(Y.t() * Y);

  for(int  s = 0; s < iter; ++s){

    p.increment();

    if(s % 250 == 0){

      Rcpp::checkUserInterrupt();

    }


    if(prior_only == 1){

      Psi.slice(0) = wishrnd(I_k * epsilon, nu);

      // sample Theta
      Sigma.slice(0) =   wishrnd(safe_inv_sympd(Psi.slice(0)),   k - 1 + delta);

      // Sigma
      Theta.slice(0) = inv(Sigma.slice(0));


    } else {

      Psi.slice(0) = wishrnd(safe_inv_sympd(BMPinv + Theta.slice(0)), nuMP + deltaMP + k - 1);

      // sample Theta
      Theta.slice(0) =   wishrnd(safe_inv_sympd(Psi.slice(0) + S_Y),  (deltaMP + k - 1) + (n - 1));

      // Sigma
      Sigma.slice(0) = inv(Theta.slice(0));

    }

    // partial correlations
    pcors = diagmat(1 / sqrt(Theta.slice(0).diag())) *
      Theta.slice(0) *
      diagmat(1 / sqrt(Theta.slice(0).diag()));

    // store posterior samples
    pcors_mcmc.slice(s) =  -(pcors - I_k);

    }

  arma::cube fisher_z = atanh(pcors_mcmc);

  Rcpp::List ret;
  ret["pcors"] = pcors_mcmc;
  ret["fisher_z"] = fisher_z;
  return ret;
}


// [[Rcpp::export]]
Rcpp::List mv_continuous(arma::mat Y,
                          arma::mat X,
                          float delta,
                          float epsilon,
                          int iter,
                          arma::mat start,
                          bool progress){


  // progress
  Progress  pr(iter, progress);

  // number of rows
  int n = Y.n_rows;

  // number of dependent variables
  int k = Y.n_cols;

  // number of predictors
  int p = X.n_cols;

  int nu = 1/ epsilon;

  // #nu in Mulder & Pericchi (2018) formula (30) line 1.
  int nuMP = delta + k - 1;

  // #delta in Mulder & Pericchi (2018) formula (30) line 1.
  int deltaMP = nu - k + 1 ;

  // k * k identity mat
  arma::mat  I_k(k, k, arma::fill::eye);

  // p * p identity mat
  arma::mat  I_p(p, p, arma::fill::eye);

  // scatter matrix X' * X
  arma::mat S_X(X.t() * X + I_p * 0.000001);

  // inv S_X
  arma::mat Sinv_X(inv(S_X));

  // Psi update
  arma::cube Psi(k, k, 1, arma::fill::zeros);

  // scatter matrix dependent variables
  arma::mat S_Y(k, k, arma::fill::zeros);
  arma::mat B(epsilon*I_k);
  arma::mat BMP(inv(B));
  arma::mat BMPinv(inv(BMP));

  // precison matrix
  arma::cube Theta(k, k, 1, arma::fill::zeros);
  arma::cube Theta_mcmc(k, k, iter, arma::fill::zeros);

  // partial correlations
  arma::mat pcors(k,k);
  arma::cube pcors_mcmc(k, k, iter, arma::fill::zeros);

  // correlations
  arma::mat  cors(k,k);
  arma::cube cors_mcmc(k, k, iter, arma::fill::zeros);

  // covariance matrix
  arma::cube Sigma(k, k, 1, arma::fill::zeros);
  arma::cube Sigma_mcmc(k, k, iter, arma::fill::zeros);

  // coefficients
  arma::mat beta(p, k, arma::fill::zeros);
  arma::cube beta_mcmc(p, k, iter,  arma::fill::zeros);

  // starting value
  Sigma.slice(0) = inv(start);
  Psi.slice(0).fill(arma::fill::eye);
  Theta.slice(0) = start;

  for(int s = 0; s < iter; ++s){

    pr.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }

    // draw coefficients
    beta = reshape(mvnrnd(reshape(Sinv_X * X.t() * Y, k*p , 1),
                          kron(Sigma.slice(0), Sinv_X)),
                          p, k);

    // scatter matrix
    S_Y = Y.t() * Y + I_k - beta.t() * S_X * beta;

    // sample Psi
    Psi.slice(0) = wishrnd(safe_inv_sympd(BMPinv + Theta.slice(0)), nuMP + deltaMP + k - 1);

    // sample Theta
    Theta.slice(0) =   wishrnd(safe_inv_sympd(Psi.slice(0) + S_Y),  (deltaMP + k - 1) + (n - 1));

    // Sigma
    Sigma.slice(0) = inv(Theta.slice(0));

    // correlation
    cors =  diagmat(1 / sqrt(Sigma.slice(0).diag())) *
      Sigma.slice(0) *
      diagmat(1 / sqrt(Sigma.slice(0).diag()));

    // partial correlations
    pcors = diagmat(1 / sqrt(Theta.slice(0).diag())) *
      Theta.slice(0) *
      diagmat(1 / sqrt(Theta.slice(0).diag()));


    beta_mcmc.slice(s) = beta;
    pcors_mcmc.slice(s) =  -(pcors - I_k);
  }

  arma::cube fisher_z = atanh(pcors_mcmc);

  arma::mat  pcor_mat = mean(pcors_mcmc.tail_slices(iter - 50), 2);

  Rcpp::List ret;
  ret["pcors"] = pcors_mcmc;
  ret["pcor_mat"] =  pcor_mat;
  ret["beta"] = beta_mcmc;
  ret["fisher_z"] = fisher_z;
  return ret;
}

// [[Rcpp::export]]
Rcpp::List trunc_mvn(arma::mat mu,
                     arma::mat rinv,
                     arma::mat z,
                     arma::mat y,
                     arma::rowvec cutpoints){

  // adapted from Aline Talhouk (matlab)
  // mu: beta hat (p x 1)
  // rinv: matrix-F precision matrix
  // z: latent data (p x 1)
  // y: observed binary data (for now)
  // cutpoint: thresholds for truncati

  // number of columns
  int T = rinv.n_cols;

  // number of cutpoints
  int n_cuts = cutpoints.n_cols;

  // substrat mean
  arma::mat zt(z - mu);

  // cutpoints (a: low, b: upper)
  arma::mat a(1, n_cuts - 1); a.fill(0);
  arma::mat b(1, n_cuts - 1); a.fill(0);

  for(int i = 0; i < (n_cuts - 1); ++i){

    a.col(i) = cutpoints(i);

    b.col(i) = cutpoints(i + 1);

  }

  // match cutpoints to T
  arma::mat a2(T, 1);
  arma::mat b2(T, 1);

  for(int i = 0; i < T; ++i){

    a2.row(i) = a.col(arma::as_scalar(y.row(i)));

    b2.row(i) = b.col(arma::as_scalar(y.row(i)));

  }

  // alpha mat
  arma::mat alpha(a2 - mu);
  // beta mat
  arma::mat beta(b2 - mu);

  // h2
  arma::mat h2(1 / rinv.diag());

  // temporary holders to make c
  arma::mat diag1(T, T);
  arma::mat diag2 = rinv.diag();

  for(int i = 0; i < T; ++i){

    diag1.row(i) = arma::repmat(diag2.row(i), 1, T);

  }

  // c
  arma::mat c(- rinv / diag1);
  for(int i = 0; i < T; ++i){
    c.row(i).col(i) = 0;
  }

  // term
  arma::mat  term(T, 1, arma::fill::zeros);

  // upper bound
  arma::mat  lb(T, 1, arma::fill::zeros);

  // lower bound
  arma::mat  ub(T, 1, arma::fill::zeros);

  // epsilon
  // arma:vec  eps(1, fill::zeros);

  for(int i = 0; i < T; ++i){

    arma::mat term(c * zt);

    arma::mat lb((alpha - repmat(term.row(i), T, 1)) / sqrt(h2(i)));

    arma::mat ub((beta - repmat(term.row(i), T, 1)) / sqrt(h2(i)));

    arma::vec  eps(rtruncnorm(1, 0,  1,  lb(i),  ub(i)));

    zt.row(i) = term.row(i)  + sqrt(h2(i)) *  eps;

  }

  arma::mat zr(zt + mu);

  // return
  Rcpp::List ret;
  ret["zr"] = zr;
  return  ret;
}

// binary sampler
// [[Rcpp::export]]
Rcpp::List mv_binary(arma::mat Y,
		     arma::mat X,
		     float delta,
		     float epsilon,
		     int iter,
		     float beta_prior,
		     arma::rowvec cutpoints,
		     arma::mat start,
		     bool progress){

  // Y: data matrix (n * k)
  // X: predictors (n * p) (blank for "network")
  // iter: number of samples
  // cutpoints: truncation points
  // delta: hyperparameter

  // progress
  Progress  pr(iter, progress);

  // dependent variables
  int k = Y.n_cols;

  // sample size
  int n = Y.n_rows;

  // number of predictors
  int p = X.n_cols;

  // arma::rowvec ct = cutpoints;

  // int epsilon1 = epsilon;

  int nu = 1 / epsilon;
  // #nu in Mulder & Pericchi (2018) formula (30) line 1.
  int nuMP = delta + k - 1 ;

  // #delta in Mulder & Pericchi (2018) formula (30) line 1.
  int deltaMP = nu - k + 1 ;

  // k * k identity mat
  arma::mat  I_k(k, k, arma::fill::eye);

  // p * p identity mat
  arma::mat  I_p(p, p, arma::fill::eye);

  // scatter matrix X' * X
  arma::mat S_X(X.t() * X + I_p * beta_prior);

  // inv S_X
  arma::mat Sinv_X(inv(S_X));

  // Psi update
  arma::cube Psi(k, k, 1, arma::fill::zeros);

  // scatter matrix dependent variables
  arma::mat S_Y(k, k, arma::fill::zeros);
  arma::mat B(epsilon * I_k);
  arma::mat BMP(inv(B));
  arma::mat BMPinv(inv(BMP));

  // precison matrix
  arma::cube Theta(k, k, 1, arma::fill::zeros);
  arma::cube Theta_mcmc(k, k, iter, arma::fill::zeros);

  // partial correlations
  arma::mat pcors(k,k);
  arma::cube pcors_mcmc(k, k, iter, arma::fill::zeros);

  // correlations
  arma::mat  cors(k,k);
  // arma::cube cors_mcmc(k, k, iter, arma::fill::zeros);

  // covariance matrix
  arma::cube Sigma(k, k, 1, arma::fill::zeros);
  // arma::cube Sigma_mcmc(k, k, iter, arma::fill::zeros);

  // latent data
  arma::cube z0(n, k, 1,  arma::fill::zeros);

  // expanded latent data
  arma::mat w(n, k, arma::fill::zeros);

  // conditonal data
  arma::cube Xbhat(n, k,  1, arma::fill::zeros);

  // Rinv update
  arma::cube Rinv(k, k, 1, arma::fill::zeros);

  arma::cube R(k, k, 1, arma::fill::zeros);

  // coefficients
  arma::mat M(p, k, arma::fill::zeros);

  // expanded coefs
  arma::mat beta(p, k, arma::fill::zeros);
  arma::cube beta_mcmc(p, k, iter,  arma::fill::zeros);

  // draw coefs conditional on w
  arma::mat gamma(p, k, arma::fill::zeros);

  arma::cube Dinv(k, k, 1, arma::fill::zeros);
  arma::mat D(k, k,  arma::fill::eye);

  // starting values
  Sigma.slice(0) = inv(start);
  Theta.slice(0) = start;
  Psi.slice(0).fill(arma::fill::eye);
  Dinv.slice(0).fill(arma::fill::eye);
  Rinv.slice(0).fill(arma::fill::eye);
  R.slice(0).fill(arma::fill::eye);

  arma::mat ss(1, 1);
  arma::mat mm(n,1);

  // start sampling
  for(int s = 0; s < iter; ++s){

    pr.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }


    for(int i = 0; i < k; ++i){

      mm = Xbhat.slice(0).col(i).t() +
        Sigma_i_not_i(R.slice(0), i) *
        inv(remove_row(remove_col(R.slice(0), i), i)) *
        (remove_col(z0.slice(0), i).t() - remove_col(Xbhat.slice(0), i).t());

      ss = select_row(R.slice(0), i).col(i) -
        Sigma_i_not_i(R.slice(0), i) *
        inv(remove_row(remove_col(R.slice(0), i), i)) *
        Sigma_i_not_i(R.slice(0), i).t();


      for(int j = 0 ; j < n; ++j){

        arma::vec temp = Y.col(i).row(j);

        if(temp(0) == 0){


          arma::vec temp_j = rtruncnorm(1,   mm(j),  sqrt(ss(0)),  -arma::datum::inf,  0);

          z0.slice(0).col(i).row(j) =   temp_j;


        } else {



          arma::vec temp_j = rtruncnorm(1,   mm(j),  sqrt(ss(0)),  0, arma::datum::inf);

          z0.slice(0).col(i).row(j) =   temp_j;

        }
      }
    }

    // D matrix
    for(int i = 0; i < k; ++i){

      D.row(i).col(i) = sqrt(1 / R::rgamma((delta + k - 1) / 2,
                   2 / arma::as_scalar(Rinv.slice(0).row(i).col(i))));
    }

    w = z0.slice(0) * D;
    
    M  = Sinv_X * X.t() * w;

    gamma = reshape(mvnrnd(reshape(M, k * p , 1),
                           kron(Sigma.slice(0), Sinv_X)),
                           p, k);

    beta  = gamma * Dinv.slice(0);

    Xbhat.slice(0) = X * beta;

    S_Y =   w.t() * w + I_k - M.t() * S_X * M;

    // sample Psi
    // Debugging:
    // Declare the bmpinv variable
    // arma::mat bmpinv;
    // bmpinv = inv(BMPinv + Theta.slice(0));
    // if (!bmpinv.is_sympd()) {
    //   // Print the bmpinv matrix for debugging
    //   bmpinv.print("bmpinv matrix:");
    //   Rcpp::stop("bmpinv matrix is not symmetric positive definite.");
    // }
    // END Debug

    Psi.slice(0) = wishrnd(safe_inv_sympd(BMPinv + Theta.slice(0)), nuMP + deltaMP + k - 1);

    // sample Theta
    // Debugging:
    // Declare the psisl variable
    // arma::mat psisl;
    // psisl = inv(Psi.slice(0) + S_Y);
    // if (!psisl.is_sympd()) {
    //   // Print the matrix for debugging
    //   Psi.slice(0).print("Psi.slice(0):" );
    //   S_Y.print("S_Y:");
    //   w.print("w:" );
    //   I_k.print("I_k:");
    //   M.print("M:");
    //   S_X.print("S_X:");
    //   psisl.print("psisl matrix:");
    //   Rcpp::stop("psisl matrix is not symmetric positive definite.");
    // }
    // END Debug

    Theta.slice(0) =   wishrnd(safe_inv_sympd(Psi.slice(0) + S_Y),  (deltaMP + k - 1) + (n - 1));

    // Sigma
    Sigma.slice(0) = inv(Theta.slice(0));

    // correlation
    cors =  diagmat(1 / sqrt(Sigma.slice(0).diag())) *
      Sigma.slice(0) *
      diagmat(1 / sqrt(Sigma.slice(0).diag()));

    // partial correlations
    pcors = diagmat(1 / sqrt(Theta.slice(0).diag())) *
      Theta.slice(0) *
      diagmat(1 / sqrt(Theta.slice(0).diag()));


    Dinv.slice(0)  = inv(diagmat(sqrt(Sigma.slice(0).diag())));

    Rinv.slice(0)   = inv(cors);

    R.slice(0) = cors;

    beta_mcmc.slice(s) =reshape(beta, p,k);
    pcors_mcmc.slice(s) =  -(pcors - I_k);

  }

  arma::cube fisher_z = atanh(pcors_mcmc);
  arma::mat  pcor_mat = mean(pcors_mcmc.tail_slices(iter - 50), 2);

  Rcpp::List ret;
  ret["pcors"] = pcors_mcmc;
  ret["pcor_mat"] = pcor_mat;
  ret["beta"] = beta_mcmc;
  ret["fisher_z"] = fisher_z;
  return  ret;
}



  // ordinal sampler
  // [[Rcpp::export]]
  Rcpp::List mv_ordinal_cowles(arma::mat Y,
                             arma::mat X,
                             float delta,
                             float epsilon,
                             int iter, float MH) {

    int n = Y.n_rows;
    int k = Y.n_cols;
    int p = X.n_cols;

    Progress pr(iter, true);

    int nu = 1 / epsilon;
    int nuMP = delta + k - 1;
    int deltaMP = nu - k + 1;

    arma::mat I_k = arma::eye(k, k);
    arma::mat I_p = arma::eye(p, p);

    arma::mat S_X = X.t() * X + I_p * 1e-6;
    arma::mat Sinv_X = safe_inv_sympd(S_X);

    arma::cube Psi(k, k, 1, arma::fill::zeros);
    arma::mat S_Y(k, k, arma::fill::zeros);
    arma::mat B = epsilon * I_k;
    arma::mat BMP = safe_inv_sympd(B);
    arma::mat BMPinv = safe_inv_sympd(BMP);

    arma::cube Theta(k, k, 1, arma::fill::zeros);
    arma::cube Theta_mcmc(k, k, iter, arma::fill::zeros);

    arma::mat pcors(k, k, arma::fill::zeros);
    arma::cube pcors_mcmc(k, k, iter, arma::fill::zeros);

    arma::mat cors(k, k, arma::fill::zeros);
    arma::cube cors_mcmc(k, k, iter, arma::fill::zeros);

    arma::cube Sigma(k, k, 1, arma::fill::zeros);
    arma::cube Sigma_mcmc(k, k, iter, arma::fill::zeros);

    arma::cube z0(n, k, 1, arma::fill::zeros);
    arma::mat& z_current = z0.slice(0);

    arma::mat w(n, k, arma::fill::zeros);
    arma::cube Xbhat(n, k, 1, arma::fill::zeros);
    arma::mat& Xbhat_current = Xbhat.slice(0);

    arma::cube Rinv(k, k, 1, arma::fill::zeros);
    arma::cube R(k, k, 1, arma::fill::zeros);

    arma::mat M(p, k, arma::fill::zeros);
    arma::mat beta(p, k, arma::fill::zeros);
    arma::cube beta_mcmc(p, k, iter, arma::fill::zeros);
    arma::mat gamma(p, k, arma::fill::zeros);

    arma::cube Dinv(k, k, 1, arma::fill::zeros);
    arma::mat D = arma::eye(k, k);

    arma::vec acc(k, arma::fill::zeros);

    Theta.slice(0) = I_k;
    Sigma.slice(0) = I_k;
    Psi.slice(0) = I_k;
    Rinv.slice(0) = I_k;
    R.slice(0) = I_k;
    Dinv.slice(0) = I_k;

    arma::ivec K_by_var(k, arma::fill::ones);
    int K_max = 1;
    for(int var = 0; var < k; ++var) {
      arma::vec col = Y.col(var);
      arma::uvec finite_idx = arma::find_finite(col);
      int K_var = 1;
      if(finite_idx.n_elem > 0) {
        K_var = static_cast<int>(arma::max(col.elem(finite_idx)));
      }
      if(K_var < 1) {
        K_var = 1;
      }
      K_by_var(var) = K_var;
      if(K_var > K_max) {
        K_max = K_var;
      }
    }

    arma::mat thresholds(k, K_max + 1, arma::fill::zeros);
    for(int var = 0; var < k; ++var) {
      int K_var = K_by_var(var);
      thresholds(var, 0) = -arma::datum::inf;
      for(int level = 1; level <= K_max; ++level) {
        thresholds(var, level) = arma::datum::inf;
      }
      if(K_var > 1) {
        arma::rowvec init = initialize_thresholds(Y.col(var), K_var);
        for(int level = 1; level < K_var; ++level) {
          thresholds(var, level) = init(level);
        }
        thresholds(var, K_var) = arma::datum::inf;
      } else {
        thresholds(var, 1) = arma::datum::inf;
      }
    }

    arma::cube thresh(iter, K_max + 1, k, arma::fill::zeros);

    arma::vec cond_mean(n);
    double cond_var = 0.0;

    for(int s = 0; s < iter; ++s) {
      pr.increment();

      if(s % 250 == 0) {
        Rcpp::checkUserInterrupt();
      }

      for(int i = 0; i < k; ++i) {
        conditional_gaussian(R.slice(0), z_current, Xbhat_current, i, cond_mean, cond_var);
        double sd = std::sqrt(std::max(cond_var, 1e-8));
        int K_var = K_by_var(i);

        for(int j = 0; j < n; ++j) {
          double y_val = Y(j, i);
          if(!std::isfinite(y_val)) {
            continue;
          }
          int cat = static_cast<int>(y_val);
          if(cat <= 0) {
            continue;
          }
          if(cat > K_var) {
            cat = K_var;
          }
          double lower = thresholds(i, cat - 1);
          double upper = (cat >= K_var) ? arma::datum::inf : thresholds(i, cat);
          z_current(j, i) = truncated_normal_sample(lower, upper, cond_mean(j), sd);
        }
      }

      for(int var = 0; var < k; ++var) {
        int K_var = K_by_var(var);
        if(K_var > 1) {
          double shift = arma::mean(z_current.col(var));
          z_current.col(var) -= shift;
          if(std::abs(shift) > 1e-12) {
            for(int level = 1; level < K_var; ++level) {
              double current = thresholds(var, level);
              if(std::isfinite(current)) {
                thresholds(var, level) = current - shift;
              }
            }
          }
          update_thresholds_uniform(Y.col(var), z_current.col(var), thresholds.row(var), K_var);
          acc(var) += 1.0;
        } else {
          z_current.col(var) -= arma::mean(z_current.col(var));
        }
      }

      for(int i = 0; i < k; ++i) {
        double diag_val = Rinv.slice(0)(i, i);
        if(!std::isfinite(diag_val) || diag_val <= 0.0) {
          diag_val = 1.0;
        }
        D(i, i) = std::sqrt(1.0 / R::rgamma((delta + k - 1.0) / 2.0, 2.0 / diag_val));
      }

      w = z_current * D;
      M = Sinv_X * X.t() * w;

      arma::vec mean_vec = arma::vectorise(M);
      arma::mat cov_mat = arma::kron(Sigma.slice(0), Sinv_X);
      gamma = arma::reshape(mvnrnd(mean_vec, cov_mat), p, k);
      beta = gamma * Dinv.slice(0);
      Xbhat_current = X * beta;

      S_Y = w.t() * w + I_k - M.t() * S_X * M;

      Psi.slice(0) = wishrnd(safe_inv_sympd(BMPinv + Theta.slice(0)), nuMP + deltaMP + k - 1);
      Theta.slice(0) = wishrnd(safe_inv_sympd(Psi.slice(0) + S_Y), (deltaMP + k - 1) + (n - 1));
      Sigma.slice(0) = safe_inv_sympd(Theta.slice(0));

      arma::vec sigma_diag = Sigma.slice(0).diag();
      arma::vec theta_diag = Theta.slice(0).diag();

      arma::vec inv_sqrt_sigma = 1.0 / arma::sqrt(sigma_diag);
      arma::vec inv_sqrt_theta = 1.0 / arma::sqrt(theta_diag);

      cors = diagmat(inv_sqrt_sigma) * Sigma.slice(0) * diagmat(inv_sqrt_sigma);
      pcors = diagmat(inv_sqrt_theta) * Theta.slice(0) * diagmat(inv_sqrt_theta);

      arma::mat Dinv_mat = diagmat(arma::sqrt(sigma_diag));
      Dinv.slice(0) = arma::inv(Dinv_mat);

      R.slice(0) = cors;
      Rinv.slice(0) = safe_inv_sympd(cors);

      beta_mcmc.slice(s) = beta;
      pcors_mcmc.slice(s) = -(pcors - I_k);
      cors_mcmc.slice(s) = cors;
      Sigma_mcmc.slice(s) = Sigma.slice(0);
      Theta_mcmc.slice(s) = Theta.slice(0);

      for(int var = 0; var < k; ++var) {
        int K_var = K_by_var(var);
        for(int level = 0; level <= K_max; ++level) {
          if(level == 0) {
            thresh(s, level, var) = -arma::datum::inf;
          } else if(level > K_var) {
            thresh(s, level, var) = arma::datum::inf;
          } else {
            thresh(s, level, var) = thresholds(var, level);
          }
        }
      }
    }

    arma::cube fisher_z = atanh(pcors_mcmc);

    Rcpp::List ret;
    ret["pcors"] = pcors_mcmc;
    ret["cors"] = cors_mcmc;
    ret["beta"] = beta_mcmc;
    ret["Theta"] = Theta_mcmc;
    ret["Sigma"] = Sigma_mcmc;
    ret["acc"] = acc;
    ret["thresh"] = thresh;
    ret["fisher_z"] = fisher_z;
    return ret;
  }

  // ordinal sampler
  // [[Rcpp::export]]
  Rcpp::List mv_ordinal_cowles(arma::mat Y,
                             arma::mat X,
                             float delta,
                             float epsilon,
                             int iter, float MH) {

    int n = Y.n_rows;
    int k = Y.n_cols;
    int p = X.n_cols;

    Progress pr(iter, true);

    int nu = 1 / epsilon;
    int nuMP = delta + k - 1;
    int deltaMP = nu - k + 1;

    arma::mat I_k = arma::eye(k, k);
    arma::mat I_p = arma::eye(p, p);

    arma::mat S_X = X.t() * X + I_p * 1e-6;
    arma::mat Sinv_X = safe_inv_sympd(S_X);

    arma::cube Psi(k, k, 1, arma::fill::zeros);
    arma::mat S_Y(k, k, arma::fill::zeros);
    arma::mat B = epsilon * I_k;
    arma::mat BMP = safe_inv_sympd(B);
    arma::mat BMPinv = safe_inv_sympd(BMP);

    arma::cube Theta(k, k, 1, arma::fill::zeros);
    arma::cube Theta_mcmc(k, k, iter, arma::fill::zeros);

    arma::mat pcors(k, k, arma::fill::zeros);
    arma::cube pcors_mcmc(k, k, iter, arma::fill::zeros);

    arma::mat cors(k, k, arma::fill::zeros);
    arma::cube cors_mcmc(k, k, iter, arma::fill::zeros);

    arma::cube Sigma(k, k, 1, arma::fill::zeros);
    arma::cube Sigma_mcmc(k, k, iter, arma::fill::zeros);

    arma::cube z0(n, k, 1, arma::fill::zeros);
    arma::mat& z_current = z0.slice(0);

    arma::mat w(n, k, arma::fill::zeros);
    arma::cube Xbhat(n, k, 1, arma::fill::zeros);
    arma::mat& Xbhat_current = Xbhat.slice(0);

    arma::cube Rinv(k, k, 1, arma::fill::zeros);
    arma::cube R(k, k, 1, arma::fill::zeros);

    arma::mat M(p, k, arma::fill::zeros);
    arma::mat beta(p, k, arma::fill::zeros);
    arma::cube beta_mcmc(p, k, iter, arma::fill::zeros);
    arma::mat gamma(p, k, arma::fill::zeros);

    arma::cube Dinv(k, k, 1, arma::fill::zeros);
    arma::mat D = arma::eye(k, k);

    arma::vec acc(k, arma::fill::zeros);

    Theta.slice(0) = I_k;
    Sigma.slice(0) = I_k;
    Psi.slice(0) = I_k;
    Rinv.slice(0) = I_k;
    R.slice(0) = I_k;
    Dinv.slice(0) = I_k;

    arma::ivec K_by_var(k, arma::fill::ones);
    int K_max = 1;
    for(int var = 0; var < k; ++var) {
      arma::vec col = Y.col(var);
      arma::uvec finite_idx = arma::find_finite(col);
      int K_var = 1;
      if(finite_idx.n_elem > 0) {
        K_var = static_cast<int>(arma::max(col.elem(finite_idx)));
      }
      if(K_var < 1) {
        K_var = 1;
      }
      K_by_var(var) = K_var;
      if(K_var > K_max) {
        K_max = K_var;
      }
    }

    arma::mat thresholds(k, K_max + 1, arma::fill::zeros);
    for(int var = 0; var < k; ++var) {
      int K_var = K_by_var(var);
      thresholds(var, 0) = -arma::datum::inf;
      for(int level = 1; level <= K_max; ++level) {
        thresholds(var, level) = arma::datum::inf;
      }
      if(K_var > 1) {
        arma::rowvec init = initialize_thresholds(Y.col(var), K_var);
        for(int level = 1; level < K_var; ++level) {
          thresholds(var, level) = init(level);
        }
        thresholds(var, K_var) = arma::datum::inf;
      } else {
        thresholds(var, 1) = arma::datum::inf;
      }
    }

    arma::cube thresh(iter, K_max + 1, k, arma::fill::zeros);

    arma::vec cond_mean(n);
    double cond_var = 0.0;

    for(int s = 0; s < iter; ++s) {
      pr.increment();

      if(s % 250 == 0) {
        Rcpp::checkUserInterrupt();
      }

      for(int i = 0; i < k; ++i) {
        conditional_gaussian(R.slice(0), z_current, Xbhat_current, i, cond_mean, cond_var);
        double sd = std::sqrt(std::max(cond_var, 1e-8));
        int K_var = K_by_var(i);

        for(int j = 0; j < n; ++j) {
          double y_val = Y(j, i);
          if(!std::isfinite(y_val)) {
            continue;
          }
          int cat = static_cast<int>(y_val);
          if(cat <= 0) {
            continue;
          }
          if(cat > K_var) {
            cat = K_var;
          }
          double lower = thresholds(i, cat - 1);
          double upper = (cat >= K_var) ? arma::datum::inf : thresholds(i, cat);
          z_current(j, i) = truncated_normal_sample(lower, upper, cond_mean(j), sd);
        }
      }

      for(int var = 0; var < k; ++var) {
        int K_var = K_by_var(var);
        if(K_var > 1) {
          double shift = arma::mean(z_current.col(var));
          z_current.col(var) -= shift;
          if(std::abs(shift) > 1e-12) {
            for(int level = 1; level < K_var; ++level) {
              double current = thresholds(var, level);
              if(std::isfinite(current)) {
                thresholds(var, level) = current - shift;
              }
            }
          }
          update_thresholds_uniform(Y.col(var), z_current.col(var), thresholds.row(var), K_var);
          acc(var) += 1.0;
        } else {
          z_current.col(var) -= arma::mean(z_current.col(var));
        }
      }

      for(int i = 0; i < k; ++i) {
        double diag_val = Rinv.slice(0)(i, i);
        if(!std::isfinite(diag_val) || diag_val <= 0.0) {
          diag_val = 1.0;
        }
        D(i, i) = std::sqrt(1.0 / R::rgamma((delta + k - 1.0) / 2.0, 2.0 / diag_val));
      }

      w = z_current * D;
      M = Sinv_X * X.t() * w;

      arma::vec mean_vec = arma::vectorise(M);
      arma::mat cov_mat = arma::kron(Sigma.slice(0), Sinv_X);
      gamma = arma::reshape(mvnrnd(mean_vec, cov_mat), p, k);
      beta = gamma * Dinv.slice(0);
      Xbhat_current = X * beta;

      S_Y = w.t() * w + I_k - M.t() * S_X * M;

      Psi.slice(0) = wishrnd(safe_inv_sympd(BMPinv + Theta.slice(0)), nuMP + deltaMP + k - 1);
      Theta.slice(0) = wishrnd(safe_inv_sympd(Psi.slice(0) + S_Y), (deltaMP + k - 1) + (n - 1));
      Sigma.slice(0) = safe_inv_sympd(Theta.slice(0));

      arma::vec sigma_diag = Sigma.slice(0).diag();
      arma::vec theta_diag = Theta.slice(0).diag();

      arma::vec inv_sqrt_sigma = 1.0 / arma::sqrt(sigma_diag);
      arma::vec inv_sqrt_theta = 1.0 / arma::sqrt(theta_diag);

      cors = diagmat(inv_sqrt_sigma) * Sigma.slice(0) * diagmat(inv_sqrt_sigma);
      pcors = diagmat(inv_sqrt_theta) * Theta.slice(0) * diagmat(inv_sqrt_theta);

      arma::mat Dinv_mat = diagmat(arma::sqrt(sigma_diag));
      Dinv.slice(0) = arma::inv(Dinv_mat);

      R.slice(0) = cors;
      Rinv.slice(0) = safe_inv_sympd(cors);

      beta_mcmc.slice(s) = beta;
      pcors_mcmc.slice(s) = -(pcors - I_k);
      cors_mcmc.slice(s) = cors;
      Sigma_mcmc.slice(s) = Sigma.slice(0);
      Theta_mcmc.slice(s) = Theta.slice(0);

      for(int var = 0; var < k; ++var) {
        int K_var = K_by_var(var);
        for(int level = 0; level <= K_max; ++level) {
          if(level == 0) {
            thresh(s, level, var) = -arma::datum::inf;
          } else if(level > K_var) {
            thresh(s, level, var) = arma::datum::inf;
          } else {
            thresh(s, level, var) = thresholds(var, level);
          }
        }
      }
    }

// ordinal sampler customary
// [[Rcpp::export]]
Rcpp::List mv_ordinal_albert(arma::mat Y,
                          arma::mat X,
                          int iter,
                          float delta,
                          float epsilon,
                          int K,
                          arma::mat start,
                          bool progress
                          ) {

  Progress pr(iter, progress);

  int n = Y.n_rows;
  int k = Y.n_cols;
  int p = X.n_cols;

  arma::mat I_k(k, k, arma::fill::eye);
  arma::mat I_p(p, p, arma::fill::eye);

  arma::mat S_X = X.t() * X + 0.001 * I_p;
  arma::mat Sinv_X = safe_solve_sympd_mat(S_X, I_p);

  int nu = 1 / epsilon;
  int nuMP = delta + k - 1;
  int deltaMP = nu - k + 1;

  arma::mat B = epsilon * I_k;
  arma::mat BMP = safe_inv_sympd(B);
  arma::mat BMPinv = safe_inv_sympd(BMP);

  arma::mat Theta_current = arma::symmatu(start);
  arma::mat Sigma_current = safe_inv_sympd(Theta_current);

  arma::vec sigma_diag = Sigma_current.diag();
  sigma_diag.transform([](double val) { return std::max(val, 1e-8); });
  arma::vec inv_sqrt_sigma = 1.0 / arma::sqrt(sigma_diag);
  arma::mat Dinv_current = arma::diagmat(inv_sqrt_sigma);

  arma::mat cors = arma::symmatu(Dinv_current * Sigma_current * Dinv_current);
  arma::mat R_current = cors;
  arma::mat Rinv_current = safe_inv_sympd(R_current);

  arma::mat D = arma::eye(k, k);

  arma::mat beta = arma::zeros(p, k);
  arma::mat gamma = arma::zeros(p, k);
  arma::mat Xbhat = X * beta;
  arma::mat w = arma::zeros(n, k);

  arma::cube pcors_mcmc(k, k, iter, arma::fill::zeros);
  arma::cube beta_mcmc(p, k, iter, arma::fill::zeros);
  arma::cube thresh_store(iter, K + 1, k, arma::fill::zeros);
  arma::ivec levels_per_var(k);
  for (int var = 0; var < k; ++var) {
    arma::vec col = Y.col(var);
    arma::uvec finite_idx = arma::find_finite(col);
    int max_cat = 1;
    if (!finite_idx.is_empty()) {
      double max_val = arma::max(col.elem(finite_idx));
      max_cat = std::max(1, static_cast<int>(std::round(max_val)));
    }
    levels_per_var(var) = std::min(max_cat, K);
  }

  arma::mat current_thresh(k, K + 1, arma::fill::zeros);
  current_thresh.fill(arma::datum::inf);
  for (int var = 0; var < k; ++var) {
    current_thresh(var, 0) = -arma::datum::inf;
    int max_level = levels_per_var(var);
    current_thresh(var, max_level) = arma::datum::inf;
    for (int level = 1; level < max_level; ++level) {
      current_thresh(var, level) = static_cast<double>(level) - 1.0;
    }
  }

  arma::mat z = arma::zeros(n, k);
  arma::mat Psi = arma::eye(k, k);
  arma::mat S_Y(k, k, arma::fill::zeros);
  for (int s = 0; s < iter; ++s) {
    pr.increment();
    if ((s + 1) % 250 == 0) {
      Rcpp::checkUserInterrupt();
    }

    for (int i = 0; i < k; ++i) {
      arma::mat R_oo = remove_row(remove_col(R_current, i), i);
      arma::vec R_io = Sigma_i_not_i(R_current, i).t();

      arma::vec weights = safe_solve_sympd_vec(R_oo, R_io);
      double cond_var = R_current(i, i) - arma::dot(R_io, weights);
      cond_var = std::max(cond_var, 1e-8);
      double cond_sd = std::sqrt(cond_var);

      arma::mat z_minus = remove_col(z, i);
      arma::mat xb_minus = remove_col(Xbhat, i);
      arma::vec cond_mean = Xbhat.col(i) + (z_minus - xb_minus) * weights;

      int max_level = levels_per_var(i);
      for (int j = 0; j < n; ++j) {
        int cat = static_cast<int>(std::round(Y(j, i)));
        if (cat < 1) {
          cat = 1;
        }
        if (cat > max_level) {
          cat = max_level;
        }
        double lower = current_thresh(i, cat - 1);
        double upper = current_thresh(i, cat);
        z(j, i) = sample_truncated_normal(cond_mean(j), cond_sd, lower, upper);
      }
    }

    for (int var = 0; var < k; ++var) {
      int max_level = levels_per_var(var);
      for (int level = 1; level < max_level; ++level) {
        arma::uvec idx_lower = arma::find(arma::round(Y.col(var)) == level);
        arma::uvec idx_upper = arma::find(arma::round(Y.col(var)) == (level + 1));

        double lower = current_thresh(var, level - 1);
        if (!idx_lower.is_empty()) {
          double max_lower = arma::max(z.col(var).elem(idx_lower));
          if (max_lower > lower) {
            lower = max_lower;
          }
        }

        double upper = current_thresh(var, level + 1);
        if (!idx_upper.is_empty()) {
          double min_upper = arma::min(z.col(var).elem(idx_upper));
          if (min_upper < upper) {
            upper = min_upper;
          }
        }

        if (!(upper > lower)) {
          upper = lower + 1e-8;
        }

        current_thresh(var, level) = R::runif(lower, upper);
      }

      for (int level = max_level + 1; level <= K; ++level) {
        current_thresh(var, level) = arma::datum::inf;
      }
    }
    arma::rowvec z_means = arma::mean(z, 0);
    for (int var = 0; var < k; ++var) {
      double shift = z_means(var);
      z.col(var) -= shift;
      int max_level = levels_per_var(var);
      for (int level = 1; level < max_level; ++level) {
        double val = current_thresh(var, level);
        if (std::isfinite(val)) {
          current_thresh(var, level) = val - shift;
        }
        D(i, i) = std::sqrt(1.0 / R::rgamma((delta + k - 1.0) / 2.0, 2.0 / diag_val));
      }
    }

    for (int i = 0; i < k; ++i) {
      double rate = std::max(Rinv_current(i, i), 1e-12);
      double draw = R::rgamma((delta + k - 1.0) / 2.0, 2.0 / rate);
      draw = std::max(draw, 1e-12);
      D(i, i) = std::sqrt(1.0 / draw);
    }

    w = z * D;

    arma::mat M = Sinv_X * X.t() * w;
    arma::mat kron_cov = arma::symmatu(kron(Sigma_current, Sinv_X));
    arma::vec gamma_vec = mvnrnd(reshape(M, k * p, 1), kron_cov);
    gamma = reshape(gamma_vec, p, k);

    arma::mat beta_current = gamma * Dinv_current;
    beta_mcmc.slice(s) = reshape(beta_current, p, k);
    Xbhat = X * beta_current;

    S_Y = w.t() * w + I_k - M.t() * S_X * M;
    S_Y = arma::symmatu(S_Y);

    Psi = wishrnd(safe_inv_sympd(BMPinv + Theta_current), nuMP + deltaMP + k - 1);
    Theta_current = wishrnd(safe_inv_sympd(S_Y + Psi), (deltaMP + k - 1) + (n - 1));
    Theta_current = arma::symmatu(Theta_current);

    Sigma_current = safe_inv_sympd(Theta_current);

    sigma_diag = Sigma_current.diag();
    sigma_diag.transform([](double val) { return std::max(val, 1e-8); });
    inv_sqrt_sigma = 1.0 / arma::sqrt(sigma_diag);
    Dinv_current = arma::diagmat(inv_sqrt_sigma);

    arma::mat inv_sqrt_sigma_mat = Dinv_current;
    cors = arma::symmatu(inv_sqrt_sigma_mat * Sigma_current * inv_sqrt_sigma_mat);

    arma::vec theta_diag = Theta_current.diag();
    theta_diag.transform([](double val) { return std::max(val, 1e-8); });
    arma::vec inv_sqrt_theta = 1.0 / arma::sqrt(theta_diag);
    arma::mat inv_sqrt_theta_mat = arma::diagmat(inv_sqrt_theta);
    arma::mat pcors = arma::symmatu(inv_sqrt_theta_mat * Theta_current * inv_sqrt_theta_mat);

    R_current = cors;
    Rinv_current = safe_inv_sympd(R_current);

    pcors_mcmc.slice(s) = -(pcors - I_k);

    for (int var = 0; var < k; ++var) {
      thresh_store.slice(var).row(s) = current_thresh.row(var);
    }
  }

  arma::cube fisher_z = atanh(pcors_mcmc);
  arma::mat pcor_mat;
  if (iter > 50) {
    pcor_mat = mean(pcors_mcmc.tail_slices(iter - 50), 2);
  } else {
    pcor_mat = mean(pcors_mcmc, 2);
  }

  Rcpp::List ret;
  ret["pcors"] = pcors_mcmc;
  ret["pcor_mat"] = pcor_mat;
  ret["beta"] = beta_mcmc;
  ret["thresh"] = thresh_store;
  ret["fisher_z"] = fisher_z;
  return ret;
}

// mixed data sampler
// [[Rcpp::export]]
Rcpp::List  copula(arma::mat z0_start,
                   arma::mat levels,
                   arma::vec K,
                   arma::mat Sigma_start,
                   int iter,
                   float delta,
                   float epsilon,
                   arma::vec idx,
                   bool progress
                   ) {

  Progress pr(iter, progress);

  int n = z0_start.n_rows;
  int k = z0_start.n_cols;

  arma::mat I_k(k, k, arma::fill::eye);

  int nu = 1 / epsilon;
  int nuMP = delta + k - 1;
  int deltaMP = nu - k + 1;

  arma::mat B = epsilon * I_k;
  arma::mat BMP = safe_inv_sympd(B);
  arma::mat BMPinv = safe_inv_sympd(BMP);

  arma::mat Sigma_current = stabilize_pd(arma::symmatu(Sigma_start));
  arma::mat Theta_current = safe_inv_sympd(Sigma_current);

  arma::cube pcors_mcmc(k, k, iter, arma::fill::zeros);
  arma::ivec K_int = arma::conv_to<arma::ivec>::from(K);
  int K_max = 0;
  if (!K_int.is_empty()) {
    K_max = K_int.max();
  }
  if (K_max < 1) {
    K_max = 1;
  }

  arma::mat current_thresh(k, K_max + 1, arma::fill::zeros);
  current_thresh.fill(0.0);
  for (int var = 0; var < k; ++var) {
    if (idx(var) >= 0.5) {
      int K_var = std::max(1, K_int(var));
      current_thresh(var, 0) = -arma::datum::inf;
      current_thresh(var, K_var) = arma::datum::inf;
      for (int level = 1; level < K_var; ++level) {
        current_thresh(var, level) = static_cast<double>(level) - 1.0;
      }
      for (int level = K_var + 1; level <= K_max; ++level) {
        current_thresh(var, level) = arma::datum::inf;
      }
    }
  }

  arma::cube thresh_store(iter, K_max + 1, k, arma::fill::zeros);

  arma::imat level_codes(levels.n_rows, levels.n_cols);
  for (arma::uword j = 0; j < levels.n_rows; ++j) {
    for (arma::uword i = 0; i < levels.n_cols; ++i) {
      double val = levels(j, i);
      if (!std::isfinite(val)) {
        level_codes(j, i) = 0;
      } else {
        int rounded = static_cast<int>(std::round(val));
        level_codes(j, i) = std::max(0, rounded);
      }
    }
  }

  arma::mat z = z0_start;
  z.replace(arma::datum::nan, 0.0);
  for (int s = 0; s < iter; ++s) {
    pr.increment();
    if ((s + 1) % 250 == 0) {
      Rcpp::checkUserInterrupt();
    }

    for (int i = 0; i < k; ++i) {
      arma::mat Sigma_oo = remove_row(remove_col(Sigma_current, i), i);
      arma::vec Sigma_io = Sigma_i_not_i(Sigma_current, i).t();

      arma::vec weights = safe_solve_sympd_vec(Sigma_oo, Sigma_io);
      double cond_var = Sigma_current(i, i) - arma::dot(Sigma_io, weights);
      cond_var = std::max(cond_var, 1e-8);
      double cond_sd = std::sqrt(cond_var);

      arma::mat z_minus = remove_col(z, i);
      arma::vec cond_mean = z_minus * weights;

      if (idx(i) >= 0.5) {
        int K_var = std::max(1, K_int(i));
        for (int j = 0; j < n; ++j) {
          int cat = level_codes(j, i);
          if (cat <= 0) {
            continue;
          }
          if (cat > K_var) {
            cat = K_var;
          }
          double lower = current_thresh(i, cat - 1);
          double upper = current_thresh(i, cat);
          z(j, i) = sample_truncated_normal(cond_mean(j), cond_sd, lower, upper);
        }
      } else {
        arma::vec draws = cond_mean + cond_sd * arma::randn<arma::vec>(n);
        z.col(i) = draws;
      }

    for (int var = 0; var < k; ++var) {
      if (idx(var) < 0.5) {
        continue;
      }
      int K_var = std::max(1, K_int(var));
      for (int level = 1; level < K_var; ++level) {
        arma::uvec idx_lower = arma::find(level_codes.col(var) == level);
        arma::uvec idx_upper = arma::find(level_codes.col(var) == (level + 1));

        double lower = current_thresh(var, level - 1);
        if (!idx_lower.is_empty()) {
          double max_lower = arma::max(z.col(var).elem(idx_lower));
          if (max_lower > lower) {
            lower = max_lower;
          }
        }

        double upper = current_thresh(var, level + 1);
        if (!idx_upper.is_empty()) {
          double min_upper = arma::min(z.col(var).elem(idx_upper));
          if (min_upper < upper) {
            upper = min_upper;
          }
        }

        if (!(upper > lower)) {
          upper = lower + 1e-8;
        }

        current_thresh(var, level) = R::runif(lower, upper);
      }

    arma::rowvec z_means = arma::mean(z, 0);
    for (int var = 0; var < k; ++var) {
      double shift = z_means(var);
      z.col(var) -= shift;
      if (idx(var) >= 0.5) {
        int K_var = std::max(1, K_int(var));
        for (int level = 1; level < K_var; ++level) {
          double val = current_thresh(var, level);
          if (std::isfinite(val)) {
            current_thresh(var, level) = val - shift;
          }
          update_thresholds_uniform(levels.col(var), z_current.col(var), thresholds.row(var), K_var);
        }
      }
    }
    arma::mat S_Y = arma::symmatu(z.t() * z);

    arma::mat Psi = wishrnd(safe_inv_sympd(BMPinv + Theta_current), nuMP + deltaMP + k - 1);
    Theta_current = wishrnd(safe_inv_sympd(S_Y + Psi), (deltaMP + k - 1) + (n - 1));
    Theta_current = arma::symmatu(Theta_current);

    Sigma_current = safe_inv_sympd(Theta_current);

    arma::vec theta_diag = Theta_current.diag();
    theta_diag.transform([](double val) { return std::max(val, 1e-8); });
    arma::vec inv_sqrt_theta = 1.0 / arma::sqrt(theta_diag);
    arma::mat inv_sqrt_theta_mat = arma::diagmat(inv_sqrt_theta);
    arma::mat pcors = arma::symmatu(inv_sqrt_theta_mat * Theta_current * inv_sqrt_theta_mat);

    pcors_mcmc.slice(s) = -(pcors - I_k);

    for (int var = 0; var < k; ++var) {
      thresh_store.slice(var).row(s) = current_thresh.row(var);
    }
  }

  arma::cube fisher_z = atanh(pcors_mcmc);
  arma::mat pcor_mat;
  if (iter > 50) {
    pcor_mat = mean(pcors_mcmc.tail_slices(iter - 50), 2);
  } else {
    pcor_mat = mean(pcors_mcmc, 2);
  }

  Rcpp::List ret;
  ret["pcors"] = pcors_mcmc;
  ret["pcor_mat"] = pcor_mat;
  ret["fisher_z"] = fisher_z;
  ret["thresh"] = thresh_store;
  return ret;
}

// partials to correlations
// [[Rcpp::export]]
Rcpp::List pcor_to_cor_internal(arma::cube x, int p) {

  // slices
  int iter = x.n_slices;

  // correlation matrix (R)
  arma::cube R(p, p, iter);

  // precision matrix
  arma::mat Theta_s(p, p);

  // sigma matrix
  arma::mat Sigma_s(p, p);

  for(int s = 0; s < iter; ++s){

    Theta_s =   x.slice(s);

    for(int j = 0; j < p; ++j) {

      Theta_s.col(j).row(j) = 1;

    }

    arma::mat Sigma_s = inv(Theta_s);

    R.slice(s) =  diagmat(1 / sqrt(Sigma_s.diag())) * Sigma_s * diagmat(1 / sqrt(Sigma_s.diag()));

  }

  arma::mat R_mean = mean(R, 2);
  Rcpp::List ret;
  ret["R"] = R;
  ret["R_mean"] = R_mean;
  return ret;
}


// partials to correlations
// [[Rcpp::export]]
Rcpp::List predictability_helper(arma::mat Y,
                                 arma::colvec y,
                                 arma::cube XX,
                                 arma::mat Xy,
                                 int n,
                                 int iter) {


  arma::mat r2(iter, 1, arma::fill::zeros);
  arma::mat ppc(n,1);

  for(int s = 0; s < iter; ++s){

    // coefs
    arma::mat coefs = Xy.col(s).t() * inv(XX.slice(s));

    // predict mean
    arma::mat mu = Y * coefs.t();

    for(int j = 0; j < n; ++j){

      // ppc person j
      arma::vec ppc_j = Rcpp::rnorm(1, mu[j], stddev(mu - y));

      // y rep
      ppc.row(j) = arma::as_scalar(ppc_j);

    }

    // bayes R2
    r2.row(s) = var(mu) / var(ppc);

  }

  Rcpp::List ret;
  ret["r2"] = r2;
  return ret;
}

// regression coefficients
// [[Rcpp::export]]
Rcpp::List beta_helper_fast(arma::cube XX,
                       arma::mat Xy,
                       int p,
                       int iter) {


  arma::mat coefs(iter, p, arma::fill::zeros);

  for(int s = 0; s < iter; ++s){

    arma::vec coef_s = inv(XX.slice(s)).t() * Xy.col(s);
    coefs.row(s) =  coef_s.t();

  }

  Rcpp::List ret;
  ret["coefs"] = coefs;
  return ret;
}


// [[Rcpp::export]]
Rcpp::List pred_helper_latent(arma::mat Y,
                       arma::cube XX,
                       arma::mat Xy,
                       arma::vec quantiles,
                       int n,
                       int iter) {


  arma::mat yhat(iter, n, arma::fill::zeros);

  for(int s = 0; s < iter; ++s){

    arma::vec yhat_s = Y * inv(XX.slice(s)).t() * Xy.col(s) ;

    yhat.row(s) =  yhat_s.t();

  }

  // yhat
  arma::mat yhat_mean = mean(yhat);

  // quantiles
  arma::mat yhat_quantiles = quantile(yhat, quantiles);

  arma::mat yhat_sd = stddev(yhat);

  // returned
  Rcpp::List ret;
  ret["yhat"] = yhat;
  ret["yhat_mean"] = yhat_mean;
  ret["yhat_sd"] = yhat_sd;
  ret["yhat_quantiles"] = yhat_quantiles;
  return ret;
}


// [[Rcpp::export]]
float KL_univariate(float var_1, float var_2){

  float kl = log(sqrt(var_2)/sqrt(var_1)) + (var_1/(2 * var_2)) -  0.5;
  return kl;

}

// [[Rcpp::export]]
Rcpp::List ppc_helper_nodewise_fast(arma::cube Theta,
                                    int n1,
                                    int n2,
                                    int p){

  int iter = Theta.n_slices;

  Progress  pr(iter, TRUE);

  arma::vec mu(p, arma::fill::zeros);

  arma::mat kl(iter, p,  arma::fill::zeros);

  for(int s = 0; s < iter; ++s){

    pr.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }

    arma::mat Sigma = inv(Theta.slice(s));

    arma::mat R =  diagmat(1 / sqrt(Sigma.diag())) * Sigma * diagmat(1 / sqrt(Sigma.diag()));

    arma::mat Yrep_1  = mvnrnd(mu, R, n1).t();
    arma::mat Yrep_2  = mvnrnd(mu, R, n2).t();

    arma::mat Yrep_cor_1 = cor(Yrep_1);
    arma::mat Yrep_cor_2 = cor(Yrep_2);

    for(int j = 0; j < p; ++j){

      arma::mat pred_1 =  remove_col(Yrep_1, j) * trans(Sigma_i_not_i(Yrep_cor_1, j) *
        inv(remove_row(remove_col(Yrep_cor_1, j), j)));

      arma::mat pred_2 =  remove_col(Yrep_2, j) * trans(Sigma_i_not_i(Yrep_cor_2, j) *
        inv(remove_row(remove_col(Yrep_cor_2, j), j)));

      arma::mat var_1 = var(pred_1);
      arma::mat var_2 = var(pred_2);

      kl(s, j) = (KL_univariate(arma::as_scalar(var_1), arma::as_scalar(var_2)) +
        KL_univariate(arma::as_scalar(var_2), arma::as_scalar(var_1)))  * 0.5;

    }
  }

  Rcpp::List ret;
  ret["kl"] = kl;
  return ret;

}


// [[Rcpp::export]]
double KL_divergnece_mvn(arma::mat Theta_1, arma::mat Theta_2) {

  // number of variables
  int p = Theta_1.n_cols;

  arma::mat Sigma_1 = inv(Theta_1);

  // identity matrix
  arma::mat  I_p(p, p, arma::fill::eye);

  double kl = 0.50 * (trace(Sigma_1 * Theta_2) -  log(det(Sigma_1 * Theta_2)) - p);

  return kl;


}

// [[Rcpp::export]]
float sum_squares(arma::mat Rinv_1, arma::mat Rinv_2){

  arma::rowvec ss = sum(square(Rinv_1 - Rinv_2), 0);

  return sum(ss);

}

// [[Rcpp::export]]
arma::vec my_dnorm( arma::vec x, arma::vec means, arma::vec sds){

  int n = x.size();

  arma::vec res(n);

  for( int i=0; i < n; i++) res[i] = R::dnorm(x[i], means[i], sds[i],  FALSE) ;

  return res;
}

// [[Rcpp::export]]
float hamming_distance(arma::mat Rinv_1,
                       arma::mat Rinv_2,
                       float df1,
                       float df2,
                       float dens,
                       int pcors,
                       float BF_cut){

  // approximate post sd
  arma::mat se_1  = sqrt((1 - square(Rinv_1)) / (df1));
  arma::mat se_2  = sqrt((1 - square(Rinv_2)) / (df2));

  // upper-triangular
  arma::uvec ids = arma::trimatu_ind(size(se_1), 1);

  // partial correlations
  arma::vec r_1 = Rinv_1(ids);
  arma::vec r_2 = Rinv_2(ids);

  // sds
  arma::vec se_up_1 = se_1(ids);
  arma::vec se_up_2 = se_2(ids);

  // matrix of zeros
  arma::vec zerovec(pcors, arma::fill::zeros);

  // mat for 0's and 1's
  arma::vec sig_1(pcors, arma::fill::zeros);
  arma::vec sig_2(pcors, arma::fill::zeros);

  // density at zero
  arma::vec dens_1 = my_dnorm(zerovec, r_1, se_up_1);
  arma::vec dens_2 = my_dnorm(zerovec, r_2, se_up_2);

  //
  for(int i = 0; i < pcors; ++i){

    // check BF_cut (group 1)
    if((1 / (dens_1(i) / dens))  > BF_cut){

      sig_1(i) = 1;

    } else {

      sig_1(i) = 0;

    }

    // check BF_cut (group 2)
    if((1 / (dens_2(i) / dens))  > BF_cut){

      sig_2(i) = 1;

    } else {

      sig_2(i) = 0;

    }
  }

  return sum(square(sig_1 - sig_2));


}

// [[Rcpp::export]]
float correlation(arma::mat Rinv_1,
                  arma::mat Rinv_2){

  arma::uvec ids = arma::trimatu_ind(size(Rinv_1), 1);

  arma::vec r_1 = Rinv_1(ids);

  arma::vec r_2 = Rinv_2(ids);

  arma::mat cor_nets = cor(r_1, r_2);

  float cors = arma::as_scalar(cor_nets);

  return cors;

}


// [[Rcpp::export]]
Rcpp::List ppc_helper_fast(arma::cube Theta,
                           int n1,
                           int n2,
                           int p,
                           float BF_cut,
                           float dens,
                           bool ppc_ss,
                           bool ppc_cors,
                           bool ppc_hd){

  int iter = Theta.n_slices;

  // progress bar
  Progress  pr(iter, TRUE);

  // mean vector
  arma::vec mu(p, arma::fill::zeros);

  // KL storage
  arma::vec kl(iter, arma::fill::zeros);

  // sum of squares storage
  arma::vec ss(iter, arma::fill::zeros);

  // hamming distance storage
  arma::vec hd(iter, arma::fill::zeros);

  // correlation storage
  arma::vec cors(iter, arma::fill::zeros);

  arma::mat Yrep_Rinv_1(p, p);

  int df1 = n1 - (p-2) - 2;

  int df2 = n2 - (p-2) - 2;

  int pcors = (p * (p - 1)) * 0.5;

  for(int s = 0; s < iter; ++s){

    pr.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }

    arma::mat Sigma = inv(Theta.slice(s));

    arma::mat R =  diagmat(1 / sqrt(Sigma.diag())) * Sigma * diagmat(1 / sqrt(Sigma.diag()));

    arma::mat Yrep_1  = mvnrnd(mu, R, n1).t();
    arma::mat Yrep_2  = mvnrnd(mu, R, n2).t();

    arma::mat Yrep_Theta_1 = inv(cov(Yrep_1));
    arma::mat Yrep_Theta_2 = inv(cov(Yrep_2));

    arma::mat Yrep_Rinv_1 = diagmat(1 / sqrt(Yrep_Theta_1.diag())) *
      Yrep_Theta_1 *
      diagmat(1 / sqrt(Yrep_Theta_1.diag()));

    arma::mat Yrep_Rinv_2 = diagmat(1 / sqrt(Yrep_Theta_2.diag())) *
      Yrep_Theta_2 *
      diagmat(1 / sqrt(Yrep_Theta_2.diag()));

    kl(s) = 0.5 * (KL_divergnece_mvn(Yrep_Rinv_1, Yrep_Rinv_2) +
      KL_divergnece_mvn(Yrep_Rinv_2, Yrep_Rinv_1));

    if(ppc_ss){

      ss(s) = sum_squares(Yrep_Rinv_1, Yrep_Rinv_2) * 0.50;

    }

    if(ppc_hd){

      hd(s) = hamming_distance(Yrep_Rinv_1,
         Yrep_Rinv_2,
         df1, df2, dens,
         pcors, BF_cut);
    }

    if(ppc_cors){

      cors(s) = correlation(Yrep_Rinv_1, Yrep_Rinv_2);

    }

  }

  Rcpp::List ret;
  ret["kl"] = kl;
  ret["ss"] = ss;
  ret["hd"] = hd;
  ret["cors"] = cors;
  return ret;

}


// [[Rcpp::export]]
arma::mat mvnrnd(int n, arma::vec mu, arma::mat Sigma){

  arma::mat Y = mvnrnd(mu, Sigma, n).t();

  return Y;

}


// [[Rcpp::export]]
Rcpp::List var(arma::mat Y,
                         arma::mat X,
                         float delta,
                         float epsilon,
                         arma::mat beta_prior,
                         int iter,
                         arma::mat start,
                         bool progress){


  // progress
  Progress  pr(iter, progress);

  // number of rows
  int n = Y.n_rows;

  // number of dependent variables
  int k = Y.n_cols;

  // number of predictors
  int p = X.n_cols;

  int nu = 1/ epsilon;

  // #nu in Mulder & Pericchi (2018) formula (30) line 1.
  int nuMP = delta + k - 1;

  // #delta in Mulder & Pericchi (2018) formula (30) line 1.
  int deltaMP = nu - k + 1 ;

  // k * k identity mat
  arma::mat  I_k(k, k, arma::fill::eye);

  // p * p identity mat
  arma::mat  I_p(p, p, arma::fill::eye);

  // scatter matrix X' * X
  arma::mat S_X(X.t() * X + beta_prior);

  // inv S_X
  arma::mat Sinv_X(inv(S_X));

  // Psi update
  arma::cube Psi(k, k, 1, arma::fill::zeros);

  // scatter matrix dependent variables
  arma::mat S_Y(k, k, arma::fill::zeros);
  arma::mat B(epsilon*I_k);
  arma::mat BMP(inv(B));
  arma::mat BMPinv(inv(BMP));

  // precison matrix
  arma::cube Theta(k, k, 1, arma::fill::zeros);
  arma::cube Theta_mcmc(k, k, iter, arma::fill::zeros);

  // partial correlations
  arma::mat pcors(k,k);
  arma::cube pcors_mcmc(k, k, iter, arma::fill::zeros);

  // correlations
  arma::mat  cors(k,k);
  arma::cube cors_mcmc(k, k, iter, arma::fill::zeros);

  // covariance matrix
  arma::cube Sigma(k, k, 1, arma::fill::zeros);
  arma::cube Sigma_mcmc(k, k, iter, arma::fill::zeros);

  // coefficients
  arma::mat beta(p, k, arma::fill::zeros);
  arma::cube beta_mcmc(p, k, iter,  arma::fill::zeros);

  // starting value
  Sigma.slice(0) = inv(start);
  Psi.slice(0).fill(arma::fill::eye);
  Theta.slice(0) = start;

  for(int s = 0; s < iter; ++s){

    pr.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }

    // draw coefficients
    beta = reshape(mvnrnd(reshape(Sinv_X * X.t() * Y, k*p , 1),
                          kron(Sigma.slice(0), Sinv_X)),
                          p, k);

    // scatter matrix
    S_Y = Y.t() * Y + I_k - beta.t() * S_X * beta;

    // sample Psi
    Psi.slice(0) = wishrnd(safe_inv_sympd(BMPinv + Theta.slice(0)), nuMP + deltaMP + k - 1);

    // sample Theta
    Theta.slice(0) =   wishrnd(safe_inv_sympd(Psi.slice(0) + S_Y),  (deltaMP + k - 1) + (n - 1));

    // Sigma
    Sigma.slice(0) = inv(Theta.slice(0));

    // correlation
    cors =  diagmat(1 / sqrt(Sigma.slice(0).diag())) *
      Sigma.slice(0) *
      diagmat(1 / sqrt(Sigma.slice(0).diag()));

    // partial correlations
    pcors = diagmat(1 / sqrt(Theta.slice(0).diag())) *
      Theta.slice(0) *
      diagmat(1 / sqrt(Theta.slice(0).diag()));


    beta_mcmc.slice(s) = beta;
    pcors_mcmc.slice(s) =  -(pcors - I_k);
  }

  arma::cube fisher_z = atanh(pcors_mcmc);

  arma::mat  pcor_mat = mean(pcors_mcmc.tail_slices(iter - 50), 2);

  Rcpp::List ret;
  ret["pcors"] = pcors_mcmc;
  ret["pcor_mat"] =  pcor_mat;
  ret["beta"] = beta_mcmc;
  ret["fisher_z"] = fisher_z;
  return ret;
}


// [[Rcpp::export]]
Rcpp::List hft_algorithm(arma::mat Sigma, arma::mat adj, double tol, double max_iter) {

  arma::mat S = Sigma;
  arma::mat W = S;

  arma::uvec upper_indices = trimatu_ind( size(S) );
  arma::mat W_previous = S;
  double p = S.n_cols;
  arma::mat iter(1,1, arma::fill::zeros);
  double max_diff = 100;
  arma::mat w_12(1, p-1);

  while(max_diff > tol){

    for(int i = 0; i < p; ++i){

      arma::mat beta(1,p-1, arma::fill::zeros);
      arma::uvec pad_index =  find(Sigma_i_not_i(adj, i) == 1);

      if(pad_index.n_elem == 0 ){
        w_12 = beta;
      } else {

        arma::mat W_11 = remove_row(remove_col(W , i), i);
        arma::mat s_12 = Sigma_i_not_i(S,i);

        arma::mat W_11_star = W_11.submat(pad_index, pad_index);
        arma::mat s_12_star = s_12(pad_index);

        beta(pad_index) = inv(W_11_star) * s_12_star;
        arma::mat w_12 = W_11 * beta.t();
        arma::mat temp = W.col(i).row(i);
        w_12.insert_rows(i, temp);

        for(int k = 0; k < p; ++k){
          W.row(i).col(k) = w_12(k);
          W.row(k).col(i) = w_12(k);
        }

        max_diff = max(W.elem(upper_indices) -  W_previous.elem(upper_indices));
        W_previous = W;
      }
    }

    iter(0,0) = iter(0,0) + 1;

    if(iter(0,0) == max_iter){
      break;
    }

  }

  arma::mat Theta = inv(W) % adj;
  Rcpp::List ret;
  ret["Theta"] = Theta;
  return ret;
}

// [[Rcpp::export]]
double bic_fast(arma::mat Theta, arma::mat S, double n, float prior_prob){

  arma::mat UU = trimatu(Theta,  1);

  arma::vec nonzero = nonzeros(UU);
  double neg_ll =  -2 * ((n*0.5) * (log(det(Theta)) - trace(S * Theta)));

  double bic = neg_ll + (nonzero.n_elem * log(n) - (nonzero.n_elem * log(prior_prob / (1 - prior_prob))));

  return bic;
}

// [[Rcpp::export]]
Rcpp::List find_ids(arma::mat x){
  arma::mat UU = trimatu(x,  1);
  arma::uvec alt_lower_indices = trimatl_ind( size(x),  -1);
  UU.elem(alt_lower_indices).fill(10);
  UU.diag().fill(10);
  arma::uvec zero = find(UU == 0);
  arma::uvec nonzero = find(UU == 1);

  Rcpp::List ret;
  ret["nonzero"] = nonzero;
  ret["zero"] = zero;
  return ret;
}

// Search for possible graphs, accepting if BIC_new is better than BIC_old
// MH algo does not work probabilistically - accept only better, never worse.
// Revisit this function 
// [[Rcpp::export]]
Rcpp::List search(arma::mat S,
                  float iter,
                  double old_bic,
                  arma::mat start_adj,
                  float n,
                  float gamma,
                  int stop_early,
                  bool progress){

  Progress  pr(iter, progress);

  int p = S.n_cols;

  arma::cube adj(p, p, iter);
  
  // Copy start_adj to adj_s
  arma::mat adj_s = start_adj; 

  // Create start object containing position of zero and nonzeros
  Rcpp::List start = find_ids(start_adj);
  
  arma::uvec zeros = start["zero"];
  arma::uvec nonzeros = start["nonzero"];

  // Use adj_start from R
  arma::mat mat_old = start_adj;
  // Initialize adj_mat to match mat_old
  arma::mat adj_mat = mat_old;    
  
  // initialize vectors
  arma::vec bics(iter, arma::fill::zeros);
  arma::vec acc(1, arma::fill::zeros);
  arma::vec repeats(1, arma::fill::zeros);
 
  // Loop through iterations
  for(int s = 0; s < iter; ++s){

    // Incrementing progress bar
    pr.increment();

    // Catch user abort key
    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }
    
    adj_s = mat_old;

    // Flip one edge at a time
    if (s % 2 == 0){
      arma::vec id_add = Rcpp::RcppArmadillo::sample(arma::conv_to<arma::vec>::from(zeros), 1, false);
      adj_s.elem(arma::conv_to<arma::uvec>::from(id_add)).fill(1);
    } else {
      arma::vec id_add = Rcpp::RcppArmadillo::sample(arma::conv_to<arma::vec>::from(nonzeros), 1, false);
      adj_s.elem(arma::conv_to<arma::uvec>::from(id_add)).fill(0);
    }

    // Ensure that adj_mat is symmetric
    adj_mat = symmatu(adj_s);
    adj_mat.diag().fill(1);
    
    // Run the hft_algorithm and compute the BIC
    Rcpp::List fit1 = hft_algorithm(S, adj_mat, 0.00001, 10);
    arma::mat  Theta = fit1["Theta"];
    double new_bic = bic_fast(Theta, S, n, gamma);
    
    // Specifically compute delta to facilitate debugging
    double delta =  new_bic - old_bic;

    // Generate a random uniform number for probabilistic acceptance
    // double random_uniform = arma::randu();

    // Metropolis-Hastings acceptance criterion
    if(exp(-0.5 *  delta ) >= 1) { //random_uniform ){
      // go back to >=1, as accepting probabilistically has BIC creep up for unknown reason
      mat_old = adj_mat;
      adj.slice(s) = adj_mat;
      old_bic = new_bic;
      acc(0)++;
      repeats(0) = 0;
      Rcpp::List start =  find_ids(adj_mat);
      arma::uvec zeros = start["zero"];
      arma::uvec nonzeros = start["nonzero"];
    } else {
      adj.slice(s) = mat_old;
      repeats(0)++;
    }
    
    bics(s) = old_bic;

    if(repeats(0) > stop_early){
      Rcpp::Rcout << "Stopping early at iteration " << s << std::endl;
      break;
    } 
    
  }

  Rcpp::List ret;
  ret["p"] = p;
  ret["adj_mat"] = adj_mat;
  ret["bics"] = bics;
  ret["adj"]= adj;
  ret["acc"] = acc;

  return ret;
}


// random walk sampler for known graph using matrix-F

// [[Rcpp::export]]
Rcpp::List fast_g_matrix_F(arma::mat Y,
                           arma::mat adj,
                           arma::vec mu_samples,
                           arma::mat cov_samples,
                           int iter,
                           int p,
                           float N,
                           float prior_sd,
                           float kappa1,
                           bool progress){


  Progress  pr(iter, progress);

  arma::cube Theta_G(p, p, iter, arma::fill::zeros);

  arma::vec uniform(1, 1);

  arma::vec acc(1, arma::fill::zeros);

  arma::mat UU = trimatl(adj,  1);

  arma::uvec alt_lower_indices = trimatu_ind(size(adj),  1);

  UU.elem(alt_lower_indices).fill(10);

  UU.diag().fill(1);

  arma::uvec nonzero = find(UU == 1);

  arma::vec kappa_store(iter, arma::fill::zeros);

  arma::mat Theta_can1(p, p, arma::fill::zeros);

  arma::mat Theta_s1(p,p,arma::fill::zeros);

  Theta_s1.elem(nonzero) = mu_samples;

  arma::mat Theta_s = symmatl(Theta_s1);

  arma::mat SSY = Y.t() * Y;

  float log_det_s = log(det(Theta_s));

  arma::mat Binv(p, p, arma::fill::zeros);

  Binv.diag().fill(10000);

  arma::vec acc_prob(1, 1, arma::fill::zeros);

  arma::mat I_p(p, p, arma::fill::eye);

  float deltaF   = 1/(prior_sd * prior_sd) - 1;

  double logpriorF_s = (10000 - p - 1) / 2 *
                       log_det_s - (deltaF + 10000 + p - 1)/2 *
                       log(det(I_p + Theta_s * Binv));

  double log_lik_s = (N/2) * log_det_s - 0.5 * trace(SSY * Theta_s);

  double log_post_s = logpriorF_s + log_lik_s;

  for(int s = 0; s < iter; ++s){

    pr.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }


    arma::vec theta_can =  mvnrnd(mu_samples, kappa1 * cov_samples);

    Theta_can1.elem(nonzero) = theta_can.t();

    arma::mat Theta_can = symmatl(Theta_can1);

    if(Theta_can.is_sympd()) {

      float log_det_can =  log(det(Theta_can));

      double logpriorF_can = (10000 - p - 1) / 2 *
                             log_det_can - (deltaF + 10000 + p - 1) / 2 *
                             log(det(I_p + Theta_can * Binv));

      double log_lik_can = (N/2) * log_det_can - 0.5 * trace(SSY * Theta_can);

      double log_post_can = logpriorF_can + log_lik_can;

      arma::vec uniform(1, 1, arma::fill::randu);

      double test = exp(log_post_can -  log_post_s);

      if(test >  uniform(0) ){

        acc(0) = acc(0) + 1;

        Theta_s = Theta_can;

        mu_samples = theta_can;

        log_post_s = log_post_can;

      }

      acc_prob(0) = acc(0) / s;

      if(acc_prob(0) < 0.30){
        kappa1 = kappa1 * 0.9;
      }

      if(acc_prob(0) > 0.50){
        kappa1 = kappa1 * 1.1;
      }
    }

    kappa_store(s) = kappa1;

    Theta_G.slice(s) = Theta_s;

  }

  Rcpp::List ret;
  ret["acc"] = acc;
  ret["Theta_G"] = Theta_G;
  ret["acc_prob"] = acc_prob;
  ret["kappa"] = kappa_store;
  return ret;
}


// [[Rcpp::export]]
arma::cube contrained_helper(arma::cube cors,
                             arma::mat adj,
                             int iter,
                             bool progress){


  Progress  pr(iter, progress);

  int p = cors.slice(0).n_cols;

  arma::cube Theta(p, p, iter, arma::fill::zeros);

  for(int s = 0; s < iter; ++s){

    pr.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }

    Rcpp::List fit1 = hft_algorithm(cors.slice(s), adj, 0.00001, 10);

    arma::mat  Theta_s = fit1["Theta"];

    Theta.slice(s) = Theta_s;

    }

  return Theta;
}

// [[Rcpp::export]]
// [[Rcpp::export]]
Rcpp::List missing_copula(arma::mat Y,
                             arma::mat Y_missing,
                             arma::mat z0_start,
                             arma::mat Sigma_start,
                             arma::mat levels,
                             int iter_missing,
                             bool progress_impute,
                             arma::vec K,
                             arma::vec idx,
                             float epsilon,
                             float delta) {
  Progress pr(iter_missing, progress_impute);

  int n = Y.n_rows;
  int p = Y.n_cols;

  arma::mat I_p(p, p, arma::fill::eye);

  int nu = 1 / epsilon;
  int nuMP = delta + p - 1;
  int deltaMP = nu - p + 1;

  arma::mat B = epsilon * I_p;
  arma::mat BMP = safe_inv_sympd(B);
  arma::mat BMPinv = safe_inv_sympd(BMP);

  arma::mat Sigma_current = stabilize_pd(arma::symmatu(Sigma_start));
  arma::mat Theta_current = safe_inv_sympd(Sigma_current);

  arma::cube pcors_mcmc(p, p, iter_missing, arma::fill::zeros);
  arma::cube Y_collect(n, p, iter_missing, arma::fill::zeros);

  arma::ivec K_int = arma::conv_to<arma::ivec>::from(K);
  int K_max = 0;
  if (!K_int.is_empty()) {
    K_max = K_int.max();
  }
  if (K_max < 1) {
    K_max = 1;
  }

  arma::mat current_thresh(p, K_max + 1, arma::fill::zeros);
  current_thresh.fill(0.0);
  for (int var = 0; var < p; ++var) {
    if (idx(var) >= 0.5) {
      int K_var = std::max(1, K_int(var));
      current_thresh(var, 0) = -arma::datum::inf;
      current_thresh(var, K_var) = arma::datum::inf;
      for (int level = 1; level < K_var; ++level) {
        current_thresh(var, level) = static_cast<double>(level) - 1.0;
      }
      for (int level = K_var + 1; level <= K_max; ++level) {
        current_thresh(var, level) = arma::datum::inf;
      }
    }
  }

  arma::cube thresh_store(iter_missing, K_max + 1, p, arma::fill::zeros);

  arma::imat level_codes(levels.n_rows, levels.n_cols);
  for (arma::uword j = 0; j < levels.n_rows; ++j) {
    for (arma::uword i = 0; i < levels.n_cols; ++i) {
      double val = levels(j, i);
      if (!std::isfinite(val)) {
        level_codes(j, i) = 0;
      } else {
        int rounded = static_cast<int>(std::round(val));
        level_codes(j, i) = std::max(0, rounded);
      }
    }
  }

  arma::mat z = z0_start;
  z.replace(arma::datum::nan, 0.0);
  for (int s = 0; s < iter_missing; ++s) {
    pr.increment();
    if ((s + 1) % 250 == 0) {
      Rcpp::checkUserInterrupt();
    }

    for (int i = 0; i < p; ++i) {
      arma::mat Sigma_oo = remove_row(remove_col(Sigma_current, i), i);
      arma::vec Sigma_io = Sigma_i_not_i(Sigma_current, i).t();

      arma::vec weights = safe_solve_sympd_vec(Sigma_oo, Sigma_io);
      double cond_var = Sigma_current(i, i) - arma::dot(Sigma_io, weights);
      cond_var = std::max(cond_var, 1e-8);
      double cond_sd = std::sqrt(cond_var);

      arma::mat z_minus = remove_col(z, i);
      arma::vec cond_mean = z_minus * weights;

      if (idx(i) >= 0.5) {
        int K_var = std::max(1, K_int(i));
        for (int j = 0; j < n; ++j) {
          int cat = level_codes(j, i);
          if (cat <= 0) {
            continue;
          }
          if (cat > K_var) {
            cat = K_var;
          }
          double lower = current_thresh(i, cat - 1);
          double upper = current_thresh(i, cat);
          z(j, i) = sample_truncated_normal(cond_mean(j), cond_sd, lower, upper);
        }

        arma::uvec miss_idx = arma::find(Y_missing.col(i) == 1);
        for (arma::uword m = 0; m < miss_idx.n_elem; ++m) {
          arma::uword obs = miss_idx[m];
          double draw = R::rnorm(cond_mean(obs), cond_sd);
          z(obs, i) = draw;
        }
      } else {
        arma::vec draws = cond_mean + cond_sd * arma::randn<arma::vec>(n);
        z.col(i) = draws;
      }
    }

    for (int var = 0; var < p; ++var) {
      if (idx(var) < 0.5) {
        continue;
      }
      int K_var = std::max(1, K_int(var));
      for (int level = 1; level < K_var; ++level) {
        arma::uvec idx_lower = arma::find(level_codes.col(var) == level);
        arma::uvec idx_upper = arma::find(level_codes.col(var) == (level + 1));

        double lower = current_thresh(var, level - 1);
        if (!idx_lower.is_empty()) {
          double max_lower = arma::max(z.col(var).elem(idx_lower));
          if (max_lower > lower) {
            lower = max_lower;
          }
        }

        double upper = current_thresh(var, level + 1);
        if (!idx_upper.is_empty()) {
          double min_upper = arma::min(z.col(var).elem(idx_upper));
          if (min_upper < upper) {
            upper = min_upper;
          }
        }

        if (!(upper > lower)) {
          upper = lower + 1e-8;
        }

        current_thresh(var, level) = R::runif(lower, upper);
      }
    }

    arma::rowvec z_means = arma::mean(z, 0);
    for (int var = 0; var < p; ++var) {
      double shift = z_means(var);
      z.col(var) -= shift;
      if (idx(var) >= 0.5) {
        int K_var = std::max(1, K_int(var));
        for (int level = 1; level < K_var; ++level) {
          double val = current_thresh(var, level);
          if (std::isfinite(val)) {
            current_thresh(var, level) = val - shift;
          }
        }
      }
    }

    arma::mat S_Y = arma::symmatu(z.t() * z);

    arma::mat Psi = wishrnd(safe_inv_sympd(BMPinv + Theta_current), nuMP + deltaMP + p - 1);
    Theta_current = wishrnd(safe_inv_sympd(S_Y + Psi), (deltaMP + p - 1) + (n - 1));
    Theta_current = arma::symmatu(Theta_current);

    Sigma_current = safe_inv_sympd(Theta_current);

    arma::vec theta_diag = Theta_current.diag();
    theta_diag.transform([](double val) { return std::max(val, 1e-8); });
    arma::vec inv_sqrt_theta = 1.0 / arma::sqrt(theta_diag);
    arma::mat inv_sqrt_theta_mat = arma::diagmat(inv_sqrt_theta);
    arma::mat pcors = arma::symmatu(inv_sqrt_theta_mat * Theta_current * inv_sqrt_theta_mat);

    pcors_mcmc.slice(s) = -(pcors - I_p);

    Y_collect.slice(s) = z;

    for (int var = 0; var < p; ++var) {
      thresh_store.slice(var).row(s) = current_thresh.row(var);
    }
  }

  arma::cube fisher_z = atanh(pcors_mcmc);
  arma::mat pcor_mat;
  if (iter_missing > 50) {
    pcor_mat = mean(pcors_mcmc.tail_slices(iter_missing - 50), 2);
  } else {
    pcor_mat = mean(pcors_mcmc, 2);
  }

  Rcpp::List ret;
  ret["pcors"] = pcors_mcmc;
  ret["pcor_mat"] = pcor_mat;
  ret["fisher_z"] = fisher_z;
  ret["Y_collect"] = Y_collect;
  ret["thresh"] = thresh_store;
  return  ret;
}

Rcpp::List missing_copula_data(arma::mat Y,
                   arma::mat Y_missing,
                   arma::mat z0_start,
                   arma::mat Sigma_start,
                   arma::mat levels,
                   int iter_missing,
                   bool progress_impute,
                   arma::vec K,
                   arma::vec idx,
                   float lambda) {
  // progress
  Progress  pr(iter_missing, progress_impute);

  int p = Y.n_cols;

  int n = Y.n_rows;

  arma::mat Y_impute = Y;

  arma::cube Y_collect(n, p,
                       iter_missing,
                       arma::fill::zeros);

  arma::mat I_p(p, p, arma::fill::eye);

  arma::cube z0(n, p, 1,  arma::fill::zeros);

  z0.slice(0) = z0_start;

  arma::cube Sigma(p, p, 1, arma::fill::zeros);
  Sigma.slice(0) = Sigma_start;

  arma::mat mm(n,1);
  arma::mat ss(1,1);

  arma::ivec K_int = arma::conv_to<arma::ivec>::from(K);
  int K_max = 0;
  for(int var = 0; var < p; ++var){
    if(idx(var) == 1){
      int K_var = K_int(var);
      if(K_var > K_max){
        K_max = K_var;
      }
    }
  }
  if(K_max == 0){
    K_max = 1;
  }

  arma::cube thresh(iter_missing, K_max + 1, p, arma::fill::zeros);

  for(int var = 0; var < p; ++var){
    if(idx(var) == 1){
      int K_var = K_int(var);
      thresh.slice(var).col(0).fill(-arma::datum::inf);
      thresh.slice(var).col(K_var).fill(arma::datum::inf);
      for(int level = 1; level < K_var; ++level){
        thresh.slice(var).col(level).fill(level - 1);
      }
    }
  }

  for(int  s = 0; s < iter_missing; ++s){

    pr.increment();

    if (s % 250 == 0){
      Rcpp::checkUserInterrupt();
    }

    if(s > 0){
      for(int var = 0; var < p; ++var){
        if(idx(var) == 1){
          thresh.slice(var).row(s) = thresh.slice(var).row(s - 1);
        }
      }
    }

    if(s > 0){
      for(int var = 0; var < p; ++var){
        if(idx(var) != 1){
          continue;
        }

        int K_var = K_int(var);
        for(int level = 1; level < K_var; ++level){

          arma::uvec idx_current = find(levels.col(var) == level);
          arma::uvec idx_next = find(levels.col(var) == level + 1);

          double lower = thresh.slice(var)(s - 1, level - 1);
          if(idx_current.n_elem > 0){
            double max_current = arma::max(select_col(z0.slice(0), var).elem(idx_current));
            if(max_current > lower){
              lower = max_current;
            }
          }

          double upper = thresh.slice(var)(s - 1, level + 1);
          if(idx_next.n_elem > 0){
            double min_next = arma::min(select_col(z0.slice(0), var).elem(idx_next));
            if(min_next < upper){
              upper = min_next;
            }
          }

          if(lower >= upper){
            thresh.slice(var)(s, level) = lower;
          } else {
            arma::vec v = Rcpp::runif(1, lower, upper);
            thresh.slice(var)(s, level) = arma::as_scalar(v);
          }
        }
      }
    }

    for(int i = 0; i < p; ++i){

      mm = Sigma_i_not_i(Sigma.slice(0), i) *
        inv(remove_row(remove_col(Sigma.slice(0), i), i)) *
        remove_col(z0.slice(0), i).t();

      ss = select_row(Sigma.slice(0), i).col(i) -
        Sigma_i_not_i(Sigma.slice(0), i) *
        inv(remove_row(remove_col(Sigma.slice(0), i), i)) *
        Sigma_i_not_i(Sigma.slice(0), i).t();

      if(idx(i) == 1){
        int row_index = s;
        for(int j = 0; j < n; ++j){
          int cat = static_cast<int>(levels.col(i)[j]);
          if(cat == 0){
            continue;
          }
          double lower = thresh.slice(i)(row_index, cat - 1);
          double upper = thresh.slice(i)(row_index, cat);
          z0.slice(0).row(j).col(i) = R::qnorm(R::runif(
            R::pnorm(lower, mm(j), sqrt(ss(0)), TRUE, FALSE),
            R::pnorm(upper, mm(j), sqrt(ss(0)), TRUE, FALSE)),
            mm(j), sqrt(ss(0)), TRUE, FALSE);
        }
      }

      arma::vec Y_j = Y_missing.col(i);

      double check_na = sum(Y_j);


      if(check_na > 0){

        arma::uvec  index_j = find(Y_missing.col(i) == 1);

        int  n_missing = index_j.n_elem;

        for(int m = 0; m < n_missing; ++m){

          arma::vec ppd_i = Rcpp::rnorm(1,  mm(index_j[m]), sqrt(ss(0)));

          z0.slice(0).col(i).row(index_j[m]) = ppd_i(0);

        }
      }
    }

    arma::rowvec z_means = arma::mean(z0.slice(0), 0);
    for(int var = 0; var < p; ++var){
      if(idx(var) == 1){
        double shift = z_means(var);
        z0.slice(0).col(var) -= shift;
        if(shift != 0.0){
          int K_var = K_int(var);
          for(int level = 1; level < K_var; ++level){
            double current = thresh.slice(var)(s, level);
            if(std::isfinite(current)){
              thresh.slice(var)(s, level) = current - shift;
            }
          }
        }
      }
    }

    arma::mat S_Y = z0.slice(0).t() * z0.slice(0);

    Sigma.slice(0) =  safe_inv_sympd(wishrnd(safe_inv_sympd(S_Y + I_p * lambda), n + lambda));

    for(int i = 0; i < p; ++i){

      arma::mat  zz = z0.slice(0).col(i);

      ss = Sigma.slice(0).col(i).row(i);

      arma::vec Y_jnew = Y_missing.col(i);

      double check_nanew = sum(Y_jnew);

      if(check_nanew > 0){

        arma::uvec  index_j = find(Y_missing.col(i) == 1);

        int  n_missing = index_j.n_elem;

        for(int m = 0; m < n_missing; ++m){

          Y_impute.col(i).row(index_j[m]) =  quantile_type_1(Y.col(i),
                       R::pnorm(zz(index_j[m]), 0, sqrt(ss(0)), TRUE, FALSE));
          }
      }
    }

    Y_collect.slice(s) = Y_impute;

  }

  Rcpp::List ret;
  ret["Y_collect"] = Y_collect;
  ret["thresh"] = thresh;
  return  ret;
}
