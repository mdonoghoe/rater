/* Extension of Dawid and Skene's noisy categorical rating model to include
 * 'unsure' ratings.
 * This implementation requires data in a 'long' format in order to allow
 * incomplete designs. This implementation is heavily based on the implementation
 * of this model for complete designs given in the Stan Manual section TODO.
 * as well as publicly avaliable code for fitting MAP estimates via EM
 * for the same model. That code can be veiwed here:
 * This code was written before the author was aware of an implemtation of this
 * model in Paun et. al. 2018.
 */

data {
  int<lower=1> J;               // number of annotators
  int<lower=1> K;               // number of annotation categories (excluding unsure)
  int<lower=1> I;               // number of items
  int<lower=1> N;               // number of annotations (excluding unsure)
  array[N] int<lower=1, upper=I> ii;  // item index for annotation n
  array[N] int<lower=1, upper=J> jj;  // annotator for annotation n
  array[N] int<lower=1, upper=K> y;   // annotation for observation n
  int<lower=0> N0;     // number of unsure annotations
  array[N0] int<lower=1, upper=I> ii0;  // item index for unsure annotations
  array[N0] int<lower=1, upper=J> jj0;  // annotator index for unsure annotations
  vector<lower=0>[K] alpha;     // prior for pi
  array[J, K] vector<lower=0>[K] beta;   // prior for theta
  vector<lower=0>[2] diff_mu;     // hyperprior for difficulty mean
  vector<lower=0>[2] diff_kappa;  // hyperprior for difficulty sample size
  real<lower=0> conf_s;           // hyperprior for confidence SD
  real<lower=0> delta_sd;         // hyperprior for diffusion rate SD
}

parameters {
  simplex[K] pi;
  array[J, K] simplex[K] theta;
  real<lower=0, upper=1> diff_mean;
  real<lower=0> diff_ssize;
  array[I] real<lower=0, upper=1> difficulty;
  real<lower=0> conf_sigma;
  array[J] real confidence;
  real delta;
}

transformed parameters {
  array[I] vector[K] log_p_z;
  for (i in 1:I) {
    log_p_z[i] = log(pi);
  }
  for (n in 1:N) {
    real log_p_resp = log_inv_logit(logit(1 - difficulty[ii[n]]) + confidence[jj[n]]);
    real log_p_diffusion_d = log(1 + exp(logit(difficulty[ii[n]]) + delta + log(K)));
    for (k in 1:K) {
      real log_p_diffusion_n = log(theta[jj[n], k, y[n]] + exp(logit(difficulty[ii[n]]) + delta));
      // Here we marginalise over the latent discrete parameter
      log_p_z[ii[n], k] = log_p_z[ii[n], k] + log_p_diffusion_n - log_p_diffusion_d + log_p_resp;
    }
  }
  if (N0 > 0) {
    for (n0 in 1:N0) {
      log_p_z[ii0[n0]] = log_p_z[ii0[n0]] + log_inv_logit(logit(difficulty[ii0[n0]]) - confidence[jj0[n0]]);
    }
  }
}

model {
  // hyperprior on difficulty
  diff_mean ~ beta(diff_mu[1], diff_mu[2]);
  diff_ssize ~ pareto(diff_kappa[1], diff_kappa[2]);

  // prior on difficulty
  difficulty ~ beta_proportion(diff_mean, diff_ssize);

  // hyperprior on confidence
  conf_sigma ~ normal(0, conf_s);

  // prior on confidence
  confidence ~ normal(0, conf_sigma);

  // prior on diffusion rate
  delta ~ normal(0, delta_sd);

  // prior on pi
  pi ~ dirichlet(alpha);

  for (j in 1:J) {
    for (k in 1:K) {
       //prior on theta
       theta[j, k] ~ dirichlet(beta[j, k]);
    }
  }

  for (i in 1:I) {
    // log_sum_exp used for numerical stability
    target += log_sum_exp(log_p_z[i]);
  }
}

generated quantities {
  vector[I] log_lik;
  for (i in 1:I) {
    log_lik[i] = log_sum_exp(log_p_z[i]);
  }
}
