#' @name models
#'
#' @title Probabilistic models of repeated categorical rating
#' @description Functions to set up models and change their prior
#'   parameters for use in [rater()].
#'
#' @return a rater model object that can be passed to [rater()].
#'
NULL

#' @rdname models
#'
#' @param alpha prior parameter for pi
#' @param beta prior parameter for theta. This can either be a K * K matrix, in
#'   which case it is interpreted as the prior parameter of all of the J
#'   raters, or a J by K by K array in which case it is the fully specified
#'   prior parameter for all raters. (Here K is the number of categories in the
#'   data and J is the number of raters in the data.)
#'
#' @examples
#' # Model with default prior parameters:
#' default_m <- dawid_skene()
#'
#' # Changing alpha:
#' set_alpha_m <- dawid_skene(alpha = c(2, 2, 2))
#'
#' # Changing beta, single matrix:
#' # (See details for how this is interpreted.)
#' beta_mat <- matrix(1, nrow = 4, ncol = 4)
#' diag(beta_mat) <- 4
#' beta_mat_m <- dawid_skene()
#'
#' # The above is equivalent (when the model is fit - see details) to:
#' beta_array <- array(NA, dim = c(2, 4, 4))
#' for (i in 1:2) {
#'   beta_array[i, , ] <- beta_mat
#' }
#' beta_array_m <- dawid_skene(beta = beta_array)
#'
#' # But you can also specify an array where each slice is different.
#' # (Again, see details for how this is interpreted.)
#' beta_array[1, , ] <- matrix(1, nrow = 4, ncol = 4)
#' beta_array_m <- dawid_skene(beta = beta_array)
#'
#' @export
#'
dawid_skene <- function(alpha = NULL, beta = NULL) {
  validate_alpha(alpha)

  # `beta` can either be a K * K matrix which we interpret as the prior on the
  # error matrix on each of the raters as in {1.0.0} or a J * K * K array, the
  # set of priors on the error matrices of the J raters.

  if (!is.null(beta) && !is.matrix(beta) && !is.array(beta)) {
    stop("beta must be a numeric matrix or array", call. = FALSE)
  }

  alpha_k <- NULL
  beta_k <- NULL

  # Beta as matrix case.
  if (is.matrix(beta)) {

    # Test if the matrix is square.
    if (nrow(beta) != ncol(beta)) {
      stop("beta must a square matrix", call. = FALSE)
    }

    beta_k <- unique(dim(beta))
  }

  # Beta as array case.
  if (is.array(beta) && length(dim(beta)) > 2) {

    if (length(dim(beta)) != 3) {
      stop("`beta` must be a 3 dimensional array", call. = FALSE)
    }

    if (length(unique(dim(beta)[2:3])) != 1) {
      stop("Subslices of `beta` must be square matrices.", call. = FALSE)
    }

    beta_k <- unique(dim(beta)[2:3])
  }

  if (is.numeric(alpha)) {
    alpha_k <- length(alpha)
  }

  ks <- c(alpha_k, beta_k)

  # ks will be NULL if both alpha and beta are not specified.
  if (is.null(ks)) {
    K <- NULL
  } else {
    if (length(unique(ks)) > 1) {
      # FIXME: Make this error more informative.
      stop("`alpha` and `beta` are not compatible.", call. = FALSE)
    } else {
      K <- unique(ks)
    }
  }

  m <- list(parameters = list(alpha = alpha, beta = beta),
            name = "Bayesian Dawid and Skene Model",
            file = "dawid_skene",
            K = K)
  class(m) <- c("dawid_skene", "rater_model")
  m
}

#' @rdname models
#'
#' @param alpha prior parameter for pi
#'
#' @examples
#' # Default:
#' hier_dawid_skene()
#'
#' # Changing alpha
#' hier_dawid_skene(alpha = c(2, 2))
#'
#' @export
#'
hier_dawid_skene <- function(alpha = NULL) {
  # Note: this does not allow the user to change the N(0, 1) hyperpriors.
  validate_alpha(alpha)

  K <- if (!is.null(alpha)) length(alpha) else NULL

  m <- list(parameters = list(alpha = alpha),
            name = "Bayesian Hierarchical Dawid and Skene Model",
            file = "hierarchical_dawid_skene",
            K = K)
  class(m) <- c("hier_dawid_skene", "rater_model")
  m
}

#' @rdname models
#'
#' @param beta_1 First on diagonal prior probability parameter
#' @param beta_2 Second on diagonal prior probability parameter for theta
#'
#' @examples
#' # Default:
#' class_conditional_dawid_skene()
#'
#' # Not default:
#' class_conditional_dawid_skene(
#'   alpha = c(2, 2),
#'   beta_1 = c(4, 4),
#'   beta_2 = c(2, 2)
#' )
#'
#' @export
#'
class_conditional_dawid_skene <- function(alpha = NULL,
                                          beta_1 = NULL,
                                          beta_2 = NULL) {
  validate_alpha(alpha)

  # length(NULL) = 0.
  ks <- c(length(alpha), length(beta_1), length(beta_2))
  ks <- ks[ks > 0]

  if (length(unique(ks)) > 1) {
    stop("Prior parameters are not compatible.", call. = FALSE)
  }
  K <- if (length(ks) > 0) unique(ks) else NULL

  m <- list(parameters = list(alpha = alpha, beta_1 = beta_1, beta_2 = beta_2),
            name = "Bayesian Class conditional Dawid and Skene Model",
            file = "class_conditional_dawid_skene",
            K = K)
  class(m) <- c("class_conditional_dawid_skene", "rater_model")
  m
}

#' @rdname models
#'
#' @param alpha prior parameter for pi
#' @param beta prior parameter for theta. This can either be a K * K matrix, in
#'   which case it is interpreted as the prior parameter of all of the J
#'   raters, or a J by K by K array in which case it is the fully specified
#'   prior parameter for all raters. (Here K is the number of categories in the
#'   data and J is the number of raters in the data.)
#'
#' @examples
#' # Model with default prior parameters:
#' default_m <- dawid_skene()
#'
#' # Changing alpha:
#' set_alpha_m <- dawid_skene(alpha = c(2, 2, 2))
#'
#' # Changing beta, single matrix:
#' # (See details for how this is interpreted.)
#' beta_mat <- matrix(1, nrow = 4, ncol = 4)
#' diag(beta_mat) <- 4
#' beta_mat_m <- dawid_skene()
#'
#' # The above is equivalent (when the model is fit - see details) to:
#' beta_array <- array(NA, dim = c(2, 4, 4))
#' for (i in 1:2) {
#'   beta_array[i, , ] <- beta_mat
#' }
#' beta_array_m <- dawid_skene(beta = beta_array)
#'
#' # But you can also specify an array where each slice is different.
#' # (Again, see details for how this is interpreted.)
#' beta_array[1, , ] <- matrix(1, nrow = 4, ncol = 4)
#' beta_array_m <- dawid_skene(beta = beta_array)
#'
#' @export
#'
dawid_skene_unsure <- function(alpha = NULL, beta = NULL, diff_mu = NULL, diff_kappa = NULL,
                               conf_s = NULL, delta_sd = NULL) {
  validate_alpha(alpha)

  # `beta` can either be a K * K matrix which we interpret as the prior on the
  # error matrix on each of the raters as in {1.0.0} or a J * K * K array, the
  # set of priors on the error matrices of the J raters.

  if (!is.null(beta) && !is.matrix(beta) && !is.array(beta)) {
    stop("beta must be a numeric matrix or array", call. = FALSE)
  }

  alpha_k <- NULL
  beta_k <- NULL

  # Beta as matrix case.
  if (is.matrix(beta)) {

    # Test if the matrix is square.
    if (nrow(beta) != ncol(beta)) {
      stop("beta must a square matrix", call. = FALSE)
    }

    beta_k <- unique(dim(beta))
  }

  # Beta as array case.
  if (is.array(beta) && length(dim(beta)) > 2) {

    if (length(dim(beta)) != 3) {
      stop("`beta` must be a 3 dimensional array", call. = FALSE)
    }

    if (length(unique(dim(beta)[2:3])) != 1) {
      stop("Subslices of `beta` must be square matrices.", call. = FALSE)
    }

    beta_k <- unique(dim(beta)[2:3])
  }

  if (is.numeric(alpha)) {
    alpha_k <- length(alpha)
  }

  ks <- c(alpha_k, beta_k)

  # ks will be NULL if both alpha and beta are not specified.
  if (is.null(ks)) {
    K <- NULL
  } else {
    if (length(unique(ks)) > 1) {
      # FIXME: Make this error more informative.
      stop("`alpha` and `beta` are not compatible.", call. = FALSE)
    } else {
      K <- unique(ks)
    }
  }

  if (!is.null(diff_mu) && (!is.numeric(diff_mu) || length(diff_mu) != 2 ||
        any(diff_mu < 0))) {
    stop("diff_mu must be a non-negative numeric vector of length 2", call. = FALSE)
  }

  if (!is.null(diff_kappa) && (!is.numeric(diff_kappa) || length(diff_kappa) != 2 ||
        any(diff_kappa < 0))) {
    stop("diff_kappa must be a non-negative numeric vector of length 2", call. = FALSE)
  }

  if (!is.null(conf_s) && (!is.numeric(conf_s) || length(conf_s) != 1 ||
        any(conf_s < 0))) {
    stop("conf_s must be a non-negative value")
  }

  if (!is.null(delta_sd) && (!is.numeric(delta_sd) || length(delta_sd) != 1 ||
        any(delta_sd < 0))) {
    stop("delta_sd must be a non-negative value")
  }

  m <- list(parameters = list(alpha = alpha, beta = beta, diff_mu = diff_mu,
                              diff_kappa = diff_kappa, conf_s = conf_s, delta_sd = delta_sd),
            name = "Bayesian Dawid and Skene Model with unsure responses",
            file = "dawid_skene_unsure",
            K = K)
  class(m) <- c("dawid_skene_unsure", "dawid_skene", "rater_model")
  m
}

validate_alpha <- function(alpha) {
  if (!is.null(alpha) && !is.numeric(alpha)) {
      stop("alpha must be a numeric vector", call. = FALSE)
  }
}