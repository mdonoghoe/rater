#' Fit statistical models to repeated categorical rating data using Stan
#'
#' This functions allows the user to fit statistical models of noisy
#' categorical rating, based on the Dawid-Skene model, using Bayesian
#' inference. A variety of data formats and models are supported. Inference
#' is done using Stan, allowing models to be fit efficiently, using both
#' optimisation and Markov Chain Monte Carlo (MCMC).
#'
#' @param data A 2D data object: data.frame, matrix, tibble etc. with data in
#'   either long or grouped format.
#' @param model Model to fit to data - must be rater_model or a character
#'   string - the name of the model. If the character string is used, the
#'   prior parameters will be set to their default values.
#' @param method A length 1 character vector, either `"mcmc"` or `"optim"`.
#'   This will be fitting method used by Stan. By default `"mcmc"`
#' @param data_format A length 1 character vector, `"long"`, `"wide"` and
#'   `"grouped"`. The format that the passed data is in. Defaults to `"long"`.
#'   See `vignette("data-formats)` for details.
#' @param long_data_colnames A 3-element named character vector that specifies
#'   the names of the three required columns in the long data format. The vector
#'   must have the required names:
#'     * item: the name of the column containing the item indexes,
#'     * rater: the name of the column containing the rater indexes,
#'     * rating: the name of the column containing the ratings.
#'   By default, the names of the columns are the same as the names of the
#'   vector: `"item"`, `"rater"`, and `"rating"` respectively. This argument is
#'   ignored when the `data_format` argument is either `"wide"` or `"grouped"`.
#' @param inits The initialization points of the fitting algorithm
#' @param verbose Should `rater()` produce information about the progress
#'   of the chains while using the MCMC algorithm. Defaults to `TRUE`
#' @param ... Extra parameters which are passed to the Stan fitting interface.
#
#' @return An object of class rater_fit containing the fitted parameters.
#'
#' @details The default MCMC algorithm used by Stan is No U Turn Sampling
#'   (NUTS) and the default optimisation method is LGFGS. For MCMC 4 chains
#'   are run be default with 2000 iterations in total each.
#'
#' @importFrom rstan sampling optimizing
#' @importFrom utils capture.output
#'
#' @seealso [rstan::sampling()], [rstan::optimizing()]
#'
#' @examples
#' \donttest{
#'
#' # Fit a model using MCMC (the default).
#' mcmc_fit <- rater(anesthesia, "dawid_skene")
#'
#' # Fit a model using optimisation.
#' optim_fit <- rater(anesthesia, dawid_skene(), method = "optim")
#'
#' # Fit a model using passing data grouped data.
#' grouped_fit <- rater(caries, dawid_skene(), data_format = "grouped")
#'
#' }
#'
#' @export
#'
rater <- function(data,
                  model,
                  method = "mcmc",
                  data_format = "long",
                  long_data_colnames = c(
                    item = "item",
                    rater = "rater",
                    rating = "rating"
                  ),
                  inits = NULL,
                  verbose = TRUE,
                  ...
                  ) {

  method <- match.arg(method, choices = c("mcmc", "optim"))
  data_format <- match.arg(data_format, choices = c("long", "grouped", "wide", "long_unsure"))

  check_long_data_colnames(long_data_colnames, data_format)

  model <- validate_model(model)
  data <- validate_input(data, model, data_format, long_data_colnames)

  stan_data_list <- as_stan_data(data, data_format, long_data_colnames)

  # Check the priors and data are consistent.
  check_K(stan_data_list, model)

  # Create the full passed info for stan and the initialisation points.
  priors <- parse_priors(model, stan_data_list$K, stan_data_list$J)

  if (method == "optim" && inherits(model, "dawid_skene")) {
    check_beta_values(priors$beta)
  }
  stan_data <- c(stan_data_list, priors)

  if (is.null(inits)) {
    inits <- create_inits(model, stan_data_list)
  }

  # TODO This could be made more complex if automatic switching is used.
  file <- get_stan_file(data_format, model)

  if (method == "mcmc") {

    if (!verbose) {
      capture.output(samples <- rstan::sampling(stanmodels[[file]], stan_data,
                                                init = inits, ...))
    } else {
      samples <- rstan::sampling(stanmodels[[file]], stan_data, init = inits,
                                 ...)
    }

    out <- new_mcmc_fit(model, samples, stan_data, data_format)
  } else if (method == "optim") {
    estimates <- rstan::optimizing(stanmodels[[file]], stan_data, init = inits, ...)
    out <- new_optim_fit(model, estimates, stan_data, data_format)
  }

  out
}

#' Convert validated passed data into data for Stan.
#'
#' @param data Validated passed data
#' @param data_format String specifying the format of the data
#' @param long_data_colnames A named vector specifying the names of the
#'   three column names in the long data format.
#'
#' @details The function accepts validated data. So we know that the data
#'   will be a data.frame with the appropriate column names. See
#'   [validate_data()] for details.
#'
#' @return A list of component data parts as required by the Stan models.
#'
#' @noRd
#'
as_stan_data <- function(data, data_format, long_data_colnames) {

  if (data_format == "grouped") {
    tally <- data[, ncol(data)]
    key <- data[, 1:(ncol(data) - 1)]
    stan_data <- list(
      N = nrow(data),
      K = max(key),
      J = ncol(key),
      key = key,
      tally = tally
    )
    return(stan_data)
  }

  if (data_format == "wide") {
    # This also does validation.
    data <- wide_to_long(data)
  }

  # Data is now in long format.

  item_col_name <- long_data_colnames[["item"]]
  rater_col_name <- long_data_colnames[["rater"]]
  rating_col_name <- long_data_colnames[["rating"]]

  if (data_format == "long_unsure") {

    stan_data <- list(
      I = max(data[[item_col_name]]),
      J = max(data[[rater_col_name]]),
      K = max(data[[rating_col_name]]),
      N1 = sum(data[[rating_col_name]] != 0),
      N0 = sum(data[[rating_col_name]] == 0),
      idx1 = which(data[[rating_col_name]] != 0),
      idx0 = which(data[[rating_col_name]] == 0),
      ii = data[[item_col_name]],
      jj = data[[rater_col_name]],
      y = data[[rating_col_name]]
    )
    return(stan_data)

  }

  stan_data <- list(
    N = nrow(data),
    I = max(data[[item_col_name]]),
    J = max(data[[rater_col_name]]),
    K = max(data[[rating_col_name ]]),
    ii = data[[item_col_name]],
    jj = data[[rater_col_name]],
    y = data[[rating_col_name]]
  )

  stan_data
}

#' Converts default prior parameter specification to full priors
#'
#' @param model The rater_model.
#' @param K The number of categories in the data.
#' @param J The number of raters in the data
#' @param method The passed fitting method.
#'
#' @return The fully realised prior parameters
#'
#' @noRd
#'
parse_priors <- function(model, K, J) {
  switch(get_file(model),
    "dawid_skene" = ds_parse_priors(model, K, J),
    "class_conditional_dawid_skene" =
      class_conditional_ds_parse_priors(model, K),
    "hierarchical_dawid_skene" = hier_ds_parse_priors(model, K),
    "dawid_skene_unsure" = ds_unsure_parse_priors(model, K, J),
    stop("Unsupported model type", call. = FALSE))
}

ds_parse_priors <- function(model, K, J) {
  pars <- get_parameters(model)

  # This is the default uniform prior taken from the Stan manual.
  if (is.null(pars$alpha)) {
    pars$alpha <- rep(3, K)
  }

  # We need to alter the passed beta if:
  # 1. It is a matrix - and we need to convert it into an array.
  # 2. It is null - we need to create the default prior.
  # Ideally this would be done earlier but we need to to know J. The matrix
  # has already been validated i.e. it is square.

  # 1.
  # Convert from matrix to array.
  if (is.matrix(pars$beta)) {
    beta_slice <- pars$beta
    pars$beta <- array(dim = c(J, K, K))
    for (j in 1:J) {
      pars$beta[j, , ] <- beta_slice
    }
  }

  # 2.
  # This prior parameter is based on conjugate priors for the simplified model
  # where the true class in known.
  if (is.null(pars$beta)) {
    N <- 8
    p <- 0.6
    on_diag <- N * p
    off_diag <- N * (1 - p) / (K - 1)

    beta_slice <- matrix(off_diag, nrow = K, ncol = K)
    diag(beta_slice) <- on_diag

    pars$beta <- array(dim = c(J,K,K))
    for (j in 1:J) {
      pars$beta[j, , ] <- beta_slice
    }
  }

  pars
}

hier_ds_parse_priors <- function(model, K) {
  pars <- get_parameters(model)
  if (is.null(pars$alpha)) {
    pars$alpha <- rep(3, K)
  }
  pars
}

class_conditional_ds_parse_priors <- function(model, K) {
  pars <- get_parameters(model)
  if (is.null(pars$alpha)) {
    pars$alpha <- rep(3, K)
  }
  N <- 8
  p <- 0.6
  if (is.null(pars$beta_1)) {
    pars$beta_1 <- rep(N * p, K)
  }
  if (is.null(pars$beta_2)) {
    pars$beta_2 <- rep(N * (1 - p), K)
  }
  pars
}

ds_unsure_parse_priors <- function(model, K, J) {
  pars <- get_parameters(model)

  # This is the default uniform prior taken from the Stan manual.
  if (is.null(pars$alpha)) {
    pars$alpha <- rep(3, K)
  }

  # We need to alter the passed beta if:
  # 1. It is a matrix - and we need to convert it into an array.
  # 2. It is null - we need to create the default prior.
  # Ideally this would be done earlier but we need to to know J. The matrix
  # has already been validated i.e. it is square.

  # 1.
  # Convert from matrix to array.
  if (is.matrix(pars$beta)) {
    beta_slice <- pars$beta
    pars$beta <- array(dim = c(J, K, K))
    for (j in 1:J) {
      pars$beta[j, , ] <- beta_slice
    }
  }

  # 2.
  # This prior parameter is based on conjugate priors for the simplified model
  # where the true class in known.
  if (is.null(pars$beta)) {
    N <- 8
    p <- 0.6
    on_diag <- N * p
    off_diag <- N * (1 - p) / (K - 1)

    beta_slice <- matrix(off_diag, nrow = K, ncol = K)
    diag(beta_slice) <- on_diag

    pars$beta <- array(dim = c(J,K,K))
    for (j in 1:J) {
      pars$beta[j, , ] <- beta_slice
    }
  }

  if (is.null(pars$diff_mu)) {
    # From Gelman et al 2013, Chapter 5
    pars$diff_mu <- c(1, 1)
  }

  if (is.null(pars$diff_kappa)) {
    pars$diff_kappa <- c(0.1, 1.5)
  }

  if (is.null(pars$conf_s)) {
    pars$conf_s <- 3 / (qnorm(0.975) * qnorm(0.995))
  }

  if (is.null(pars$delta_sd)) {
    pars$delta_sd <- 2 / qnorm(0.995)
  }

  pars
}

#' Creates initialization points for the Stan.
#'
#' @param model rater model
#' @param stan_data data in list form
#'
#' @return Initialization points for the chains in the format required by Stan.
#'
#' @noRd
#'
create_inits <- function(model, stan_data) {
  # better to have another short unique id...
  I <- stan_data$I
  K <- stan_data$K
  J <- stan_data$J
  switch(get_file(model),
    "dawid_skene" = dawid_skene_inits(K, J),
    "class_conditional_dawid_skene" = class_conditional_dawid_skene_inits(K, J),
    "hierarchical_dawid_skene" = hier_dawid_skene_inits(K, J),
    "dawid_skene_unsure" = dawid_skene_unsure_inits(K, J, I),
    stop("Unsupported model type", call. = FALSE))
}

#' Creates initialization points for the dawid and skene model.
#'
#' @param K number of categories
#' @param J number of raters
#'
#' @return Initialization points in the format required by Stan.
#'
#' @noRd
#'
dawid_skene_inits <- function(K, J) {
  pi_init <- rep(1/K, K)
  theta_init <- array(0.2 / (K - 1), c(J, K, K))
  for (j in 1:J) {
      diag(theta_init[j, ,]) <- 0.8
  }
  function(n) list(theta = theta_init, pi = pi_init)
}

#' Creates initialization points for the class conditional model
#'
#' @param K number of categories
#' @param J number of raters
#'
#' @return initialization points in the format required by stan
#'
#' @noRd
#'
class_conditional_dawid_skene_inits <- function(K, J) {
  pi_init <- rep(1/K, K)
  theta_init <- matrix(0.8, nrow = J, ncol = K)
  function(n) list(theta = theta_init, pi = pi_init)
}

#' Creates initialization points for the Hierarchical Dawid-Skene model.
#'
#' @param K number of categories
#' @param J number of raters
#'
#' @return Initialization points in the format required by Stan.
#'
#' @noRd
#'
hier_dawid_skene_inits <- function(K, J) {
  pi_init <- rep(1 / K, K)
  mu_init <- matrix(0, nrow = K, ncol = K)
  diag(mu_init) <- 2
  # Mean of half-normal distribution.
  sigma_init <- matrix(sqrt(2) / sqrt(pi), nrow = K, ncol = K)
  beta_raw_init <- array(0, c(J, K, K))
  function(n) list(pi = pi_init, mu = mu_init, sigma = sigma_init,
                   beta_raw = beta_raw_init)
}

dawid_skene_unsure_inits <- function(K, J, I) {
  pi_init <- rep(1/K, K)
  theta_init <- array(0.2 / (K - 1), c(J, K, K))
  for (j in 1:J) {
      diag(theta_init[j, ,]) <- 0.8
  }
  # Haven't really thought about these...
  diff_mean_init = 0.5
  diff_ssize_init = 2
  difficulty_init = rep(0.5, I)
  conf_sigma_init = 1
  confidence_init = rep(0, J)
  delta_init = 0
  function(n) list(theta = theta_init, pi = pi_init, diff_mean = diff_mean_init,
                   diff_ssize = diff_ssize_init, difficulty = difficulty_init,
                   conf_sigma = conf_sigma_init, confidence = confidence_init,
                   delta = delta_init)
}

#' Helper to check if the prior parameters and data have consistent dimensions
#'
#' @param stan_data data in stan format
#' @param model the passed model
#'
#' @return The fully realised prior parameters
#'
#' @noRd
#'
check_K <- function(stan_data, model) {
  # NB: this does not/cannot tell the user which of the pars is inconsistent
  # but we can return a vector (with NULLs and parse cleverly)
  if (!is.null(model$K) && (stan_data$K != model$K)) {
    stop("The number of categories is inconsistent between data and the prior",
         " parameters", call. = FALSE)
  }
}

#' Helper get the correct stan file to run for model/data combination
#'
#' @param data_format A string containing the specification of the data format
#' @param model a rater_model object
#'
#' @return the name (no .stan) of the stan file that should be run
#'
#' @noRd
#'
get_stan_file <- function(data_format, model) {

  file <- get_file(model)
  # If the data is grouped override this. We are assuming we have a
  # valid model/format pair.
  if (data_format == "grouped") {
    file <- "grouped_data"
  }
  file
}

#' Helper to check if the passed model is valid.
#'
#' This function will return a rater_model object if one can be constructed
#' from the input.
#'
#' @param model The `model` argument passed to [`rater()`]
#'
#' @return A rater model object.
#'
#' @noRd
#'
validate_model <- function(model) {

  if (is.character(model)) {
    model <- switch(model,
      "dawid_skene" = dawid_skene(),
      "hier_dawid_skene" = hier_dawid_skene(),
      "class_conditional_dawid_skene" = class_conditional_dawid_skene(),
      "dawid_skene_unsure" = dawid_skene_unsure(),
      stop("Invalid model string specification.", call. = FALSE))
  }

  if (!is.rater_model(model)) {
    stop("`model` must be a rater model object.", call. = FALSE)
  }

  model
}

#' Helper function to check that the `long_data_colnames` is valid.
#'
#' @param long_data_colnames The `long_data_colnames` argument
#'   passed to [rater()]
#' @param data_format The `data` argument passed to [rater()]
#' @noRd
#'
check_long_data_colnames <- function(long_data_colnames, data_format) {

  if (!length(long_data_colnames) == 3L) {
    stop("`long_data_colnames` must be length three.", call. = FALSE)
  }

  if (!is.character(long_data_colnames)) {
    stop("`long_data_colnames` must be a character vector.", call. = FALSE)
  }

  required_names <- c("item", "rater", "rating")
  passed_names <- names(long_data_colnames)
  if (is.null(passed_names) || !all(passed_names %in% required_names)) {
    stop("`long_data_colnames` must have names: `item`, `rater` and `rating`.",
         call. = FALSE)
  }

  default_long_data_colnames <- c(
    item = "item",
    rater = "rater",
    rating = "rating"
  )
  same_as_default <- all(long_data_colnames == default_long_data_colnames)
  if (!same_as_default && data_format != "long") {
    warning("Non-default `long_data_colnames` will be ignored as ",
            "`data_format` is not `'long'`.", call. = FALSE)
  }
}

#' Helper to check if passed data and model are valid and consistent
#'
#' @param data The `data` argument passed to [rater()]
#' @param model The `model` argument passed to [rater()]
#' @param data_format The `data_format` argument passed to [rater()]
#' @param long_data_colnames The `long_data_colnames` argument passed to
#'   [rater()]
#'
#' @return Validated data. This will always be a data.frame with the
#'   appropriate column names for the column names.
#'
#' @noRd
#'
validate_input <- function(data, model, data_format, long_data_colnames) {

  if (data_format == "grouped" & !is.dawid_skene(model)) {
    stop("Grouped data can only be used with the Dawid and Skene model.",
         call. = FALSE)
  }

  validate_data(data, data_format, long_data_colnames)
}

#' Validate the data passed into the rater function
#'
#' @param data The `data` argument passed to [rater()]
#' @param data_format The `data_format` argument passed to [rater()]
#' @param long_data_colnames The `long_data_colnames` argument passed to
#'   [rater()]
#' @return Validated data. This will always be a data.frame with the
#'   appropriate column names for the column names.
#'
#' @noRd
#'
validate_data <- function(data, data_format, long_data_colnames) {

  # TODO: The error message in this function should refer to a vignette
  # or vignette section about data format.

  # Note that this test for allow things like tibbles to be accepted. We
  # next use as.data.frame to standardise the input.
  if (!inherits(data, "data.frame") &&  !inherits(data, "matrix")) {
    stop("`data` must be a data.frame or matrix.", call. = FALSE)
  }
  data <- as.data.frame(data)

  # FIXME We should accept non-numeric data (GitHub issue: #81) but for
  # now we explicitly check that is all columns contain numeric values.
  if (!all(vapply(data, is.numeric, FUN.VALUE = logical(1)))) {
    stop("All columns in `data` must contain only numeric values.",
         call. = FALSE)
  }

  if (data_format == "long") {

    # The data probably isn't in long format.
    if (ncol(data) > 3) {
      # If there is a value greater than 30 then the data probability includes
      # a count/tally as a 30 category rating would be silly.
      if (max(data, na.rm = TRUE) > 30) {
         message("Is your data in grouped format? Consider using `data_format = grouped`.")
      } else {
        # Probably just wide data.
        message("Is your data in wide format? Consider using `data_format = wide`.")
      }
    }

    if (!ncol(data) == 3L) {
      stop("Long format `data` must have exactly three columns.", call. = FALSE)
    }

    if (!(all(long_data_colnames %in% colnames(data)))) {
      stop("Long format `data` must have three columns with names: ",
           paste0(long_data_colnames, collapse = ", "), ".", call. = FALSE)
    }

    # The following are errors about 0 elements. We try to show these errors
    # all at once to prevent undue frustration.
    error_messages <- character(0)
    if (any(data[[long_data_colnames[["item"]]]] == 0)) {
       error_messages <- c(error_messages, paste0(
        "Some item indexes are 0. All indexes must be in 1:I",
        " where I is the number of items."))
    }

    if (any(data[[long_data_colnames[["rater"]]]] == 0)) {
      error_messages <- c(error_messages, paste0(
        "Some rater indexes are 0. All indexes must be in 1:J",
        " where J is the number of raters."))
    }

    if (any(data[[long_data_colnames[["rating"]]]] == 0)) {
      error_messages <- c(error_messages, paste0(
        "Some ratings are 0. All ratings must be in 1:K",
        " where K is the number of classes."))
    }

    if (length(error_messages) > 0) {

      if (length(error_messages) == 1) {
        stop(error_messages[[1]], call. = FALSE)
      } else if (length(error_messages) > 1) {
        stop("\n", paste0("* ", error_messages, collapse = "\n"), call. = FALSE)
      }

    }

  } else if (data_format == "grouped") {

    last_col_name <- colnames(data)[[ncol(data)]]
    if (!last_col_name == "n") {
      stop("The last column must be named `n`.", call. = FALSE)
    }

    error_messages <- character(0)
    tally <- data$n
    if (any(tally == 0)) {
      error_messages <- c(error_messages,
        "All elements of the column `n` must be > 0."
      )
    }

    rest <- data[1:(ncol(data) - 1)]
    if (any(rest == 0)) {
      error_messages <- c(error_messages, paste0(
        "Some ratings are 0. All ratings must be in 1:K",
        " where K is the number of classes."))
    }

    if (length(error_messages) == 1) {
        stop(error_messages[[1]], call. = FALSE)
    }

    if (length(error_messages) == 2) {
        stop("\n", paste0("* ", error_messages, collapse = "\n"), call. = FALSE)
    }

  } else if (data_format == "long_unsure") {

    rating_col_name <- long_data_colnames[["rating"]]
    idx1 <- which(data[[rating_col_name]] != 0)
    data_rating <- data[idx1, , drop = FALSE]
    idx0 <- which(data[[rating_col_name]] == 0)
    data_unsure <- data[idx0, , drop = FALSE]

    data_checked <- rbind(validate_data(data_rating, "long", long_data_colnames),
                          data_unsure)

    data <- data_checked[order(c(idx1, idx0)), , drop = FALSE]

  }

  # Case: wide data
  # We don't need to validate wide data because wide_to_long already validates.

  data
}

check_beta_values <- function(beta) {

  J <- dim(beta)[[1]]
  problems <- logical(J)
  for (j in 1:J) {
    beta_j <- beta[j, , ]
    problems[[j]] <- any(beta_j[row(beta_j) != col(beta_j)] < 1.0)
  }
  off_diag_problem <- any(problems)
  if (off_diag_problem) {
      warning("Optimization may not converge if the off diagonal elements of ",
              "beta are less than 1. Consider changing the prior parameters.",
              call. = FALSE)
  }
}
