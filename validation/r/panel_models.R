# Helpers for independent R parity fits of limiteddepkit panel models.
#
# ordinal::clmm documents negative nAGQ values as non-adaptive
# Gauss-Hermite quadrature.  That is the closest maintained R estimator to
# limiteddepkit's RandomEffectsOrderedLogit.  The finite-node implementation
# in ordinal is not algebraically identical to limiteddepkit's sqrt(2)-scaled
# rule, so callers should describe this as implementation parity rather than
# exact-node identity.

.ldk_panel_assert <- function(condition, message) {
  if (!isTRUE(condition)) {
    stop(message, call. = FALSE)
  }
}

.ldk_panel_scalar_text <- function(value) {
  if (is.null(value) || length(value) == 0L || is.na(value[[1L]])) {
    return(NA_character_)
  }
  as.character(value[[1L]])
}

.ldk_panel_threshold_names <- function(category_order) {
  paste0(
    "threshold: ",
    head(category_order, -1L),
    " | ",
    tail(category_order, -1L)
  )
}

.ldk_panel_fixed_probabilities <- function(fit, data, feature_names) {
  design <- as.matrix(data[, feature_names, drop = FALSE])
  storage.mode(design) <- "double"
  linear_predictor <- drop(design %*% unname(fit$beta))
  thresholds <- unname(fit$alpha)

  cumulative <- outer(
    linear_predictor,
    thresholds,
    function(eta, cutpoint) stats::plogis(cutpoint - eta)
  )
  if (length(thresholds) == 1L) {
    cumulative <- matrix(cumulative, ncol = 1L)
  }
  bounds <- cbind(0, cumulative, 1)
  probabilities <-
    bounds[, -1L, drop = FALSE] - bounds[, -ncol(bounds), drop = FALSE]

  .ldk_panel_assert(
    all(is.finite(probabilities)),
    "Fixed-only panel predictions contain non-finite probabilities."
  )
  .ldk_panel_assert(
    min(probabilities) >= -1e-12 && max(probabilities) <= 1 + 1e-12,
    "Fixed-only panel predictions fall outside the probability bounds."
  )
  .ldk_panel_assert(
    max(abs(rowSums(probabilities) - 1)) <= 1e-12,
    "Fixed-only panel probabilities do not sum to one."
  )
  probabilities
}

#' Fit a canonical random-intercept Ordered Logit parity model in R.
#'
#' Static and dynamic models use the same likelihood here.  A dynamic caller
#' must pass the exact post-initial augmented design exported by
#' DynamicRandomEffectsOrderedLogit; this helper deliberately does not
#' reconstruct lags or initial-condition controls from raw panel rows.
#'
#' @param data Data frame containing the estimation sample.
#' @param model Canonical model identifier written to every returned table.
#' @param dataset Canonical dataset identifier written to every returned table.
#' @param feature_names Fixed-effect column names in `data`, in fit order.
#' @param canonical_feature_names Output names corresponding one-for-one to
#'   `feature_names`.  This maps Stata-safe dynamic aliases such as `state_1`
#'   to toolkit names such as `state[1]`.
#' @param category_order Ordered response levels, in increasing order.
#' @param quadrature_points Number of non-adaptive Gauss-Hermite points.
#' @param prediction_obs_ids Optional unique observation IDs to predict.  The
#'   default predicts every estimation row.
#' @param response_name Response column name.
#' @param entity_name Grouping column name.
#' @param obs_id_name Stable observation-ID column name.
#' @param optimizer_evaluations Maximum outer objective evaluations.
#' @param optimizer_iterations Maximum outer optimizer iterations.
#'
#' @return A list with canonical `estimates`, `covariance`, `fit`, and
#'   fixed-only `predictions` data frames, a one-row `metadata` data frame, the
#'   full raw-to-canonical `jacobian`, and the fitted `clmm` object.
ldk_fit_panel_model <- function(
    data,
    model,
    dataset,
    feature_names,
    canonical_feature_names = feature_names,
    category_order,
    quadrature_points,
    prediction_obs_ids = NULL,
    response_name = "y",
    entity_name = "entity",
    obs_id_name = "obs_id",
    optimizer_evaluations = 5000L,
    optimizer_iterations = 5000L) {
  .ldk_panel_assert(
    requireNamespace("ordinal", quietly = TRUE),
    "The R package 'ordinal' is required for panel parity fits."
  )
  .ldk_panel_assert(is.data.frame(data), "data must be a data frame.")
  .ldk_panel_assert(
    is.character(model) && length(model) == 1L && nzchar(model),
    "model must be one non-empty string."
  )
  .ldk_panel_assert(
    is.character(dataset) && length(dataset) == 1L && nzchar(dataset),
    "dataset must be one non-empty string."
  )
  .ldk_panel_assert(
    is.character(feature_names) && length(feature_names) > 0L &&
      !anyDuplicated(feature_names),
    "feature_names must contain unique column names."
  )
  .ldk_panel_assert(
    is.character(canonical_feature_names) &&
      length(canonical_feature_names) == length(feature_names) &&
      !anyDuplicated(canonical_feature_names),
    paste0(
      "canonical_feature_names must contain one unique name per feature " ,
      "column."
    )
  )
  .ldk_panel_assert(
    length(category_order) >= 2L && !anyNA(category_order) &&
      !anyDuplicated(category_order),
    "category_order must contain at least two unique, non-missing levels."
  )
  .ldk_panel_assert(
    length(quadrature_points) == 1L && is.finite(quadrature_points) &&
      quadrature_points >= 3 && quadrature_points == as.integer(quadrature_points),
    "quadrature_points must be an integer of at least three."
  )
  .ldk_panel_assert(
    length(optimizer_evaluations) == 1L && optimizer_evaluations >= 1,
    "optimizer_evaluations must be positive."
  )
  .ldk_panel_assert(
    length(optimizer_iterations) == 1L && optimizer_iterations >= 1,
    "optimizer_iterations must be positive."
  )

  required_columns <- c(
    response_name,
    entity_name,
    obs_id_name,
    feature_names
  )
  missing_columns <- setdiff(required_columns, names(data))
  .ldk_panel_assert(
    length(missing_columns) == 0L,
    paste0("Panel data are missing columns: ", paste(missing_columns, collapse = ", "))
  )
  .ldk_panel_assert(
    !anyDuplicated(data[[obs_id_name]]) && !anyNA(data[[obs_id_name]]),
    "Observation IDs must be unique and non-missing."
  )
  .ldk_panel_assert(
    !anyNA(data[[entity_name]]),
    "Panel entity labels must be non-missing."
  )
  .ldk_panel_assert(
    all(vapply(data[, feature_names, drop = FALSE], is.numeric, logical(1L))),
    "Every panel feature must be numeric."
  )
  .ldk_panel_assert(
    all(is.finite(as.matrix(data[, feature_names, drop = FALSE]))),
    "Panel features must be finite and non-missing."
  )

  observed_categories <- unique(data[[response_name]])
  .ldk_panel_assert(
    !anyNA(observed_categories) &&
      setequal(as.character(observed_categories), as.character(category_order)),
    "The response support must match category_order exactly."
  )

  work <- data
  work[[response_name]] <- ordered(
    work[[response_name]],
    levels = category_order
  )
  work[[entity_name]] <- factor(work[[entity_name]])
  .ldk_panel_assert(
    nlevels(work[[entity_name]]) >= 2L,
    "Panel parity requires at least two entity groups."
  )

  formula_text <- paste0(
    response_name,
    " ~ ",
    paste(feature_names, collapse = " + "),
    " + (1 | ",
    entity_name,
    ")"
  )
  model_formula <- stats::as.formula(formula_text, env = environment())
  quadrature_points <- as.integer(quadrature_points)
  control <- ordinal::clmm.control(
    method = "nlminb",
    innerCtrl = "giveError",
    checkRanef = "error",
    eval.max = as.integer(optimizer_evaluations),
    iter.max = as.integer(optimizer_iterations)
  )
  fitted <- ordinal::clmm(
    model_formula,
    data = work,
    link = "logit",
    threshold = "flexible",
    Hess = TRUE,
    model = TRUE,
    nAGQ = -quadrature_points,
    control = control
  )

  .ldk_panel_assert(
    identical(as.integer(fitted$nAGQ), -quadrature_points),
    "ordinal::clmm did not retain the requested non-adaptive quadrature rule."
  )
  .ldk_panel_assert(
    identical(names(fitted$beta), feature_names),
    "The fitted R slope order does not match feature_names."
  )
  .ldk_panel_assert(
    length(fitted$alpha) == length(category_order) - 1L,
    "The fitted R threshold count does not match category_order."
  )

  random_variance <- ordinal::VarCorr(fitted)[[1L]]
  sigma_entity <- unname(attr(random_variance, "stddev")[[1L]])
  .ldk_panel_assert(
    length(sigma_entity) == 1L && is.finite(sigma_entity) && sigma_entity > 0,
    "The fitted R random-intercept standard deviation is invalid."
  )

  raw_covariance <- stats::vcov(fitted)
  .ldk_panel_assert(
    is.matrix(raw_covariance) && nrow(raw_covariance) == ncol(raw_covariance) &&
      all(is.finite(raw_covariance)),
    "The fitted R covariance matrix is missing or non-finite."
  )
  raw_threshold_names <- names(fitted$alpha)
  raw_slope_names <- names(fitted$beta)
  raw_tau_names <- setdiff(
    rownames(raw_covariance),
    c(raw_threshold_names, raw_slope_names)
  )
  .ldk_panel_assert(
    length(raw_tau_names) == 1L,
    "Expected exactly one log-standard-deviation covariance parameter."
  )

  threshold_names <- .ldk_panel_threshold_names(category_order)
  canonical_names <- c(
    canonical_feature_names,
    threshold_names,
    "sigma_entity"
  )
  .ldk_panel_assert(
    !anyDuplicated(canonical_names),
    "Canonical panel parameter names must be unique."
  )

  # vcov(clmm) is ordered as (alpha, beta, tau), where tau = log(sigma).
  # This full Jacobian simultaneously reorders alpha/beta and applies
  # d exp(tau) / d tau = sigma, preserving every cross-covariance.
  jacobian <- matrix(
    0,
    nrow = length(canonical_names),
    ncol = nrow(raw_covariance),
    dimnames = list(canonical_names, rownames(raw_covariance))
  )
  for (index in seq_along(feature_names)) {
    jacobian[canonical_feature_names[[index]], raw_slope_names[[index]]] <- 1
  }
  for (index in seq_along(threshold_names)) {
    jacobian[threshold_names[[index]], raw_threshold_names[[index]]] <- 1
  }
  jacobian["sigma_entity", raw_tau_names] <- sigma_entity

  canonical_covariance <- jacobian %*% raw_covariance %*% t(jacobian)
  canonical_covariance <-
    (canonical_covariance + t(canonical_covariance)) / 2
  covariance_diagonal <- diag(canonical_covariance)
  covariance_finite <- all(is.finite(canonical_covariance))
  covariance_nonnegative <- all(covariance_diagonal >= -1e-12)
  standard_errors <- sqrt(pmax(covariance_diagonal, 0))

  estimates_vector <- c(
    stats::setNames(unname(fitted$beta), canonical_feature_names),
    stats::setNames(unname(fitted$alpha), threshold_names),
    sigma_entity = sigma_entity
  )
  estimates <- data.frame(
    model = model,
    dataset = dataset,
    parameter = canonical_names,
    estimate = unname(estimates_vector[canonical_names]),
    standard_error = unname(standard_errors[canonical_names]),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  n_canonical_parameters <- length(canonical_names)
  covariance <- data.frame(
    model = rep(model, n_canonical_parameters * n_canonical_parameters),
    dataset = rep(dataset, n_canonical_parameters * n_canonical_parameters),
    row_parameter = rep(canonical_names, each = n_canonical_parameters),
    column_parameter = rep(canonical_names, times = n_canonical_parameters),
    covariance = as.vector(t(canonical_covariance)),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  if (is.null(prediction_obs_ids)) {
    prediction_rows <- seq_len(nrow(work))
  } else {
    .ldk_panel_assert(
      !anyNA(prediction_obs_ids) && !anyDuplicated(prediction_obs_ids),
      "prediction_obs_ids must be unique and non-missing."
    )
    prediction_rows <- match(prediction_obs_ids, work[[obs_id_name]])
    .ldk_panel_assert(
      !anyNA(prediction_rows),
      "Some prediction_obs_ids are absent from the panel estimation sample."
    )
  }
  prediction_data <- work[prediction_rows, , drop = FALSE]
  probability_matrix <- .ldk_panel_fixed_probabilities(
    fitted,
    prediction_data,
    feature_names
  )
  predictions <- data.frame(
    model = model,
    dataset = dataset,
    obs_id = rep(prediction_data[[obs_id_name]], each = length(category_order)),
    category = rep(category_order, times = nrow(prediction_data)),
    probability = as.vector(t(probability_matrix)),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  log_likelihood_object <- stats::logLik(fitted)
  log_likelihood <- as.numeric(log_likelihood_object)
  n_parameters <- as.integer(attr(log_likelihood_object, "df"))
  n_observations <- as.integer(stats::nobs(fitted))
  n_groups <- as.integer(nlevels(work[[entity_name]]))
  aic <- -2 * log_likelihood + 2 * n_parameters
  bic <- -2 * log_likelihood + log(n_observations) * n_parameters

  convergence_code <- as.integer(fitted$optRes$convergence)
  .ldk_panel_assert(
    length(convergence_code) == 1L && !is.na(convergence_code),
    "ordinal::clmm did not expose one optimizer convergence code."
  )
  convergence_message <- .ldk_panel_scalar_text(fitted$optRes$message)
  gradient <- as.numeric(fitted$gradient)
  gradient_finite <- length(gradient) > 0L && all(is.finite(gradient))
  max_abs_gradient <-
    if (gradient_finite) {
      max(abs(gradient))
    } else {
      NA_real_
    }
  hessian <- as.matrix(fitted$Hessian)
  hessian_finite <- length(hessian) > 0L && all(is.finite(hessian))
  hessian_eigenvalues <-
    if (hessian_finite) {
      eigen(
        (hessian + t(hessian)) / 2,
        symmetric = TRUE,
        only.values = TRUE
      )$values
    } else {
      NA_real_
    }
  min_hessian_eigenvalue <-
    if (all(is.finite(hessian_eigenvalues))) {
      min(hessian_eigenvalues)
    } else {
      NA_real_
    }
  hessian_positive_definite <-
    is.finite(min_hessian_eigenvalue) && min_hessian_eigenvalue > 0
  converged <-
    isTRUE(convergence_code == 0L) && gradient_finite &&
    is.finite(log_likelihood) &&
    all(is.finite(estimates$estimate))
  inference_valid <-
    converged && covariance_finite && covariance_nonnegative &&
    all(is.finite(standard_errors)) && hessian_positive_definite

  fit <- data.frame(
    model = model,
    dataset = dataset,
    nobs = n_observations,
    n_groups = n_groups,
    n_params = n_parameters,
    loglike = log_likelihood,
    aic = aic,
    bic = bic,
    converged = converged,
    inference_valid = inference_valid,
    quadrature_points = quadrature_points,
    constraint_slack = NA_real_,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  metadata <- data.frame(
    model = model,
    dataset = dataset,
    engine = "ordinal::clmm",
    r_version = as.character(getRversion()),
    package_version = as.character(utils::packageVersion("ordinal")),
    optimizer = "nlminb",
    convergence_code = convergence_code,
    convergence_message = convergence_message,
    gradient_finite = gradient_finite,
    max_abs_gradient = max_abs_gradient,
    min_hessian_eigenvalue = min_hessian_eigenvalue,
    hessian_positive_definite = hessian_positive_definite,
    quadrature_method = "nonadaptive-gauss-hermite",
    quadrature_points = quadrature_points,
    nAGQ = -quadrature_points,
    covariance_input_scale = "thresholds, slopes, log(random-effect SD)",
    covariance_output_scale = "slopes, thresholds, random-effect SD",
    prediction_conditioning = "fixed-only: random intercept = 0",
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  list(
    estimates = estimates,
    covariance = covariance,
    fit = fit,
    predictions = predictions,
    metadata = metadata,
    jacobian = jacobian,
    object = fitted
  )
}
