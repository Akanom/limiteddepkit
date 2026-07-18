#!/usr/bin/env Rscript

# Independent R application suite for the promoted limiteddepkit families.
#
# Usage from the repository root:
#   Rscript --vanilla validation/promoted/run_parity.R validation/promoted/work

options(digits = 17, OutDec = ".", warn = 1)

stopf <- function(format, ...) stop(sprintf(format, ...), call. = FALSE)

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1L) {
  stopf("Usage: Rscript --vanilla validation/promoted/run_parity.R <prepared-work-directory>")
}
work_directory <- normalizePath(arguments[[1L]], winslash = "/", mustWork = TRUE)
output_directory <- file.path(work_directory, "r")
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

maintained_outputs <- file.path(
  output_directory,
  c(
    "estimates.csv", "covariance.csv", "fit.csv", "predictions.csv", "metadata.csv",
    "comparison_report.csv", "comparison_summary.md", "parity_certificate.json"
  )
)
for (artifact in maintained_outputs[file.exists(maintained_outputs)]) {
  status <- unlink(artifact, recursive = FALSE, force = FALSE)
  if (!identical(status, 0L) || file.exists(artifact)) {
    stopf("Could not invalidate stale promoted R artifact: %s", artifact)
  }
}

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(file_argument) != 1L) stopf("Cannot determine run_parity.R location.")
script_path <- normalizePath(sub("^--file=", "", file_argument), winslash = "/", mustWork = TRUE)
script_directory <- dirname(script_path)
legacy_library <- file.path(dirname(script_directory), "r", "work", "library")
if (dir.exists(legacy_library)) .libPaths(c(legacy_library, .libPaths()))

require_package <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stopf("Required R package %s is not installed.", sQuote(package))
  }
}
for (package in c("MASS", "survival", "numDeriv", "ordinal")) require_package(package)
# clogit constructs and evaluates an internal coxph call in the caller's
# environment, so attach survival after the explicit availability check.
suppressPackageStartupMessages(library(survival))

read_fixture <- function(dataset) {
  path <- file.path(work_directory, "data", paste0(dataset, ".csv"))
  if (!file.exists(path)) stopf("Prepared data file is missing: %s", path)
  data <- utils::read.csv(
    path, check.names = FALSE, stringsAsFactors = FALSE,
    na.strings = c("", "NA", "NaN")
  )
  if (!"obs_id" %in% names(data) || anyDuplicated(data$obs_id)) {
    stopf("%s must contain unique obs_id values.", dataset)
  }
  data
}

require_columns <- function(data, columns, context) {
  missing <- setdiff(columns, names(data))
  if (length(missing)) stopf("%s is missing columns: %s", context, paste(missing, collapse = ", "))
}

design_matrix <- function(data, features, context) {
  require_columns(data, features, context)
  X <- as.matrix(data[, features, drop = FALSE])
  storage.mode(X) <- "double"
  if (any(!is.finite(X))) stopf("%s contains non-finite design values.", context)
  if (qr(X)$rank != ncol(X)) stopf("%s design matrix is rank deficient.", context)
  colnames(X) <- features
  X
}

safe_inverse <- function(information, context, tolerance = 1e-10) {
  information <- 0.5 * (information + t(information))
  eigenvalues <- eigen(information, symmetric = TRUE, only.values = TRUE)$values
  if (any(!is.finite(eigenvalues)) || min(eigenvalues) <= tolerance * max(eigenvalues)) {
    stopf("%s observed-information matrix is not positive definite.", context)
  }
  covariance <- solve(information)
  0.5 * (covariance + t(covariance))
}

parameter_frames <- function(model, dataset, parameters, covariance) {
  parameter_names <- names(parameters)
  if (is.null(parameter_names) || anyDuplicated(parameter_names)) {
    stopf("%s produced missing or duplicate parameter names.", model)
  }
  if (!identical(dim(covariance), c(length(parameters), length(parameters)))) {
    stopf("%s covariance dimensions do not match parameters.", model)
  }
  dimnames(covariance) <- list(parameter_names, parameter_names)
  standard_errors <- sqrt(diag(covariance))
  if (any(!is.finite(parameters)) || any(!is.finite(covariance)) ||
      any(!is.finite(standard_errors))) {
    stopf("%s produced non-finite estimates or covariance.", model)
  }
  estimates <- data.frame(
    model = model, dataset = dataset, parameter = parameter_names,
    estimate = unname(parameters), standard_error = unname(standard_errors),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  p <- length(parameters)
  covariance_frame <- data.frame(
    model = rep(model, p * p), dataset = rep(dataset, p * p),
    row_parameter = rep(parameter_names, each = p),
    column_parameter = rep(parameter_names, times = p),
    covariance = as.vector(t(covariance)), stringsAsFactors = FALSE,
    check.names = FALSE
  )
  list(estimates = estimates, covariance = covariance_frame)
}

fit_frame <- function(
    model, dataset, nobs, n_params, loglike, converged = TRUE,
    inference_valid = TRUE, n_groups = NA_integer_, n_events = NA_integer_,
    n_contributing_entities = NA_integer_, n_cutoff_clones = NA_integer_,
    n_pseudo_observations = NA_integer_,
    n_censored = NA_integer_, n_interval = NA_integer_, n_exact = NA_integer_,
    n_left_censored = NA_integer_, n_right_censored = NA_integer_,
    score_norm = NA_real_, scaled_score_norm = NA_real_, backend,
    covariance_type, information_criteria = TRUE, penalized_loglike = NA_real_,
    jeffreys_penalty = NA_real_) {
  data.frame(
    model = model, dataset = dataset, nobs = as.integer(nobs),
    n_params = as.integer(n_params), loglike = as.numeric(loglike),
    aic = if (information_criteria) -2 * loglike + 2 * n_params else NA_real_,
    bic = if (information_criteria) -2 * loglike + log(nobs) * n_params else NA_real_,
    converged = isTRUE(converged), inference_valid = isTRUE(inference_valid),
    n_groups = as.integer(n_groups),
    n_contributing_entities = as.integer(n_contributing_entities),
    n_cutoff_clones = as.integer(n_cutoff_clones),
    n_pseudo_observations = as.integer(n_pseudo_observations),
    n_events = as.integer(n_events),
    n_censored = as.integer(n_censored), n_interval = as.integer(n_interval),
    n_exact = as.integer(n_exact), n_left_censored = as.integer(n_left_censored),
    n_right_censored = as.integer(n_right_censored), score_norm = as.numeric(score_norm),
    scaled_score_norm = as.numeric(scaled_score_norm),
    penalized_loglike = as.numeric(penalized_loglike),
    jeffreys_penalty = as.numeric(jeffreys_penalty), backend = backend,
    covariance_type = covariance_type, stringsAsFactors = FALSE, check.names = FALSE
  )
}

model_specs <- list(
  firth_binary_logit = list(dataset = "binary_lbw", features = c("intercept", "x1", "x2", "x3", "x4")),
  poisson = list(dataset = "count_rod93", features = c("intercept", "log_age", "cohort_2", "cohort_3")),
  negative_binomial_nb2 = list(dataset = "count_rod93", features = c("intercept", "log_age", "cohort_2", "cohort_3")),
  tobit = list(dataset = "censoring_mroz87", features = c("intercept", "age_z", "education_z", "experience_z", "young_children")),
  truncated_regression = list(dataset = "censoring_mroz87", features = c("intercept", "age_z", "education_z", "experience_z", "young_children")),
  interval_regression = list(dataset = "censoring_womenwage2", features = c("intercept", "age_z", "school_z", "tenure_z", "never_married", "rural")),
  geometric_duration = list(dataset = "duration_cancer", features = c("intercept", "age_z", "drug_2", "drug_3")),
  exponential_duration = list(dataset = "duration_cancer", features = c("intercept", "age_z", "drug_2", "drug_3")),
  weibull_duration = list(dataset = "duration_cancer", features = c("intercept", "age_z", "drug_2", "drug_3")),
  gamma_duration = list(dataset = "duration_cancer_deaths", features = c("intercept", "age_z", "drug_2", "drug_3")),
  random_effects_ordered_probit = list(dataset = "ordinal_tvsfpors", features = c("x1", "x2", "x3", "x4")),
  fixed_effects_ordered_logit = list(dataset = "panel_nlswork", features = c("x1"))
)

dataset_names <- unname(unique(vapply(model_specs, `[[`, character(1), "dataset")))
datasets <- stats::setNames(lapply(dataset_names, read_fixture), dataset_names)

firth_components <- function(X, y, beta) {
  eta <- drop(X %*% beta)
  probability <- stats::plogis(eta)
  weights <- probability * (1 - probability)
  information <- crossprod(X, X * weights)
  determinant <- determinant(information, logarithm = TRUE)
  if (determinant$sign <= 0 || !is.finite(determinant$modulus)) {
    stopf("Firth information became singular.")
  }
  weighted_design <- X * sqrt(weights)
  leverage <- rowSums((weighted_design %*% solve(information)) * weighted_design)
  leverage <- pmin(pmax(leverage, 0), 1)
  adjusted_score <- drop(crossprod(X, y - probability + leverage * (0.5 - probability)))
  loglike <- sum(y * eta - pmax(eta, 0) - log1p(exp(-abs(eta))))
  list(
    penalized_loglike = loglike + 0.5 * as.numeric(determinant$modulus),
    loglike = loglike, adjusted_score = adjusted_score,
    information = information, probability = probability
  )
}

fit_firth <- function() {
  model <- "firth_binary_logit"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("y", spec$features), model)
  X <- design_matrix(data, spec$features, model); y <- as.numeric(data$y)
  if (!identical(sort(unique(y)), c(0, 1))) stopf("Firth outcome must be binary.")
  beta <- rep(0, ncol(X)); components <- firth_components(X, y, beta)
  iterations <- 0L; halvings <- 0L
  for (iteration in seq_len(1000L)) {
    score_norm <- max(abs(components$adjusted_score))
    if (score_norm <= 1e-7) break
    step <- drop(solve(components$information, components$adjusted_score))
    if (max(abs(step)) > 5) step <- step * 5 / max(abs(step))
    scale <- 1
    accepted <- FALSE
    for (halving in 0:30) {
      candidate <- tryCatch(firth_components(X, y, beta + scale * step), error = function(e) NULL)
      if (!is.null(candidate) && is.finite(candidate$penalized_loglike) &&
          candidate$penalized_loglike >= components$penalized_loglike - 1e-12) {
        accepted <- TRUE; break
      }
      scale <- scale / 2; halvings <- halvings + 1L
    }
    if (!accepted) stopf("Independent Firth step halving failed.")
    beta <- beta + scale * step; components <- candidate; iterations <- iteration
  }
  score_norm <- max(abs(components$adjusted_score))
  if (score_norm > 1e-7) stopf("Independent Firth adjusted score did not converge: %.6g", score_norm)
  names(beta) <- spec$features
  covariance <- safe_inverse(components$information, model)
  list(
    model = model, dataset = spec$dataset, parameters = beta, covariance = covariance,
    loglike = components$loglike, data = data, X = X, probability = components$probability,
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(beta), components$loglike,
      score_norm = score_norm, scaled_score_norm = score_norm / nrow(X),
      backend = "independent-firth-adjusted-score",
      covariance_type = "inverse-ordinary-fisher-at-bias-reduced-estimate",
      information_criteria = FALSE,
      penalized_loglike = components$penalized_loglike,
      jeffreys_penalty = components$penalized_loglike - components$loglike
    ), details = sprintf("iterations=%d; step_halvings=%d", iterations, halvings)
  )
}

fit_poisson <- function() {
  model <- "poisson"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("deaths", "exposure", spec$features), model)
  X <- design_matrix(data, spec$features, model); y <- as.numeric(data$deaths)
  exposure <- as.numeric(data$exposure)
  fitted <- stats::glm.fit(
    X, y, offset = log(exposure), family = stats::poisson(),
    control = stats::glm.control(epsilon = 1e-12, maxit = 1000L)
  )
  if (!isTRUE(fitted$converged) || fitted$rank != ncol(X)) stopf("Poisson did not converge.")
  beta <- stats::setNames(as.numeric(fitted$coefficients), spec$features)
  mean <- exposure * exp(drop(X %*% beta))
  information <- crossprod(X, X * mean)
  covariance <- safe_inverse(information, model)
  score <- drop(crossprod(X, y - mean)); loglike <- sum(stats::dpois(y, mean, log = TRUE))
  list(
    model = model, dataset = spec$dataset, parameters = beta, covariance = covariance,
    loglike = loglike, data = data, X = X, mean = mean,
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(beta), loglike,
      score_norm = max(abs(score)), scaled_score_norm = max(abs(score)) / nrow(X),
      backend = "stats::glm", covariance_type = "observed-information"
    ), details = "stats::glm Poisson with log(exposure) offset"
  )
}

nb2_loglike <- function(parameters, X, y, exposure) {
  p <- ncol(X); beta <- parameters[seq_len(p)]; log_alpha <- parameters[p + 1L]
  size <- exp(-log_alpha); mean <- exposure * exp(drop(X %*% beta))
  sum(stats::dnbinom(y, size = size, mu = mean, log = TRUE))
}

fit_nb2 <- function() {
  model <- "negative_binomial_nb2"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("deaths", "exposure", spec$features), model)
  X <- design_matrix(data, spec$features, model); y <- as.numeric(data$deaths)
  exposure <- as.numeric(data$exposure)
  formula <- stats::as.formula(paste("deaths ~ 0 +", paste(spec$features, collapse = " + "), "+ offset(log(exposure))"))
  fitted <- MASS::glm.nb(
    formula, data = data, control = stats::glm.control(epsilon = 1e-12, maxit = 1000L),
    link = log, init.theta = 1
  )
  beta <- stats::setNames(as.numeric(stats::coef(fitted)), spec$features)
  parameters <- c(beta, log_alpha = -log(fitted$theta))
  objective <- function(value) -nb2_loglike(value, X, y, exposure)
  information <- numDeriv::hessian(objective, parameters, method = "Richardson")
  covariance <- safe_inverse(information, model, tolerance = 1e-12)
  loglike <- nb2_loglike(parameters, X, y, exposure)
  score <- numDeriv::grad(objective, parameters, method = "Richardson")
  list(
    model = model, dataset = spec$dataset, parameters = parameters, covariance = covariance,
    loglike = loglike, data = data, X = X,
    mean = exposure * exp(drop(X %*% beta)),
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(parameters), loglike,
      score_norm = max(abs(score)), scaled_score_norm = max(abs(score)) / nrow(X),
      backend = "MASS::glm.nb", covariance_type = "full-observed-information-beta-log_alpha"
    ), details = "glm.nb theta mapped as log_alpha=-log(theta); full numerical Hessian"
  )
}

survreg_canonical <- function(fitted, features, scale_name, scale_transform, context) {
  raw_parameters <- c(stats::coef(fitted), log_scale = log(fitted$scale))
  raw_covariance <- as.matrix(fitted$var)
  if (!identical(dim(raw_covariance), c(length(raw_parameters), length(raw_parameters)))) {
    stopf("%s survreg covariance does not include Log(scale).", context)
  }
  parameters <- c(stats::setNames(as.numeric(stats::coef(fitted)), features), scale_transform$value(fitted$scale))
  names(parameters)[length(parameters)] <- scale_name
  jacobian <- diag(length(parameters)); jacobian[length(parameters), length(parameters)] <- scale_transform$derivative(fitted$scale)
  covariance <- jacobian %*% raw_covariance %*% t(jacobian)
  list(parameters = parameters, covariance = 0.5 * (covariance + t(covariance)))
}

fit_tobit <- function() {
  model <- "tobit"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("y", "positive", spec$features), model)
  X <- design_matrix(data, spec$features, model)
  response <- survival::Surv(as.numeric(data$y), as.integer(data$positive), type = "left")
  formula <- stats::as.formula(paste("response ~ 0 +", paste(spec$features, collapse = " + ")))
  fitted <- survival::survreg(formula, data = data, dist = "gaussian", control = survival::survreg.control(maxiter = 1000, rel.tolerance = 1e-12))
  canonical <- survreg_canonical(
    fitted, spec$features, "sigma",
    list(value = function(scale) scale, derivative = function(scale) scale), model
  )
  mu <- drop(X %*% canonical$parameters[spec$features]); sigma <- unname(canonical$parameters["sigma"])
  exact <- as.integer(data$positive) == 1L
  loglike <- sum(ifelse(exact, stats::dnorm(data$y, mu, sigma, log = TRUE), stats::pnorm((data$y - mu) / sigma, log.p = TRUE)))
  list(
    model = model, dataset = spec$dataset, parameters = canonical$parameters,
    covariance = canonical$covariance, loglike = loglike, data = data, X = X,
    linear_mean = mu, sigma = sigma,
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(canonical$parameters), loglike,
      n_censored = sum(!exact),
      backend = "survival::survreg-gaussian", covariance_type = "observed-information-full-jacobian"
    ), details = "survreg Gaussian AFT; raw Log(scale) mapped to sigma"
  )
}

fit_truncated <- function() {
  model <- "truncated_regression"; spec <- model_specs[[model]]; full <- datasets[[spec$dataset]]
  require_columns(full, c("y", "positive", spec$features), model)
  data <- full[as.integer(full$positive) == 1L, , drop = FALSE]
  X <- design_matrix(data, spec$features, model); y <- as.numeric(data$y); lower <- 0
  initial_beta <- stats::coef(stats::lm.fit(X, y)); initial_sigma <- sqrt(mean((y - drop(X %*% initial_beta))^2))
  objective <- function(parameters) {
    beta <- parameters[seq_len(ncol(X))]; sigma <- exp(parameters[ncol(X) + 1L]); mu <- drop(X %*% beta)
    -sum(stats::dnorm(y, mu, sigma, log = TRUE) - stats::pnorm((lower - mu) / sigma, lower.tail = FALSE, log.p = TRUE))
  }
  initial <- c(initial_beta, log(initial_sigma))
  fitted <- stats::optim(initial, objective, method = "BFGS", control = list(maxit = 5000, reltol = 1e-12))
  refined <- stats::optim(fitted$par, objective, method = "BFGS", control = list(maxit = 5000, reltol = 1e-14))
  raw <- refined$par; raw_information <- numDeriv::hessian(objective, raw, method = "Richardson")
  raw_covariance <- safe_inverse(raw_information, model, tolerance = 1e-12)
  sigma <- exp(raw[length(raw)]); parameters <- c(stats::setNames(raw[-length(raw)], spec$features), sigma = sigma)
  jacobian <- diag(length(raw)); jacobian[length(raw), length(raw)] <- sigma
  covariance <- jacobian %*% raw_covariance %*% t(jacobian)
  raw_score <- numDeriv::grad(objective, raw, method = "Richardson")
  list(
    model = model, dataset = spec$dataset, parameters = parameters, covariance = covariance,
    loglike = -objective(raw), data = data, X = X, linear_mean = drop(X %*% parameters[spec$features]), sigma = sigma,
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(parameters), -objective(raw),
      score_norm = max(abs(raw_score)), scaled_score_norm = max(abs(raw_score)) / nrow(X),
      backend = "independent-exact-truncated-normal-mle", covariance_type = "observed-information-full-jacobian"
    ), details = "Exact left-truncated normal likelihood at lower=0; no truncreg fallback"
  )
}

fit_interval <- function() {
  model <- "interval_regression"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("lower", "upper", spec$features), model)
  data$lower[is.na(data$lower)] <- -Inf; data$upper[is.na(data$upper)] <- Inf
  X <- design_matrix(data, spec$features, model)
  response <- survival::Surv(data$lower, data$upper, type = "interval2")
  formula <- stats::as.formula(paste("response ~ 0 +", paste(spec$features, collapse = " + ")))
  fitted <- survival::survreg(formula, data = data, dist = "gaussian", control = survival::survreg.control(maxiter = 1000, rel.tolerance = 1e-12))
  canonical <- survreg_canonical(
    fitted, spec$features, "sigma",
    list(value = function(scale) scale, derivative = function(scale) scale), model
  )
  mu <- drop(X %*% canonical$parameters[spec$features]); sigma <- unname(canonical$parameters["sigma"])
  exact <- is.finite(data$lower) & is.finite(data$upper) & abs(data$upper - data$lower) <= 1e-12
  left <- !is.finite(data$lower) & is.finite(data$upper)
  right <- is.finite(data$lower) & !is.finite(data$upper)
  interval <- is.finite(data$lower) & is.finite(data$upper) & !exact
  contribution <- numeric(nrow(data))
  contribution[exact] <- stats::dnorm(data$lower[exact], mu[exact], sigma, log = TRUE)
  contribution[left] <- stats::pnorm((data$upper[left] - mu[left]) / sigma, log.p = TRUE)
  contribution[right] <- stats::pnorm((data$lower[right] - mu[right]) / sigma, lower.tail = FALSE, log.p = TRUE)
  if (any(interval)) {
    pu <- stats::pnorm((data$upper[interval] - mu[interval]) / sigma)
    pl <- stats::pnorm((data$lower[interval] - mu[interval]) / sigma)
    contribution[interval] <- log(pu - pl)
  }
  loglike <- sum(contribution)
  list(
    model = model, dataset = spec$dataset, parameters = canonical$parameters,
    covariance = canonical$covariance, loglike = loglike, data = data, X = X,
    linear_mean = mu, sigma = sigma,
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(canonical$parameters), loglike,
      n_interval = sum(interval), n_exact = sum(exact),
      n_left_censored = sum(left), n_right_censored = sum(right),
      backend = "survival::survreg-gaussian-interval2", covariance_type = "observed-information-full-jacobian"
    ), details = "survreg Gaussian interval2; open endpoints restored to infinities"
  )
}

fit_geometric <- function() {
  model <- "geometric_duration"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("duration", "event", spec$features), model)
  X <- design_matrix(data, spec$features, model); duration <- as.integer(data$duration); event <- as.integer(data$event)
  fitted <- stats::glm.fit(
    X, cbind(event, duration - event), family = stats::binomial(),
    control = stats::glm.control(epsilon = 1e-12, maxit = 1000L)
  )
  if (!isTRUE(fitted$converged)) stopf("Grouped-binomial geometric identity did not converge.")
  beta <- stats::setNames(as.numeric(fitted$coefficients), spec$features)
  hazard <- stats::plogis(drop(X %*% beta))
  loglike <- sum(event * log(hazard) + (duration - event) * log1p(-hazard))
  information <- crossprod(X, X * (duration * hazard * (1 - hazard)))
  covariance <- safe_inverse(information, model)
  score <- drop(crossprod(X, event - duration * hazard))
  list(
    model = model, dataset = spec$dataset, parameters = beta, covariance = covariance,
    loglike = loglike, data = data, X = X, hazard = hazard,
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(beta), loglike,
      n_events = sum(event), score_norm = max(abs(score)),
      scaled_score_norm = max(abs(score)) / nrow(X), backend = "stats::glm-grouped-binomial-identity",
      covariance_type = "observed-information"
    ), details = "Grouped binomial-logit score identity; geometric likelihood excludes binomial constants"
  )
}

fit_exponential <- function() {
  model <- "exponential_duration"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("duration", "event", spec$features), model)
  X <- design_matrix(data, spec$features, model); duration <- as.numeric(data$duration); event <- as.integer(data$event)
  response <- survival::Surv(duration, event, type = "right")
  formula <- stats::as.formula(paste("response ~ 0 +", paste(spec$features, collapse = " + ")))
  fitted <- survival::survreg(formula, data = data, dist = "exponential", control = survival::survreg.control(maxiter = 1000, rel.tolerance = 1e-12))
  beta <- stats::setNames(as.numeric(stats::coef(fitted)), spec$features); covariance <- as.matrix(fitted$var)
  scale <- exp(drop(X %*% beta)); loglike <- sum(event * (-log(scale)) - duration / scale)
  list(
    model = model, dataset = spec$dataset, parameters = beta, covariance = covariance,
    loglike = loglike, data = data, X = X, scale = scale,
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(beta), loglike, n_events = sum(event),
      backend = "survival::survreg-exponential",
      covariance_type = "observed-information"
    ), details = "survreg exponential AFT; exp(X beta) is the toolkit duration scale"
  )
}

fit_weibull <- function() {
  model <- "weibull_duration"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("duration", "event", spec$features), model)
  X <- design_matrix(data, spec$features, model); duration <- as.numeric(data$duration); event <- as.integer(data$event)
  response <- survival::Surv(duration, event, type = "right")
  formula <- stats::as.formula(paste("response ~ 0 +", paste(spec$features, collapse = " + ")))
  fitted <- survival::survreg(formula, data = data, dist = "weibull", control = survival::survreg.control(maxiter = 1000, rel.tolerance = 1e-12))
  canonical <- survreg_canonical(
    fitted, spec$features, "log_alpha",
    list(value = function(scale) -log(scale), derivative = function(scale) -1), model
  )
  beta <- canonical$parameters[spec$features]; alpha <- exp(canonical$parameters["log_alpha"]); scale <- exp(drop(X %*% beta))
  cumulative <- (duration / scale)^alpha
  loglike <- sum(event * (log(alpha) + (alpha - 1) * log(duration) - alpha * log(scale)) - cumulative)
  list(
    model = model, dataset = spec$dataset, parameters = canonical$parameters,
    covariance = canonical$covariance, loglike = loglike, data = data, X = X,
    scale = scale, alpha = unname(alpha),
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(canonical$parameters), loglike,
      n_events = sum(event), backend = "survival::survreg-weibull",
      covariance_type = "observed-information-full-jacobian"
    ), details = "survreg Weibull AFT; log_alpha=-Log(scale) with full covariance Jacobian"
  )
}

gamma_loglike <- function(parameters, X, duration) {
  p <- ncol(X); beta <- parameters[seq_len(p)]; k <- exp(parameters[p + 1L]); scale <- exp(drop(X %*% beta))
  sum((k - 1) * log(duration) - duration / scale - k * log(scale) - lgamma(k))
}

fit_gamma <- function() {
  model <- "gamma_duration"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("duration", "event", spec$features), model)
  if (any(as.integer(data$event) != 1L)) stopf("Gamma application dataset must be uncensored.")
  X <- design_matrix(data, spec$features, model); duration <- as.numeric(data$duration)
  initial_beta <- stats::coef(stats::lm.fit(X, log(duration)))
  initial <- c(initial_beta, log_k = 0)
  objective <- function(value) -gamma_loglike(value, X, duration)
  fitted <- stats::optim(initial, objective, method = "L-BFGS-B", lower = c(rep(-Inf, ncol(X)), -10), upper = c(rep(Inf, ncol(X)), 10), control = list(maxit = 5000, factr = 1))
  refined <- stats::optim(fitted$par, objective, method = "BFGS", control = list(maxit = 5000, reltol = 1e-14))
  parameters <- refined$par; names(parameters) <- c(spec$features, "log_k")
  information <- numDeriv::hessian(objective, parameters, method = "Richardson")
  covariance <- safe_inverse(information, model, tolerance = 1e-12)
  score <- numDeriv::grad(objective, parameters, method = "Richardson")
  k <- exp(parameters["log_k"]); scale <- exp(drop(X %*% parameters[spec$features])); loglike <- -objective(parameters)
  list(
    model = model, dataset = spec$dataset, parameters = parameters, covariance = covariance,
    loglike = loglike, data = data, X = X, scale = scale, k = unname(k),
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(parameters), loglike, n_events = nrow(X),
      score_norm = max(abs(score)), scaled_score_norm = max(abs(score)) / nrow(X),
      backend = "independent-exact-uncensored-gamma-mle", covariance_type = "observed-information"
    ), details = "Independent exact Gamma shape-scale MLE on the uncensored deaths-only application"
  )
}

threshold_labels <- function(categories) paste0("threshold: ", head(categories, -1L), " | ", tail(categories, -1L))

hermite_rule <- function(n) {
  if (n < 1L) stopf("Hermite order must be positive.")
  if (n == 1L) return(list(nodes = 0, weights = sqrt(pi)))
  off_diagonal <- sqrt((1:(n - 1L)) / 2)
  jacobi <- matrix(0, n, n)
  jacobi[cbind(1:(n - 1L), 2:n)] <- off_diagonal
  jacobi[cbind(2:n, 1:(n - 1L))] <- off_diagonal
  decomposition <- eigen(jacobi, symmetric = TRUE)
  ordering <- order(decomposition$values)
  list(
    nodes = decomposition$values[ordering],
    weights = sqrt(pi) * decomposition$vectors[1L, ordering]^2
  )
}

fit_re_ordered_probit <- function() {
  model <- "random_effects_ordered_probit"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("y", "entity", spec$features), model)
  X <- design_matrix(data, spec$features, model); categories <- sort(unique(as.integer(data$y)))
  data$.ordered_y <- ordered(as.integer(data$y), levels = categories)
  formula <- stats::as.formula(paste(".ordered_y ~", paste(spec$features, collapse = " + "), "+ (1 | entity)"))
  fitted <- ordinal::clmm(
    formula, data = data, link = "probit", Hess = TRUE, nAGQ = -20L,
    control = ordinal::clmm.control(maxIter = 1000L, gradTol = 1e-7, maxLineIter = 100L)
  )
  fixed_parameters <- stats::coef(fitted)
  raw_parameters <- c(fixed_parameters, ST1 = unname(fitted$optRes$par[length(fitted$optRes$par)]))
  raw_covariance <- as.matrix(stats::vcov(fitted))
  threshold_count <- length(categories) - 1L
  if (length(raw_parameters) != threshold_count + length(spec$features) + 1L) {
    stopf("clmm parameter layout changed: expected thresholds, slopes, and one log-SD.")
  }
  raw_thresholds <- fixed_parameters[seq_len(threshold_count)]
  raw_beta <- fixed_parameters[threshold_count + seq_along(spec$features)]
  raw_tau <- raw_parameters[length(raw_parameters)]
  sigma_entity <- exp(raw_tau)
  target_order <- c(threshold_count + seq_along(spec$features), seq_len(threshold_count), length(raw_parameters))
  jacobian <- diag(length(raw_parameters))[target_order, , drop = FALSE]
  jacobian[nrow(jacobian), ] <- 0; jacobian[nrow(jacobian), length(raw_parameters)] <- sigma_entity
  parameters <- c(
    stats::setNames(as.numeric(raw_beta), spec$features),
    stats::setNames(as.numeric(raw_thresholds), threshold_labels(categories)),
    sigma_entity = unname(sigma_entity)
  )
  covariance <- jacobian %*% raw_covariance %*% t(jacobian)
  gradient <- if (!is.null(fitted$gradient)) fitted$gradient else rep(NA_real_, length(raw_parameters))
  loglike <- as.numeric(stats::logLik(fitted))
  list(
    model = model, dataset = spec$dataset, parameters = parameters, covariance = covariance,
    loglike = loglike, data = data, X = X, categories = categories,
    beta = stats::setNames(as.numeric(raw_beta), spec$features), thresholds = as.numeric(raw_thresholds),
    sigma_entity = sigma_entity, quadrature_points = 20L,
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(parameters), loglike, n_groups = length(unique(data$entity)),
      score_norm = if (all(is.finite(gradient))) max(abs(gradient)) else NA_real_,
      scaled_score_norm = if (all(is.finite(gradient))) max(abs(gradient)) / nrow(X) else NA_real_,
      backend = "ordinal::clmm-probit-nAGQ-minus20", covariance_type = "observed-information-full-jacobian",
      information_criteria = FALSE
    ), details = "clmm probit random intercept with 20-point non-adaptive quadrature"
  )
}

log_sum_exp <- function(values) {
  maximum <- max(values); maximum + log(sum(exp(values - maximum)))
}

conditional_clone <- function(beta, X, y) {
  n <- length(y); successes <- as.integer(sum(y)); p <- ncol(X)
  if (successes <= 0L || successes >= n) return(list(loglike = 0, score = rep(0, p)))
  eta <- drop(X %*% beta)
  prefix <- matrix(-Inf, n + 1L, successes + 1L); suffix <- matrix(-Inf, n + 1L, successes + 1L)
  prefix[1L, 1L] <- 0
  for (row in seq_len(n)) {
    prefix[row + 1L, ] <- prefix[row, ]
    for (count in seq_len(min(successes, row))) {
      prefix[row + 1L, count + 1L] <- log_sum_exp(c(prefix[row, count + 1L], prefix[row, count] + eta[row]))
    }
  }
  suffix[n + 1L, 1L] <- 0
  for (row in n:1L) {
    suffix[row, ] <- suffix[row + 1L, ]
    for (count in seq_len(min(successes, n - row + 1L))) {
      suffix[row, count + 1L] <- log_sum_exp(c(suffix[row + 1L, count + 1L], suffix[row + 1L, count] + eta[row]))
    }
  }
  denominator <- prefix[n + 1L, successes + 1L]; inclusion <- numeric(n)
  for (row in seq_len(n)) {
    terms <- numeric(0)
    for (left_count in 0:(successes - 1L)) {
      right_count <- successes - 1L - left_count
      if (left_count <= row - 1L && right_count <= n - row) {
        terms <- c(terms, prefix[row, left_count + 1L] + suffix[row + 1L, right_count + 1L])
      }
    }
    inclusion[row] <- exp(eta[row] + log_sum_exp(terms) - denominator)
  }
  list(loglike = sum(y * eta) - denominator, score = drop(crossprod(X, y - inclusion)))
}

make_buc_clones <- function(data, X, categories) {
  entities <- unique(data$entity); clones <- list(); clone_index <- 0L; pseudo_n <- 0L
  for (entity_index in seq_along(entities)) {
    rows <- which(data$entity == entities[entity_index]); encoded <- match(data$y[rows], categories) - 1L
    for (cutoff in 1:(length(categories) - 1L)) {
      binary <- as.integer(encoded >= cutoff)
      if (sum(binary) == 0L || sum(binary) == length(binary)) next
      clone_index <- clone_index + 1L; pseudo_n <- pseudo_n + length(rows)
      clones[[clone_index]] <- list(entity_index = entity_index, rows = rows, y = binary)
    }
  }
  list(clones = clones, entities = entities, pseudo_n = pseudo_n)
}

fit_fe_ordered_logit <- function() {
  model <- "fixed_effects_ordered_logit"; spec <- model_specs[[model]]; data <- datasets[[spec$dataset]]
  require_columns(data, c("y", "entity", spec$features), model)
  X <- design_matrix(data, spec$features, model); categories <- sort(unique(as.integer(data$y)))
  blown <- make_buc_clones(data, X, categories); if (!length(blown$clones)) stopf("BUC has no varying clones.")
  exploded <- do.call(rbind, lapply(seq_along(blown$clones), function(index) {
    clone <- blown$clones[[index]]
    frame <- data.frame(y_binary = clone$y, stratum = index)
    for (feature_index in seq_along(spec$features)) frame[[spec$features[feature_index]]] <- X[clone$rows, feature_index]
    frame
  }))
  formula <- stats::as.formula(paste("y_binary ~", paste(spec$features, collapse = " + "), "+ survival::strata(stratum)"))
  fitted <- survival::clogit(
    formula, data = exploded, method = "exact",
    control = survival::coxph.control(iter.max = 1000L, eps = 1e-10)
  )
  beta <- stats::setNames(as.numeric(stats::coef(fitted)), spec$features)
  entity_values <- unique(vapply(blown$clones, `[[`, integer(1), "entity_index"))
  evaluate <- function(value) {
    loglikes <- numeric(length(blown$entities)); scores <- matrix(0, length(blown$entities), length(beta))
    for (clone in blown$clones) {
      contribution <- conditional_clone(value, X[clone$rows, , drop = FALSE], clone$y)
      loglikes[clone$entity_index] <- loglikes[clone$entity_index] + contribution$loglike
      scores[clone$entity_index, ] <- scores[clone$entity_index, ] + contribution$score
    }
    list(loglikes = loglikes, scores = scores)
  }
  objective <- function(value) -sum(evaluate(value)$loglikes)
  evaluated <- evaluate(beta); information <- numDeriv::hessian(objective, beta, method = "Richardson")
  bread <- safe_inverse(information, model, tolerance = 1e-12)
  entity_scores <- evaluated$scores[entity_values, , drop = FALSE]
  correction <- length(entity_values) / (length(entity_values) - 1)
  if (blown$pseudo_n > length(beta)) correction <- correction * (blown$pseudo_n - 1) / (blown$pseudo_n - length(beta))
  covariance <- correction * bread %*% crossprod(entity_scores) %*% bread
  covariance <- 0.5 * (covariance + t(covariance))
  score <- colSums(evaluated$scores); loglike <- sum(evaluated$loglikes)
  list(
    model = model, dataset = spec$dataset, parameters = beta, covariance = covariance,
    loglike = loglike, data = data, X = X, linear_index = drop(X %*% beta),
    fit = fit_frame(
      model, spec$dataset, nrow(X), length(beta), loglike,
      n_groups = length(unique(data$entity)),
      n_contributing_entities = length(entity_values),
      n_cutoff_clones = length(blown$clones),
      n_pseudo_observations = blown$pseudo_n,
      score_norm = max(abs(score)),
      scaled_score_norm = max(abs(score)) / blown$pseudo_n,
      backend = "survival::clogit-exact-buc", covariance_type = "entity-clustered-CR1-composite",
      information_criteria = FALSE
    ), details = sprintf("BUC exact conditional likelihood; %d clones; %d pseudo-observations; %d contributing entities", length(blown$clones), blown$pseudo_n, length(entity_values))
  )
}

fitters <- list(
  fit_firth, fit_poisson, fit_nb2, fit_tobit, fit_truncated, fit_interval,
  fit_geometric, fit_exponential, fit_weibull, fit_gamma,
  fit_re_ordered_probit, fit_fe_ordered_logit
)

results <- list()
for (fitter in fitters) {
  result <- fitter()
  message(sprintf("Fitted %-36s (%s)", result$model, result$dataset))
  results[[result$model]] <- result
}

all_estimates <- do.call(rbind, lapply(results, function(result) {
  parameter_frames(result$model, result$dataset, result$parameters, result$covariance)$estimates
}))
all_covariance <- do.call(rbind, lapply(results, function(result) {
  parameter_frames(result$model, result$dataset, result$parameters, result$covariance)$covariance
}))
all_fit <- do.call(rbind, lapply(results, `[[`, "fit"))

prediction_path <- file.path(work_directory, "python", "predictions.csv")
if (!file.exists(prediction_path)) stopf("Python prediction key contract is missing: %s", prediction_path)
prediction_keys <- utils::read.csv(prediction_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
expected_prediction_columns <- c("model", "dataset", "obs_id", "prediction", "category", "time", "value")
if (!identical(names(prediction_keys), expected_prediction_columns)) {
  stopf("Python predictions.csv has an unexpected schema.")
}
prediction_keys$value <- NA_real_

zero_effect_probabilities <- function(result, row_indices) {
  X <- result$X[row_indices, , drop = FALSE]; eta <- drop(X %*% result$beta)
  middle <- vapply(
    result$thresholds,
    function(threshold) stats::pnorm(threshold - eta),
    numeric(nrow(X))
  )
  if (is.null(dim(middle))) middle <- matrix(middle, nrow = nrow(X))
  cumulative <- cbind(rep(0, nrow(X)), middle, rep(1, nrow(X)))
  cumulative[, -1L, drop = FALSE] - cumulative[, -ncol(cumulative), drop = FALSE]
}

for (model in names(results)) {
  result <- results[[model]]; selected <- which(prediction_keys$model == model)
  if (!length(selected)) stopf("Prediction contract contains no rows for %s.", model)
  rows <- match(prediction_keys$obs_id[selected], result$data$obs_id)
  if (anyNA(rows)) stopf("Prediction contract for %s references unknown obs_id.", model)
  types <- prediction_keys$prediction[selected]
  for (position in seq_along(selected)) {
    key_index <- selected[position]; row <- rows[position]; type <- types[position]
    value <- NA_real_
    if (identical(model, "firth_binary_logit") && type == "probability") {
      category <- as.integer(prediction_keys$category[key_index]); value <- if (category == 1L) result$probability[row] else 1 - result$probability[row]
    } else if (model %in% c("poisson", "negative_binomial_nb2") && type == "mean") {
      value <- result$mean[row]
    } else if (model == "tobit" && type == "mean") {
      mu <- result$linear_mean[row]; sigma <- result$sigma; z <- mu / sigma
      value <- mu * stats::pnorm(z) + sigma * stats::dnorm(z)
    } else if (model == "tobit" && type == "linear_index") {
      value <- result$linear_mean[row]
    } else if (model == "truncated_regression" && type == "mean") {
      mu <- result$linear_mean[row]; sigma <- result$sigma; a <- (0 - mu) / sigma
      value <- mu + sigma * exp(stats::dnorm(a, log = TRUE) - stats::pnorm(a, lower.tail = FALSE, log.p = TRUE))
    } else if (model == "truncated_regression" && type == "linear_index") {
      value <- result$linear_mean[row]
    } else if (model == "interval_regression" && type %in% c("mean", "linear_index")) {
      value <- result$linear_mean[row]
    } else if (model == "geometric_duration" && type == "mean") {
      value <- 1 / result$hazard[row]
    } else if (model == "geometric_duration" && type == "survival") {
      value <- (1 - result$hazard[row])^as.numeric(prediction_keys$time[key_index])
    } else if (model == "exponential_duration" && type == "mean") {
      value <- result$scale[row]
    } else if (model == "exponential_duration" && type == "survival") {
      value <- exp(-as.numeric(prediction_keys$time[key_index]) / result$scale[row])
    } else if (model == "weibull_duration" && type == "mean") {
      value <- result$scale[row] * gamma(1 + 1 / result$alpha)
    } else if (model == "weibull_duration" && type == "survival") {
      value <- exp(-(as.numeric(prediction_keys$time[key_index]) / result$scale[row])^result$alpha)
    } else if (model == "gamma_duration" && type == "mean") {
      value <- result$k * result$scale[row]
    } else if (model == "gamma_duration" && type == "survival") {
      value <- stats::pgamma(as.numeric(prediction_keys$time[key_index]), shape = result$k, scale = result$scale[row], lower.tail = FALSE)
    } else if (model == "random_effects_ordered_probit" && type == "probability") {
      probabilities <- zero_effect_probabilities(result, row)
      category_index <- match(as.integer(prediction_keys$category[key_index]), result$categories)
      value <- probabilities[1L, category_index]
    } else if (model == "fixed_effects_ordered_logit" && type == "linear_index") {
      value <- result$linear_index[row]
    } else {
      stopf("Unsupported prediction contract row: model=%s prediction=%s", model, type)
    }
    if (!is.finite(value)) stopf("%s produced a non-finite %s prediction.", model, type)
    prediction_keys$value[key_index] <- value
  }
}

evidence_classes <- c(
  firth_binary_logit = "independent-likelihood",
  poisson = "industrial-package",
  negative_binomial_nb2 = "industrial-package",
  tobit = "industrial-package",
  truncated_regression = "independent-likelihood",
  interval_regression = "industrial-package",
  geometric_duration = "likelihood-identity",
  exponential_duration = "industrial-package",
  weibull_duration = "industrial-package",
  gamma_duration = "independent-likelihood",
  random_effects_ordered_probit = "industrial-package",
  fixed_effects_ordered_logit = "likelihood-identity"
)
metadata <- do.call(rbind, lapply(results, function(result) data.frame(
  model = result$model, dataset = result$dataset,
  evidence_class = unname(evidence_classes[result$model]),
  engine = result$fit$backend, engine_version = switch(
    result$model,
    poisson = as.character(utils::packageVersion("stats")),
    negative_binomial_nb2 = as.character(utils::packageVersion("MASS")),
    tobit = as.character(utils::packageVersion("survival")),
    interval_regression = as.character(utils::packageVersion("survival")),
    exponential_duration = as.character(utils::packageVersion("survival")),
    weibull_duration = as.character(utils::packageVersion("survival")),
    random_effects_ordered_probit = as.character(utils::packageVersion("ordinal")),
    fixed_effects_ordered_logit = as.character(utils::packageVersion("survival")),
    as.character(utils::packageVersion("numDeriv"))
  ),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  details = result$details, completed = TRUE,
  stringsAsFactors = FALSE, check.names = FALSE
)))

write_contract <- function(frame, filename) {
  path <- file.path(output_directory, filename); temporary <- paste0(path, ".tmp")
  utils::write.csv(frame, temporary, row.names = FALSE, na = "")
  if (!file.rename(temporary, path)) stopf("Could not publish %s.", path)
}
write_contract(all_estimates, "estimates.csv")
write_contract(all_covariance, "covariance.csv")
write_contract(all_fit, "fit.csv")
write_contract(prediction_keys, "predictions.csv")
write_contract(metadata, "metadata.csv")
message(sprintf("Wrote promoted R parity artifacts to %s", output_directory))
