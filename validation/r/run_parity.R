#!/usr/bin/env Rscript

# Fit all eight parity models against a prepared limiteddepkit fixture.
# Usage from the repository root:
#   Rscript --vanilla validation/r/run_parity.R validation/stata/work
#   Rscript --vanilla validation/r/run_parity.R validation/stata/work/real_data

options(digits = 17, OutDec = ".")

stopf <- function(format, ...) {
  stop(sprintf(format, ...), call. = FALSE)
}

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 1L) {
  stopf("Usage: Rscript --vanilla validation/r/run_parity.R <prepared-work-directory>")
}
work_directory <- normalizePath(arguments[[1L]], winslash = "/", mustWork = TRUE)
output_directory <- file.path(work_directory, "r")
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
    stopf("Could not invalidate stale R parity artifact: %s", artifact)
  }
}

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(file_argument) != 1L) {
  stopf("Cannot determine the location of validation/r/run_parity.R.")
}
script_path <- normalizePath(
  sub("^--file=", "", file_argument),
  winslash = "/",
  mustWork = TRUE
)
script_directory <- dirname(script_path)
local_library <- file.path(script_directory, "work", "library")
if (dir.exists(local_library)) {
  .libPaths(c(local_library, .Library))
} else {
  .libPaths(.Library)
}
sys.source(file.path(script_directory, "flexible_models.R"), envir = .GlobalEnv)
sys.source(file.path(script_directory, "panel_models.R"), envir = .GlobalEnv)

require_package <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stopf("Required R package %s is not installed.", sQuote(package))
  }
}

require_columns <- function(data, columns, context) {
  missing <- setdiff(columns, names(data))
  if (length(missing) > 0L) {
    stopf("%s is missing columns: %s", context, paste(missing, collapse = ", "))
  }
}

as_manifest_vector <- function(value) {
  unname(as.character(unlist(value, use.names = FALSE)))
}

category_label <- function(value) {
  if (!is.finite(value) || abs(value - round(value)) > 1e-12) {
    stopf("Ordinal outcome categories must be finite integers; found %s.", value)
  }
  as.character(as.integer(round(value)))
}

validate_design <- function(data, features, outcome, model, binary = FALSE) {
  require_columns(data, c("obs_id", outcome, features), model)
  if (anyDuplicated(data$obs_id)) {
    stopf("%s contains duplicate obs_id values.", model)
  }
  obs_id <- as.numeric(data$obs_id)
  if (any(!is.finite(obs_id)) || any(abs(obs_id - round(obs_id)) > 1e-12)) {
    stopf("%s obs_id values must be finite integers.", model)
  }
  values <- as.matrix(data[, features, drop = FALSE])
  storage.mode(values) <- "double"
  y <- as.numeric(data[[outcome]])
  if (any(!is.finite(values)) || any(!is.finite(y))) {
    stopf("%s contains missing or non-finite analysis values.", model)
  }
  if (qr(values)$rank != ncol(values)) {
    stopf("%s design matrix is rank deficient.", model)
  }
  support <- sort(unique(y))
  if (binary && !identical(support, c(0, 1))) {
    stopf("%s outcome support must be exactly {0, 1}.", model)
  }
  if (!binary && length(support) < 3L) {
    stopf("%s requires at least three observed outcome categories.", model)
  }
  list(X = values, y = y, support = support)
}

safe_inverse <- function(information, model) {
  information <- 0.5 * (information + t(information))
  values <- eigen(information, symmetric = TRUE, only.values = TRUE)$values
  if (any(!is.finite(values)) || min(values) <= 0) {
    stopf("%s observed-information matrix is not positive definite.", model)
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
    stopf("%s covariance dimensions do not match its parameters.", model)
  }
  dimnames(covariance) <- list(parameter_names, parameter_names)
  standard_errors <- sqrt(diag(covariance))
  if (any(!is.finite(parameters)) || any(!is.finite(covariance)) ||
      any(!is.finite(standard_errors))) {
    stopf("%s produced non-finite estimates or inference.", model)
  }

  estimates <- data.frame(
    model = model,
    dataset = dataset,
    parameter = parameter_names,
    estimate = unname(parameters),
    standard_error = unname(standard_errors),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  n_parameters <- length(parameters)
  covariance_frame <- data.frame(
    model = rep(model, n_parameters * n_parameters),
    dataset = rep(dataset, n_parameters * n_parameters),
    row_parameter = rep(parameter_names, each = n_parameters),
    column_parameter = rep(parameter_names, times = n_parameters),
    covariance = as.vector(t(covariance)),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  list(estimates = estimates, covariance = covariance_frame)
}

prediction_frame <- function(model, dataset, obs_id, categories, probabilities,
                             prediction_rows) {
  probabilities <- as.matrix(probabilities)
  if (nrow(probabilities) != length(obs_id) ||
      ncol(probabilities) != length(categories)) {
    stopf("%s prediction dimensions are inconsistent.", model)
  }
  if (any(!is.finite(probabilities)) || any(probabilities < -1e-12) ||
      any(probabilities > 1 + 1e-12)) {
    stopf("%s produced invalid probabilities.", model)
  }
  row_sums <- rowSums(probabilities)
  if (max(abs(row_sums - 1)) > 1e-9) {
    stopf("%s probabilities do not sum to one.", model)
  }
  rows <- seq_len(min(prediction_rows, nrow(probabilities)))
  data.frame(
    model = rep(model, length(rows) * length(categories)),
    dataset = rep(dataset, length(rows) * length(categories)),
    obs_id = rep(as.integer(obs_id[rows]), each = length(categories)),
    category = rep(as.integer(categories), times = length(rows)),
    probability = as.vector(t(probabilities[rows, , drop = FALSE])),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_frame <- function(model, dataset, nobs, n_parameters, loglike, converged,
                      inference_valid) {
  data.frame(
    model = model,
    dataset = dataset,
    nobs = as.integer(nobs),
    n_groups = NA_integer_,
    n_params = as.integer(n_parameters),
    loglike = as.numeric(loglike),
    aic = -2 * loglike + 2 * n_parameters,
    bic = -2 * loglike + log(nobs) * n_parameters,
    converged = isTRUE(converged),
    inference_valid = isTRUE(inference_valid),
    quadrature_points = NA_integer_,
    constraint_slack = NA_real_,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fit_binary <- function(spec, data, prediction_rows) {
  validated <- validate_design(
    data, spec$features, spec$outcome, spec$model, binary = TRUE
  )
  X <- validated$X
  y <- validated$y
  colnames(X) <- spec$features
  fitted <- stats::glm.fit(
    x = X,
    y = y,
    family = stats::binomial(link = spec$link),
    control = stats::glm.control(epsilon = 1e-12, maxit = 1000L)
  )
  if (!isTRUE(fitted$converged) || fitted$rank != ncol(X)) {
    stopf("%s did not converge to a full-rank solution.", spec$model)
  }
  parameters <- stats::setNames(as.numeric(fitted$coefficients), spec$features)
  eta <- drop(X %*% parameters)

  if (identical(spec$link, "logit")) {
    probability_one <- stats::plogis(eta)
    weights <- probability_one * (1 - probability_one)
    softplus <- pmax(eta, 0) + log1p(exp(-abs(eta)))
    loglike <- sum(y * eta - softplus)
  } else if (identical(spec$link, "probit")) {
    probability_one <- stats::pnorm(eta)
    signed_index <- (2 * y - 1) * eta
    inverse_mills <- exp(
      stats::dnorm(signed_index, log = TRUE) -
        stats::pnorm(signed_index, log.p = TRUE)
    )
    weights <- inverse_mills * (signed_index + inverse_mills)
    loglike <- sum(stats::pnorm(signed_index, log.p = TRUE))
  } else {
    stopf("Unsupported binary link %s.", sQuote(spec$link))
  }

  information <- crossprod(X, sweep(X, 1L, weights, `*`))
  covariance <- safe_inverse(information, spec$model)
  frames <- parameter_frames(spec$model, spec$dataset, parameters, covariance)
  probabilities <- cbind(1 - probability_one, probability_one)
  predictions <- prediction_frame(
    spec$model,
    spec$dataset,
    data$obs_id,
    c(0L, 1L),
    probabilities,
    prediction_rows
  )
  fit <- fit_frame(
    spec$model,
    spec$dataset,
    nrow(X),
    length(parameters),
    loglike,
    fitted$converged,
    TRUE
  )
  c(frames, list(fit = fit, predictions = predictions))
}

fit_ordered <- function(spec, data, prediction_rows) {
  validated <- validate_design(
    data, spec$features, spec$outcome, spec$model, binary = FALSE
  )
  support <- validated$support
  category_names <- vapply(support, category_label, character(1))
  model_data <- data[, spec$features, drop = FALSE]
  model_data$.outcome <- ordered(
    vapply(validated$y, category_label, character(1)),
    levels = category_names
  )
  formula <- stats::reformulate(spec$features, response = ".outcome")
  fitted <- MASS::polr(
    formula,
    data = model_data,
    method = spec$method,
    Hess = TRUE,
    model = TRUE,
    control = list(maxit = 5000L, reltol = 1e-12)
  )
  converged <- identical(as.integer(fitted$convergence), 0L)
  if (!converged) {
    stopf("%s failed with optimizer code %s.", spec$model, fitted$convergence)
  }

  slopes <- fitted$coefficients
  if (!identical(names(slopes), spec$features)) {
    stopf("%s returned an unexpected coefficient order.", spec$model)
  }
  cut_names <- paste0(
    "threshold: ",
    category_names[-length(category_names)],
    " | ",
    category_names[-1L]
  )
  parameters <- c(slopes, stats::setNames(as.numeric(fitted$zeta), cut_names))
  covariance <- stats::vcov(fitted)
  expected_covariance_names <- c(spec$features, names(fitted$zeta))
  if (!identical(rownames(covariance), expected_covariance_names) ||
      !identical(colnames(covariance), expected_covariance_names)) {
    stopf("%s returned an unexpected covariance order.", spec$model)
  }
  dimnames(covariance) <- list(names(parameters), names(parameters))
  frames <- parameter_frames(spec$model, spec$dataset, parameters, covariance)

  probabilities <- stats::predict(fitted, newdata = model_data, type = "probs")
  probabilities <- as.matrix(probabilities)
  probabilities <- probabilities[, category_names, drop = FALSE]
  predictions <- prediction_frame(
    spec$model,
    spec$dataset,
    data$obs_id,
    support,
    probabilities,
    prediction_rows
  )
  loglike <- as.numeric(stats::logLik(fitted))
  fit <- fit_frame(
    spec$model,
    spec$dataset,
    nrow(model_data),
    length(parameters),
    loglike,
    converged,
    TRUE
  )
  c(frames, list(fit = fit, predictions = predictions))
}

fixture_specs <- function(manifest) {
  suite <- as.character(manifest$suite)
  if (identical(suite, "controlled_synthetic_certification")) {
    return(list(
      list(
        model = "binary_logit", kind = "binary", link = "logit",
        dataset = "cross_section", file = "cross_section.csv",
        outcome = "y_logit", features = c("intercept", "x1", "x2")
      ),
      list(
        model = "binary_probit", kind = "binary", link = "probit",
        dataset = "cross_section", file = "cross_section.csv",
        outcome = "y_probit", features = c("intercept", "x1", "x2")
      ),
      list(
        model = "ordered_logit", kind = "ordered", method = "logistic",
        dataset = "cross_section", file = "cross_section.csv",
        outcome = "y_ologit", features = c("ox1", "ox2")
      ),
      list(
        model = "ordered_probit", kind = "ordered", method = "probit",
        dataset = "cross_section", file = "cross_section.csv",
        outcome = "y_oprobit", features = c("ox1", "ox2")
      )
    ))
  }

  if (identical(suite, "real_data_application")) {
    model_specs <- manifest$comparison_model_specs
    if (is.null(model_specs)) {
      stopf("The real-data manifest has no comparison_model_specs section.")
    }
    return(list(
      list(
        model = "binary_logit", kind = "binary", link = "logit",
        dataset = "binary_lbw", file = "binary_lbw.csv", outcome = "y",
        features = as_manifest_vector(model_specs$binary_logit$features)
      ),
      list(
        model = "binary_probit", kind = "binary", link = "probit",
        dataset = "binary_lbw", file = "binary_lbw.csv", outcome = "y",
        features = as_manifest_vector(model_specs$binary_probit$features)
      ),
      list(
        model = "ordered_logit", kind = "ordered", method = "logistic",
        dataset = "ordinal_tvsfpors", file = "ordinal_tvsfpors.csv", outcome = "y",
        features = as_manifest_vector(model_specs$ordered_logit$features)
      ),
      list(
        model = "ordered_probit", kind = "ordered", method = "probit",
        dataset = "ordinal_tvsfpors", file = "ordinal_tvsfpors.csv", outcome = "y",
        features = as_manifest_vector(model_specs$ordered_probit$features)
      )
    ))
  }

  stopf("Unsupported parity fixture suite %s.", sQuote(suite))
}

write_canonical_csv <- function(data, path) {
  utils::write.table(
    data,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = TRUE,
    quote = TRUE,
    na = "",
    qmethod = "double",
    fileEncoding = "UTF-8"
  )
}

require_package("jsonlite")
require_package("MASS")
require_package("VGAM")
require_package("ordinal")
require_package("Matrix")
require_package("nlme")

local_packages <- c("MASS", "jsonlite", "numDeriv", "ucminf", "ordinal", "VGAM")
local_root <- normalizePath(local_library, winslash = "/", mustWork = TRUE)
local_paths <- vapply(
  local_packages,
  function(package) dirname(normalizePath(find.package(package), winslash = "/")),
  ""
)
if (any(local_paths != local_root)) {
  stopf(
    "Pinned packages did not resolve from validation/r/work/library: %s",
    paste(names(local_paths)[local_paths != local_root], collapse = ", ")
  )
}
system_packages <- c("Matrix", "nlme")
system_root <- normalizePath(.Library, winslash = "/", mustWork = TRUE)
system_paths <- vapply(
  system_packages,
  function(package) dirname(normalizePath(find.package(package), winslash = "/")),
  ""
)
if (any(system_paths != system_root)) {
  stopf(
    "Recommended packages did not resolve from the R 4.5.1 library: %s",
    paste(names(system_paths)[system_paths != system_root], collapse = ", ")
  )
}

manifest_path <- file.path(work_directory, "manifest.json")
if (!file.exists(manifest_path)) {
  stopf("No manifest.json found under %s.", work_directory)
}
manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
if (!identical(as.integer(manifest$schema_version), 1L)) {
  stopf("Unsupported source manifest schema version %s.", manifest$schema_version)
}
prediction_rows <- as.integer(manifest$prediction_rows_per_model)
if (length(prediction_rows) != 1L || is.na(prediction_rows) || prediction_rows < 1L) {
  stopf("The manifest has an invalid prediction_rows_per_model value.")
}

dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

read_registered_data <- function(filename) {
  relative_path <- file.path("data", filename)
  manifest_key <- gsub("\\\\", "/", relative_path)
  if (is.null(manifest$files[[manifest_key]])) {
    stopf("The source manifest does not register %s.", relative_path)
  }
  data_path <- file.path(work_directory, relative_path)
  if (!file.exists(data_path)) {
    stopf("Prepared data file does not exist: %s", data_path)
  }
  utils::read.csv(data_path, check.names = FALSE, stringsAsFactors = FALSE)
}

specifications <- fixture_specs(manifest)
results <- vector("list", length(specifications))
for (index in seq_along(specifications)) {
  spec <- specifications[[index]]
  data <- read_registered_data(spec$file)
  if (identical(spec$kind, "binary")) {
    results[[index]] <- fit_binary(spec, data, prediction_rows)
  } else {
    results[[index]] <- fit_ordered(spec, data, prediction_rows)
  }
  message(sprintf("Fitted %-15s (%d observations)", spec$model,
                  results[[index]]$fit$nobs))
}

suite <- as.character(manifest$suite)
if (identical(suite, "controlled_synthetic_certification")) {
  flexible_spec <- list(
    file = "cross_section.csv",
    dataset = "cross_section",
    response = "y_gologit",
    features = c("gx1", "gx2"),
    varying = "gx1",
    categories = 0:2
  )
  panel_specs <- list(
    list(
      model = "random_effects_ordered_logit",
      file = "static_re.csv",
      dataset = "static_re",
      features = c("x1", "x2"),
      canonical = c("x1", "x2"),
      categories = 0:2
    ),
    list(
      model = "dynamic_random_effects_ordered_logit",
      file = "dynamic_design.csv",
      dataset = "dynamic_design",
      features = c(
        "x1", "state_1", "state_2", "initial_1", "initial_2",
        "initial_x1", "mean_x1"
      ),
      canonical = c(
        "x1", "state[1]", "state[2]", "initial[1]", "initial[2]",
        "initial_x[x1]", "mean[x1]"
      ),
      categories = 0:2
    )
  )
} else if (identical(suite, "real_data_application")) {
  flexible_spec <- list(
    file = "ordinal_tvsfpors.csv",
    dataset = "ordinal_tvsfpors",
    response = "y",
    features = c("gx1", "gx2", "gx3", "gx4"),
    varying = "gx4",
    categories = 0:3
  )
  panel_specs <- list(
    list(
      model = "random_effects_ordered_logit",
      file = "ordinal_tvsfpors.csv",
      dataset = "ordinal_tvsfpors",
      features = c("x1", "x2", "x3", "x4"),
      canonical = c("x1", "x2", "x3", "x4"),
      categories = 0:3
    ),
    list(
      model = "dynamic_random_effects_ordered_logit",
      file = "dynamic_nlswork_design.csv",
      dataset = "dynamic_nlswork_design",
      features = c(
        "x1", "state_1", "state_2", "initial_1", "initial_2",
        "initial_x1", "mean_x1"
      ),
      canonical = c(
        "x1", "state[1]", "state[2]", "initial[1]", "initial[2]",
        "initial_x[x1]", "mean[x1]"
      ),
      categories = 0:2
    )
  )
} else {
  stopf("Unsupported parity fixture suite %s.", sQuote(suite))
}

flexible_data <- read_registered_data(flexible_spec$file)
flexible_result <- ldk_fit_flexible_models(
  data = flexible_data,
  dataset = flexible_spec$dataset,
  response = flexible_spec$response,
  features = flexible_spec$features,
  varying = flexible_spec$varying,
  categories = flexible_spec$categories,
  prediction_rows = prediction_rows
)
results[[length(results) + 1L]] <- flexible_result
message("Fitted generalized_ordered_logit and partial_proportional_odds")

quadrature_points <- as.integer(manifest$quadrature_points)
if (length(quadrature_points) != 1L || is.na(quadrature_points) ||
    quadrature_points < 3L) {
  stopf("The manifest has an invalid quadrature_points value.")
}
for (panel_spec in panel_specs) {
  panel_data <- read_registered_data(panel_spec$file)
  panel_result <- ldk_fit_panel_model(
    data = panel_data,
    model = panel_spec$model,
    dataset = panel_spec$dataset,
    feature_names = panel_spec$features,
    canonical_feature_names = panel_spec$canonical,
    category_order = panel_spec$categories,
    quadrature_points = quadrature_points,
    prediction_obs_ids = head(panel_data$obs_id, prediction_rows)
  )
  results[[length(results) + 1L]] <- panel_result
  message(sprintf("Fitted %-38s (%d observations)", panel_spec$model,
                  panel_result$fit$nobs))
}

combine <- function(field) {
  do.call(rbind, lapply(results, `[[`, field))
}

actual_models <- sort(unique(as.character(combine("fit")$model)))
expected_models <- sort(names(manifest$models))
if (length(expected_models) == 0L) {
  expected_models <- sort(names(manifest$comparison_model_specs))
}
if (!identical(actual_models, expected_models)) {
  stopf(
    "R output model set differs from the manifest: expected %s; found %s.",
    paste(expected_models, collapse = ", "),
    paste(actual_models, collapse = ", ")
  )
}
metadata <- data.frame(
  key = c(
    "schema_version",
    "source_manifest_schema_version",
    "suite",
    "runner",
    "r_version",
    "stats_version",
    "MASS_version",
    "jsonlite_version",
    "VGAM_version",
    "ordinal_version",
    "numDeriv_version",
    "ucminf_version",
    "Matrix_version",
    "nlme_version",
    "models",
    "prediction_rows_per_model",
    "binary_covariance",
    "binary_probit_covariance",
    "ordered_estimator",
    "flexible_estimator",
    "flexible_covariance",
    "panel_estimator",
    "panel_quadrature",
    "panel_prediction",
    "completion_marker"
  ),
  value = c(
    "1",
    as.character(manifest$schema_version),
    as.character(manifest$suite),
    "validation/r/run_parity.R",
    as.character(getRversion()),
    as.character(utils::packageDescription("stats", fields = "Version")),
    as.character(utils::packageDescription("MASS", fields = "Version")),
    as.character(utils::packageDescription("jsonlite", fields = "Version")),
    as.character(utils::packageDescription("VGAM", fields = "Version")),
    as.character(utils::packageDescription("ordinal", fields = "Version")),
    as.character(utils::packageDescription("numDeriv", fields = "Version")),
    as.character(utils::packageDescription("ucminf", fields = "Version")),
    as.character(utils::packageDescription("Matrix", fields = "Version")),
    as.character(utils::packageDescription("nlme", fields = "Version")),
    paste(actual_models, collapse = ";"),
    as.character(prediction_rows),
    "observed-information",
    "inverse-Mills observed-information",
    "MASS::polr",
    "VGAM::vglm cumulative logit",
    "limiteddepkit-compatible central observed Hessian",
    "ordinal::clmm",
    sprintf("nonadaptive Gauss-Hermite, nAGQ=%d", -quadrature_points),
    "conditional fixed-only: random intercept = 0",
    "R_PARITY_COMPLETE"
  ),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

write_canonical_csv(combine("estimates"), file.path(output_directory, "estimates.csv"))
write_canonical_csv(combine("covariance"), file.path(output_directory, "covariance.csv"))
write_canonical_csv(combine("fit"), file.path(output_directory, "fit.csv"))
write_canonical_csv(combine("predictions"), file.path(output_directory, "predictions.csv"))
write_canonical_csv(metadata, file.path(output_directory, "metadata.csv"))

message(sprintf("R parity outputs written to %s", output_directory))
