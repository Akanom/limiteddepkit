# Canonical R references for limiteddepkit's flexible ordinal models.
#
# VGAM uses
#   logit(P(Y <= j)) = alpha_j + x' gamma_j,
# whereas limiteddepkit uses
#   logit(P(Y <= j)) = threshold_j - x' beta_j.
# Consequently, thresholds equal VGAM's intercepts and every slope changes
# sign.  VGAM stores nonparallel slopes feature-major; limiteddepkit stores
# them threshold-major.  The named extraction below applies both mappings.

.ldk_required_vgam_version <- "1.1-14"

.ldk_require_vgam <- function(required_version = .ldk_required_vgam_version) {
  if (!requireNamespace("VGAM", quietly = TRUE)) {
    stop(
      "Package 'VGAM' is required for flexible ordinal R parity.",
      call. = FALSE
    )
  }
  installed <- unname(utils::packageDescription("VGAM", fields = "Version"))
  if (!is.null(required_version) && !identical(installed, required_version)) {
    stop(
      sprintf(
        "Flexible ordinal parity requires VGAM %s; found %s.",
        required_version,
        installed
      ),
      call. = FALSE
    )
  }
  invisible(installed)
}

.ldk_flexible_fixture_spec <- function(fixture) {
  fixture <- match.arg(fixture, c("synthetic", "real"))
  if (identical(fixture, "synthetic")) {
    return(list(
      data_file = file.path("data", "cross_section.csv"),
      dataset = "cross_section",
      response = "y_gologit",
      features = c("gx1", "gx2"),
      varying = "gx1",
      categories = 0:2,
      prediction_rows = 25L
    ))
  }
  list(
    data_file = file.path("data", "ordinal_tvsfpors.csv"),
    dataset = "ordinal_tvsfpors",
    response = "y",
    features = c("gx1", "gx2", "gx3", "gx4"),
    varying = "gx4",
    categories = 0:3,
    prediction_rows = 25L
  )
}

.ldk_check_columns <- function(data, columns) {
  missing <- setdiff(columns, names(data))
  if (length(missing)) {
    stop(
      sprintf("Missing required columns: %s.", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
}

.ldk_flexible_unpack <- function(theta, n_thresholds, features, varying) {
  n_features <- length(features)
  thresholds <- theta[seq_len(n_thresholds)]
  slope_values <- theta[-seq_len(n_thresholds)]

  if (is.null(varying)) {
    slopes <- matrix(
      slope_values,
      nrow = n_thresholds,
      ncol = n_features,
      byrow = TRUE,
      dimnames = list(NULL, features)
    )
  } else {
    common <- features[!features %in% varying]
    n_common <- length(common)
    slopes <- matrix(
      NA_real_,
      nrow = n_thresholds,
      ncol = n_features,
      dimnames = list(NULL, features)
    )
    if (n_common) {
      common_values <- slope_values[seq_len(n_common)]
      slopes[, common] <- matrix(
        rep(common_values, each = n_thresholds),
        nrow = n_thresholds,
        ncol = n_common
      )
    }
    varying_values <- if (n_common) {
      slope_values[-seq_len(n_common)]
    } else {
      slope_values
    }
    slopes[, varying] <- matrix(
      varying_values,
      nrow = n_thresholds,
      ncol = length(varying),
      byrow = TRUE
    )
  }
  list(thresholds = thresholds, slopes = slopes)
}

.ldk_flexible_indices <- function(theta, design, features, varying) {
  # Infer M without depending on coefficient names: generalized models have
  # M * (p + 1) parameters; PPO models have M + p_common + M * p_varying.
  if (is.null(varying)) {
    n_thresholds <- length(theta) / (length(features) + 1)
  } else {
    n_common <- sum(!features %in% varying)
    n_thresholds <- (length(theta) - n_common) / (length(varying) + 1)
  }
  n_thresholds <- as.integer(n_thresholds)
  unpacked <- .ldk_flexible_unpack(theta, n_thresholds, features, varying)
  sweep(
    -design %*% t(unpacked$slopes),
    MARGIN = 2L,
    STATS = unpacked$thresholds,
    FUN = "+"
  )
}

.ldk_flexible_probabilities <- function(theta, design, features, varying) {
  indices <- .ldk_flexible_indices(theta, design, features, varying)
  cumulative <- stats::plogis(indices)
  bounds <- cbind(0, cumulative, 1)
  bounds[, -1L, drop = FALSE] - bounds[, -ncol(bounds), drop = FALSE]
}

.ldk_central_hessian <- function(fn, point) {
  point <- as.numeric(point)
  steps <- 1e-4 * (1 + abs(point))
  size <- length(point)
  answer <- matrix(NA_real_, nrow = size, ncol = size)
  center <- fn(point)

  for (row in seq_len(size)) {
    row_step <- numeric(size)
    row_step[row] <- steps[row]
    answer[row, row] <- (
      fn(point + row_step) - 2 * center + fn(point - row_step)
    ) / steps[row]^2

    if (row > 1L) {
      for (column in seq_len(row - 1L)) {
        column_step <- numeric(size)
        column_step[column] <- steps[column]
        value <- (
          fn(point + row_step + column_step) -
            fn(point + row_step - column_step) -
            fn(point - row_step + column_step) +
            fn(point - row_step - column_step)
        ) / (4 * steps[row] * steps[column])
        answer[row, column] <- value
        answer[column, row] <- value
      }
    }
  }
  answer
}

.ldk_pseudoinverse <- function(matrix, rcond = 1e-15) {
  decomposition <- base::svd(matrix)
  keep <- decomposition$d > rcond * max(decomposition$d)
  if (!any(keep)) {
    return(base::matrix(0, nrow = ncol(matrix), ncol = nrow(matrix)))
  }
  decomposition$v[, keep, drop = FALSE] %*%
    (diag(1 / decomposition$d[keep], nrow = sum(keep)) %*%
      t(decomposition$u[, keep, drop = FALSE]))
}

.ldk_named_vgam_mapping <- function(fit, features, varying, categories) {
  n_thresholds <- length(categories) - 1L
  splits <- paste(categories[-length(categories)], categories[-1L], sep = " | ")
  threshold_sources <- paste0("(Intercept):", seq_len(n_thresholds))
  threshold_names <- paste0("threshold: ", splits)

  if (is.null(varying)) {
    slope_sources <- unlist(
      lapply(
        seq_len(n_thresholds),
        function(split) paste0(features, ":", split)
      ),
      use.names = FALSE
    )
    slope_names <- unlist(
      lapply(
        seq_len(n_thresholds),
        function(split) paste0("slope ", splits[split], ": ", features)
      ),
      use.names = FALSE
    )
  } else {
    common <- features[!features %in% varying]
    slope_sources <- c(
      common,
      unlist(
        lapply(
          seq_len(n_thresholds),
          function(split) paste0(varying, ":", split)
        ),
        use.names = FALSE
      )
    )
    slope_names <- c(
      paste0("common: ", common),
      unlist(
        lapply(
          seq_len(n_thresholds),
          function(split) paste0("varying ", splits[split], ": ", varying)
        ),
        use.names = FALSE
      )
    )
  }

  sources <- c(threshold_sources, slope_sources)
  canonical_names <- c(threshold_names, slope_names)
  raw <- stats::coef(fit)
  missing <- setdiff(sources, names(raw))
  if (length(missing)) {
    stop(
      sprintf(
        "Unexpected VGAM coefficient names; missing: %s.",
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  signs <- c(rep(1, n_thresholds), rep(-1, length(slope_sources)))
  theta <- unname(raw[sources]) * signs
  names(theta) <- canonical_names
  theta
}

.ldk_fit_one_flexible <- function(
    data,
    dataset,
    response,
    features,
    varying,
    categories,
    prediction_rows,
    model,
    maxit,
    epsilon,
    minimum_gap,
    boundary_tolerance) {
  model_data <- data
  encoded <- match(model_data[[response]], categories)
  if (anyNA(encoded)) {
    stop("The response contains values outside category_order.", call. = FALSE)
  }
  model_data$.ldk_y <- ordered(
    model_data[[response]],
    levels = categories
  )
  formula <- stats::reformulate(features, response = ".ldk_y")
  model_varying <- if (identical(model, "generalized_ordered_logit")) {
    NULL
  } else {
    varying
  }

  if (is.null(model_varying)) {
    family <- VGAM::cumulative(
      link = "logitlink",
      parallel = FALSE,
      reverse = FALSE
    )
  } else {
    common <- features[!features %in% model_varying]
    if (length(common)) {
      parallel_formula <- stats::as.formula(
        paste("~ -1", paste(common, collapse = " + "), sep = " + ")
      )
      family <- VGAM::cumulative(
        link = "logitlink",
        parallel = parallel_formula,
        reverse = FALSE
      )
    } else {
      family <- VGAM::cumulative(
        link = "logitlink",
        parallel = FALSE,
        reverse = FALSE
      )
    }
  }

  # Pass controls through ... rather than a prebuilt control list.  VGAM's
  # categorical-family control layer otherwise resets maxit to its default.
  fit <- VGAM::vglm(
    formula,
    family = family,
    data = model_data,
    maxit = maxit,
    epsilon = epsilon,
    trace = FALSE
  )
  iterations <- VGAM::niters(fit)
  converged <- is.finite(iterations) && iterations < maxit
  if (!converged) {
    stop(sprintf("%s did not converge in VGAM.", model), call. = FALSE)
  }

  theta <- .ldk_named_vgam_mapping(
    fit,
    features = features,
    varying = model_varying,
    categories = categories
  )
  design <- as.matrix(model_data[, features, drop = FALSE])
  storage.mode(design) <- "double"
  objective <- function(parameters) {
    probabilities <- .ldk_flexible_probabilities(
      parameters,
      design,
      features,
      model_varying
    )
    selected <- probabilities[cbind(seq_len(nrow(probabilities)), encoded)]
    -sum(log(pmax(selected, 1e-15)))
  }

  loglike <- as.numeric(stats::logLik(fit))
  if (!is.finite(loglike) || abs(loglike + objective(theta)) > 1e-6) {
    stop("VGAM and canonical flexible-ordinal likelihoods disagree.", call. = FALSE)
  }

  indices <- .ldk_flexible_indices(theta, design, features, model_varying)
  gaps <- indices[, -1L, drop = FALSE] -
    indices[, -ncol(indices), drop = FALSE]
  minimum_index_gap <- min(gaps)
  constraint_slack <- minimum_index_gap - minimum_gap
  if (!is.finite(minimum_index_gap) || minimum_index_gap < minimum_gap - 1e-7) {
    stop(sprintf("%s produced crossing cumulative logits.", model), call. = FALSE)
  }
  inference_valid <- constraint_slack > boundary_tolerance

  # VGAM's vcov() is based on Fisher scoring.  limiteddepkit reports the
  # inverse observed numerical Hessian, so reproduce its central differences
  # and step sizes here rather than comparing different covariance estimands.
  information <- .ldk_central_hessian(objective, theta)
  covariance <- .ldk_pseudoinverse(information)
  covariance <- (covariance + t(covariance)) / 2
  dimnames(covariance) <- list(names(theta), names(theta))
  standard_errors <- sqrt(pmax(diag(covariance), 0))
  if (any(!is.finite(covariance)) || any(!is.finite(standard_errors))) {
    stop(sprintf("%s produced non-finite observed inference.", model), call. = FALSE)
  }

  estimates <- data.frame(
    model = model,
    dataset = dataset,
    parameter = names(theta),
    estimate = unname(theta),
    standard_error = unname(standard_errors),
    stringsAsFactors = FALSE
  )
  covariance_frame <- data.frame(
    model = model,
    dataset = dataset,
    row_parameter = rep(names(theta), each = length(theta)),
    column_parameter = rep(names(theta), times = length(theta)),
    covariance = as.vector(t(covariance)),
    stringsAsFactors = FALSE
  )

  n_parameters <- length(theta)
  n_observations <- nrow(model_data)
  fit_frame <- data.frame(
    model = model,
    dataset = dataset,
    nobs = n_observations,
    n_groups = NA_integer_,
    n_params = n_parameters,
    loglike = loglike,
    aic = -2 * loglike + 2 * n_parameters,
    bic = -2 * loglike + log(n_observations) * n_parameters,
    converged = converged,
    inference_valid = inference_valid,
    quadrature_points = NA_integer_,
    constraint_slack = constraint_slack,
    stringsAsFactors = FALSE
  )

  prediction_rows <- min(as.integer(prediction_rows), n_observations)
  prediction_data <- model_data[seq_len(prediction_rows), , drop = FALSE]
  probabilities <- VGAM::predictvglm(
    fit,
    newdata = prediction_data,
    type = "response"
  )
  probabilities <- as.matrix(probabilities)
  category_names <- as.character(categories)
  if (!setequal(colnames(probabilities), category_names)) {
    stop("VGAM returned unexpected probability columns.", call. = FALSE)
  }
  probabilities <- probabilities[, category_names, drop = FALSE]
  canonical_probabilities <- .ldk_flexible_probabilities(
    theta,
    design[seq_len(prediction_rows), , drop = FALSE],
    features,
    model_varying
  )
  if (
    any(!is.finite(probabilities)) ||
      max(abs(probabilities - canonical_probabilities)) > 1e-10 ||
      max(abs(rowSums(probabilities) - 1)) > 1e-10 ||
      min(probabilities) < -1e-10 ||
      max(probabilities) > 1 + 1e-10
  ) {
    stop("VGAM returned invalid or misordered probabilities.", call. = FALSE)
  }
  predictions <- data.frame(
    model = model,
    dataset = dataset,
    obs_id = rep(prediction_data$obs_id, each = length(categories)),
    category = rep(categories, times = prediction_rows),
    probability = as.vector(t(probabilities)),
    stringsAsFactors = FALSE
  )

  list(
    estimates = estimates,
    covariance = covariance_frame,
    fit = fit_frame,
    predictions = predictions
  )
}

ldk_fit_flexible_models <- function(
    data,
    dataset,
    response,
    features,
    varying,
    categories,
    prediction_rows,
    maxit = 1000L,
    epsilon = 1e-12,
    minimum_gap = 1e-6,
    boundary_tolerance = 1e-5,
    required_vgam_version = .ldk_required_vgam_version) {
  .ldk_require_vgam(required_vgam_version)
  .ldk_check_columns(data, c("obs_id", response, features))
  if (!length(varying) || any(!varying %in% features)) {
    stop("varying must name at least one supplied feature.", call. = FALSE)
  }
  if (length(unique(categories)) < 3L) {
    stop("Flexible ordinal parity requires at least three categories.", call. = FALSE)
  }

  models <- c("generalized_ordered_logit", "partial_proportional_odds")
  results <- lapply(
    models,
    function(model) {
      .ldk_fit_one_flexible(
        data = data,
        dataset = dataset,
        response = response,
        features = features,
        varying = varying,
        categories = categories,
        prediction_rows = prediction_rows,
        model = model,
        maxit = as.integer(maxit),
        epsilon = epsilon,
        minimum_gap = minimum_gap,
        boundary_tolerance = boundary_tolerance
      )
    }
  )
  names(results) <- models
  combined <- list(
    estimates = do.call(rbind, lapply(results, `[[`, "estimates")),
    covariance = do.call(rbind, lapply(results, `[[`, "covariance")),
    fit = do.call(rbind, lapply(results, `[[`, "fit")),
    predictions = do.call(rbind, lapply(results, `[[`, "predictions"))
  )
  for (name in names(combined)) {
    rownames(combined[[name]]) <- NULL
  }
  combined
}

ldk_run_flexible_fixture <- function(
    work_dir,
    fixture = c("synthetic", "real"),
    ...) {
  fixture <- match.arg(fixture)
  spec <- .ldk_flexible_fixture_spec(fixture)
  data_path <- file.path(work_dir, spec$data_file)
  if (!file.exists(data_path)) {
    stop(sprintf("Fixture data not found: %s", data_path), call. = FALSE)
  }
  data <- utils::read.csv(data_path, check.names = FALSE)
  ldk_fit_flexible_models(
    data = data,
    dataset = spec$dataset,
    response = spec$response,
    features = spec$features,
    varying = spec$varying,
    categories = spec$categories,
    prediction_rows = spec$prediction_rows,
    ...
  )
}
