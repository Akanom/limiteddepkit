"""External-reference validation for the experimental ML workflow layer."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from limiteddepkit import BinaryLogit
from limiteddepkit.ml import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    binary_accuracy,
    binary_balanced_accuracy,
    binary_brier_score,
    binary_log_loss,
    binary_roc_auc,
    choice_accuracy,
    choice_brier_score,
    choice_log_loss,
    continuous_mean_absolute_error,
    continuous_root_mean_squared_error,
    count_mean_absolute_error,
    count_root_mean_squared_error,
    cross_validate,
    multiclass_brier_score,
    multiclass_log_loss,
    ordinal_mean_absolute_error,
    poisson_deviance,
    quantile_check_loss,
)

sklearn_metrics = pytest.importorskip("sklearn.metrics")
sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
sklearn_linear_model = pytest.importorskip("sklearn.linear_model")
statsmodels_api = pytest.importorskip("statsmodels.api")

pytestmark = pytest.mark.validation


def _randomized_metric_audit(*, seed=20260714, repetitions=500):
    rng = np.random.default_rng(seed)
    maximum_differences = {}

    def record(name, native, reference):
        difference = float(abs(float(native) - float(reference)))
        maximum_differences[name] = max(
            maximum_differences.get(name, 0.0), difference
        )

    quantiles = (0.05, 0.20, 0.50, 0.80, 0.95)
    for _ in range(repetitions):
        nobs = int(rng.integers(40, 121))
        binary_target = rng.integers(0, 2, size=nobs)
        binary_target[:2] = [0, 1]
        binary_probability = rng.uniform(0.01, 0.99, size=nobs)
        binary_prediction = (binary_probability >= 0.5).astype(int)
        record(
            "binary_log_loss",
            binary_log_loss(binary_target, binary_probability),
            sklearn_metrics.log_loss(
                binary_target, binary_probability, labels=[0, 1]
            ),
        )
        record(
            "binary_brier_score",
            binary_brier_score(binary_target, binary_probability),
            sklearn_metrics.brier_score_loss(binary_target, binary_probability),
        )
        record(
            "binary_accuracy",
            binary_accuracy(binary_target, binary_probability),
            sklearn_metrics.accuracy_score(binary_target, binary_prediction),
        )
        record(
            "binary_balanced_accuracy",
            binary_balanced_accuracy(binary_target, binary_probability),
            sklearn_metrics.balanced_accuracy_score(
                binary_target, binary_prediction
            ),
        )
        record(
            "binary_roc_auc",
            binary_roc_auc(binary_target, binary_probability),
            sklearn_metrics.roc_auc_score(binary_target, binary_probability),
        )

        n_classes = int(rng.integers(3, 7))
        multiclass_target = np.concatenate(
            [
                np.arange(n_classes),
                rng.integers(0, n_classes, size=nobs - n_classes),
            ]
        )
        rng.shuffle(multiclass_target)
        probability_weights = rng.gamma(1.5, 1.0, size=(nobs, n_classes))
        multiclass_probability = probability_weights / probability_weights.sum(
            axis=1, keepdims=True
        )
        observed = np.eye(n_classes)[multiclass_target]
        predicted_class = np.argmax(multiclass_probability, axis=1)
        record(
            "multiclass_log_loss",
            multiclass_log_loss(multiclass_target, multiclass_probability),
            sklearn_metrics.log_loss(
                multiclass_target,
                multiclass_probability,
                labels=np.arange(n_classes),
            ),
        )
        reference_brier = n_classes * sklearn_metrics.mean_squared_error(
            observed, multiclass_probability
        )
        record(
            "multiclass_brier_score",
            multiclass_brier_score(multiclass_target, multiclass_probability),
            reference_brier,
        )
        record(
            "ordinal_mean_absolute_error",
            ordinal_mean_absolute_error(
                multiclass_target,
                predicted_class,
                labels=np.arange(n_classes),
            ),
            sklearn_metrics.mean_absolute_error(
                multiclass_target, predicted_class
            ),
        )

        groups = np.repeat(np.arange(nobs), n_classes)
        choice = observed.astype(int).reshape(-1)
        choice_probability = multiclass_probability.reshape(-1)
        record(
            "choice_log_loss",
            choice_log_loss(choice, choice_probability, groups),
            sklearn_metrics.log_loss(
                multiclass_target,
                multiclass_probability,
                labels=np.arange(n_classes),
            ),
        )
        record(
            "choice_brier_score",
            choice_brier_score(choice, choice_probability, groups),
            reference_brier,
        )
        record(
            "choice_accuracy",
            choice_accuracy(choice, choice_probability, groups),
            sklearn_metrics.accuracy_score(multiclass_target, predicted_class),
        )

        continuous_target = rng.normal(size=nobs)
        continuous_prediction = continuous_target + rng.normal(scale=0.7, size=nobs)
        record(
            "continuous_mean_absolute_error",
            continuous_mean_absolute_error(
                continuous_target, continuous_prediction
            ),
            sklearn_metrics.mean_absolute_error(
                continuous_target, continuous_prediction
            ),
        )
        record(
            "continuous_root_mean_squared_error",
            continuous_root_mean_squared_error(
                continuous_target, continuous_prediction
            ),
            sklearn_metrics.root_mean_squared_error(
                continuous_target, continuous_prediction
            ),
        )

        count_target = rng.poisson(rng.uniform(0.2, 8.0, size=nobs))
        count_mean = np.exp(rng.normal(scale=0.8, size=nobs))
        record(
            "count_mean_absolute_error",
            count_mean_absolute_error(count_target, count_mean),
            sklearn_metrics.mean_absolute_error(count_target, count_mean),
        )
        record(
            "count_root_mean_squared_error",
            count_root_mean_squared_error(count_target, count_mean),
            sklearn_metrics.root_mean_squared_error(count_target, count_mean),
        )
        record(
            "poisson_deviance",
            poisson_deviance(count_target, count_mean),
            sklearn_metrics.mean_poisson_deviance(count_target, count_mean),
        )
        for quantile in quantiles:
            record(
                f"quantile_check_loss_{quantile:.2f}",
                quantile_check_loss(
                    continuous_target,
                    continuous_prediction,
                    quantile=quantile,
                ),
                sklearn_metrics.mean_pinball_loss(
                    continuous_target,
                    continuous_prediction,
                    alpha=quantile,
                ),
            )
    return maximum_differences


def test_randomized_prediction_metric_audit_matches_sklearn():
    maximum_differences = _randomized_metric_audit()

    assert max(maximum_differences.values()) <= 5e-14, maximum_differences


def test_prediction_metrics_match_sklearn_reference_definitions():
    binary_target = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    binary_probability = np.array([0.08, 0.84, 0.61, 0.64, 0.47, 0.42, 0.58, 0.58])
    binary_prediction = (binary_probability >= 0.5).astype(int)

    assert binary_log_loss(binary_target, binary_probability) == pytest.approx(
        sklearn_metrics.log_loss(binary_target, binary_probability, labels=[0, 1]),
        abs=1e-15,
    )
    assert binary_brier_score(binary_target, binary_probability) == pytest.approx(
        sklearn_metrics.brier_score_loss(binary_target, binary_probability),
        abs=1e-15,
    )
    assert binary_accuracy(binary_target, binary_probability) == pytest.approx(
        sklearn_metrics.accuracy_score(binary_target, binary_prediction),
        abs=0.0,
    )
    assert binary_balanced_accuracy(binary_target, binary_probability) == pytest.approx(
        sklearn_metrics.balanced_accuracy_score(binary_target, binary_prediction),
        abs=0.0,
    )
    assert binary_roc_auc(binary_target, binary_probability) == pytest.approx(
        sklearn_metrics.roc_auc_score(binary_target, binary_probability),
        abs=1e-15,
    )

    multiclass_target = np.array([0, 2, 1, 0, 1, 2])
    multiclass_probability = np.array(
        [
            [0.72, 0.18, 0.10],
            [0.11, 0.50, 0.39],
            [0.45, 0.35, 0.20],
            [0.33, 0.34, 0.33],
            [0.22, 0.39, 0.39],
            [0.09, 0.50, 0.41],
        ]
    )
    observed = np.eye(3)[multiclass_target]
    assert multiclass_log_loss(
        multiclass_target, multiclass_probability
    ) == pytest.approx(
        sklearn_metrics.log_loss(
            multiclass_target,
            multiclass_probability,
            labels=[0, 1, 2],
        ),
        abs=1e-15,
    )
    # sklearn's multi-output MSE averages over class columns; multiplying by
    # the class count gives the standard unscaled multiclass Brier score.
    sklearn_multiclass_brier = 3 * sklearn_metrics.mean_squared_error(
        observed, multiclass_probability
    )
    assert multiclass_brier_score(
        multiclass_target, multiclass_probability
    ) == pytest.approx(sklearn_multiclass_brier, abs=1e-15)
    predicted_class = np.argmax(multiclass_probability, axis=1)
    assert ordinal_mean_absolute_error(
        multiclass_target,
        predicted_class,
        labels=[0, 1, 2],
    ) == pytest.approx(
        sklearn_metrics.mean_absolute_error(multiclass_target, predicted_class),
        abs=0.0,
    )

    groups = np.repeat(np.arange(len(multiclass_target)), 3)
    choice = observed.astype(int).reshape(-1)
    choice_probability = multiclass_probability.reshape(-1)
    assert choice_log_loss(choice, choice_probability, groups) == pytest.approx(
        sklearn_metrics.log_loss(
            multiclass_target,
            multiclass_probability,
            labels=[0, 1, 2],
        ),
        abs=1e-15,
    )
    assert choice_brier_score(choice, choice_probability, groups) == pytest.approx(
        sklearn_multiclass_brier,
        abs=1e-15,
    )
    assert choice_accuracy(choice, choice_probability, groups) == pytest.approx(
        sklearn_metrics.accuracy_score(multiclass_target, predicted_class),
        abs=0.0,
    )

    continuous_target = np.array([-1.2, -0.1, 0.4, 1.3, 2.2])
    continuous_prediction = np.array([-0.9, -0.3, 0.6, 1.0, 2.5])
    assert continuous_mean_absolute_error(
        continuous_target, continuous_prediction
    ) == pytest.approx(
        sklearn_metrics.mean_absolute_error(continuous_target, continuous_prediction),
        abs=0.0,
    )
    assert continuous_root_mean_squared_error(
        continuous_target, continuous_prediction
    ) == pytest.approx(
        sklearn_metrics.root_mean_squared_error(
            continuous_target, continuous_prediction
        ),
        abs=0.0,
    )

    count_target = np.array([0, 1, 4, 2, 0, 6])
    count_mean = np.array([0.4, 1.3, 3.5, 2.4, 0.7, 5.2])
    assert count_mean_absolute_error(count_target, count_mean) == pytest.approx(
        sklearn_metrics.mean_absolute_error(count_target, count_mean),
        abs=0.0,
    )
    assert count_root_mean_squared_error(count_target, count_mean) == pytest.approx(
        sklearn_metrics.root_mean_squared_error(count_target, count_mean),
        abs=0.0,
    )
    assert poisson_deviance(count_target, count_mean) == pytest.approx(
        sklearn_metrics.mean_poisson_deviance(count_target, count_mean),
        abs=1e-15,
    )
    assert quantile_check_loss(
        continuous_target,
        continuous_prediction,
        quantile=0.35,
    ) == pytest.approx(
        sklearn_metrics.mean_pinball_loss(
            continuous_target,
            continuous_prediction,
            alpha=0.35,
        ),
        abs=1e-15,
    )


def test_log_loss_endpoint_policy_matches_when_clipping_epsilon_is_aligned():
    target = np.array([0, 1])
    completely_wrong_probability = np.array([1.0, 0.0])
    reference = sklearn_metrics.log_loss(
        target,
        completely_wrong_probability,
        labels=[0, 1],
    )

    assert binary_log_loss(
        target,
        completely_wrong_probability,
        eps=np.finfo(float).eps,
    ) == pytest.approx(reference, abs=1e-14)
    assert binary_log_loss(target, completely_wrong_probability) < reference


def _assert_same_splits(native_splits, reference_splits):
    assert len(native_splits) == len(reference_splits)
    for (native_train, native_test), (reference_train, reference_test) in zip(
        native_splits, reference_splits, strict=True
    ):
        np.testing.assert_array_equal(native_train, reference_train)
        np.testing.assert_array_equal(native_test, reference_test)


def test_unshuffled_iid_and_stratified_folds_match_sklearn_indices():
    X = np.zeros((23, 2))
    _assert_same_splits(
        list(KFold(4).split(X)),
        list(sklearn_model_selection.KFold(4).split(X)),
    )

    y = np.repeat(["a", "b", "c"], [7, 5, 4])
    X = np.zeros((len(y), 2))
    _assert_same_splits(
        list(StratifiedKFold(4).split(X, y)),
        list(sklearn_model_selection.StratifiedKFold(4).split(X, y)),
    )


def test_500_randomized_unshuffled_splitter_designs_match_sklearn():
    rng = np.random.default_rng(20260714)

    for _ in range(500):
        n_splits = int(rng.integers(2, 9))
        nobs = int(rng.integers(n_splits, 200))
        X = np.zeros((nobs, 1))
        _assert_same_splits(
            list(KFold(n_splits).split(X)),
            list(sklearn_model_selection.KFold(n_splits).split(X)),
        )

        n_classes = int(rng.integers(2, 8))
        class_counts = rng.integers(
            n_splits,
            n_splits + 30,
            size=n_classes,
        )
        y = np.concatenate(
            [
                np.repeat(f"class-{category}", int(count))
                for category, count in enumerate(class_counts)
            ]
        )
        y = rng.permutation(y)
        X = np.zeros((len(y), 1))
        native = list(StratifiedKFold(n_splits).split(X, y))
        reference = list(
            sklearn_model_selection.StratifiedKFold(n_splits).split(X, y)
        )
        _assert_same_splits(native, reference)

        fold_sizes = [len(test) for _, test in native]
        assert max(fold_sizes) - min(fold_sizes) <= 1
        for label in np.unique(y):
            per_fold = [int(np.count_nonzero(y[test] == label)) for _, test in native]
            assert max(per_fold) - min(per_fold) <= 1


def test_group_folds_match_sklearn_balance_without_splitting_groups():
    groups = np.repeat(np.arange(8), np.arange(1, 9))
    X = np.zeros((len(groups), 2))
    native = list(GroupKFold(4).split(X, groups=groups))
    reference = list(sklearn_model_selection.GroupKFold(4).split(X, groups=groups))

    assert sorted(len(test) for _, test in native) == sorted(
        len(test) for _, test in reference
    )
    for train, test in native:
        assert set(groups[train]).isdisjoint(set(groups[test]))


def test_binary_cross_validation_matches_statsmodels_and_unpenalized_sklearn():
    rng = np.random.default_rng(20260714)
    nobs = 600
    X = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.normal(size=nobs),
            "x2": rng.normal(size=nobs),
        }
    )
    probability = expit(X.to_numpy() @ np.array([-0.25, 0.7, -0.45]))
    y = rng.binomial(1, probability)

    splitter = KFold(4)
    native = cross_validate(
        BinaryLogit,
        X,
        y,
        splitter=splitter,
        outcome="binary",
        fit_kwargs={"tolerance": 1e-7},
    )
    reference_splits = list(sklearn_model_selection.KFold(4).split(X))
    reference_oof = np.empty(nobs)

    for fold, (train, test) in zip(native.folds, reference_splits, strict=True):
        np.testing.assert_array_equal(fold.train_index, train)
        np.testing.assert_array_equal(fold.test_index, test)

        statsmodels_result = statsmodels_api.Logit(y[train], X.iloc[train]).fit(
            method="newton",
            maxiter=1_000,
            tol=1e-12,
            disp=False,
        )
        statsmodels_probability = np.asarray(statsmodels_result.predict(X.iloc[test]))
        reference_oof[test] = statsmodels_probability

        sklearn_result = sklearn_linear_model.LogisticRegression(
            C=np.inf,
            fit_intercept=False,
            solver="lbfgs",
            max_iter=2_000,
            tol=1e-12,
        ).fit(X.iloc[train], y[train])
        sklearn_probability = sklearn_result.predict_proba(X.iloc[test])[:, 1]

        np.testing.assert_allclose(
            fold.result.params.to_numpy(),
            np.asarray(statsmodels_result.params),
            rtol=0.0,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            np.asarray(fold.prediction[1]),
            statsmodels_probability,
            rtol=0.0,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            np.asarray(fold.prediction[1]),
            sklearn_probability,
            rtol=0.0,
            atol=2e-5,
        )
        assert fold.metrics["log_loss"] == pytest.approx(
            sklearn_metrics.log_loss(y[test], statsmodels_probability, labels=[0, 1]),
            abs=2e-9,
        )
        assert fold.metrics["brier_score"] == pytest.approx(
            sklearn_metrics.brier_score_loss(y[test], statsmodels_probability),
            abs=2e-9,
        )
        assert fold.metrics["roc_auc"] == pytest.approx(
            sklearn_metrics.roc_auc_score(y[test], statsmodels_probability),
            abs=2e-9,
        )

    native_oof = (
        native.out_of_fold_predictions()
        .set_index("row_index")
        .sort_index()["prediction_1"]
        .to_numpy()
    )
    np.testing.assert_allclose(native_oof, reference_oof, rtol=0.0, atol=2e-8)
