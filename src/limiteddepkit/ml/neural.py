"""Optional residual neural-network challenger for binary outcomes.

The estimator in this module is deliberately a *prediction challenger*, not an
econometric replacement.  It exposes no coefficient inference and therefore
sets ``inference_valid=False``.  PyTorch is imported lazily by :meth:`fit`, so
importing :mod:`limiteddepkit.ml.neural` never makes PyTorch a hard dependency.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

__all__ = [
    "NeuralTrainingEpoch",
    "ResidualBinaryMLP",
    "ResidualBinaryMLPResult",
]


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _finite_number(name: str, value: float, *, minimum: float, strict: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value.")
    numeric = float(value)
    if not np.isfinite(numeric) or (numeric <= minimum if strict else numeric < minimum):
        relation = "greater than" if strict else "greater than or equal to"
        raise ValueError(f"{name} must be finite and {relation} {minimum}.")
    return numeric


def _fit_design(X: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(X, pd.DataFrame):
        names = tuple(str(column) for column in X.columns)
        try:
            design = X.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain only numeric values.") from exc
    else:
        try:
            design = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain only numeric values.") from exc
        if design.ndim == 1:
            design = design.reshape(-1, 1)
        names = (
            tuple(f"x{column}" for column in range(design.shape[1]))
            if design.ndim == 2
            else ()
        )

    if design.ndim != 2:
        raise ValueError("X must be a two-dimensional array or DataFrame.")
    if design.shape[0] == 0 or design.shape[1] == 0:
        raise ValueError("X must contain at least one observation and one feature.")
    if len(set(names)) != len(names):
        raise ValueError("X feature names must be unique after conversion to strings.")
    if not np.isfinite(design).all():
        raise ValueError("X contains missing or non-finite values.")
    return design, names


def _prediction_design(
    X: Any,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, pd.Index]:
    design, names = _fit_design(X)
    if design.shape[1] != len(feature_names):
        raise ValueError(f"X has {design.shape[1]} columns; expected {len(feature_names)}.")
    if isinstance(X, pd.DataFrame) and names != feature_names:
        raise ValueError("DataFrame columns must match the fitted feature names and order.")
    index = X.index.copy() if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(design))
    return design, index


def _binary_outcome(y: Any, *, nobs: int) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if len(values) != nobs:
        raise ValueError("X and y must contain the same number of observations.")
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must contain numeric binary values 0 and 1.") from exc
    if not np.isfinite(numeric).all() or not np.all(np.isin(numeric, (0.0, 1.0))):
        raise ValueError("y must contain only finite binary values 0 and 1.")
    target = numeric.astype(np.float32)
    counts = np.bincount(target.astype(int), minlength=2)
    if np.any(counts < 2):
        raise ValueError(
            "Each outcome class needs at least two observations for the internal "
            "stratified validation split."
        )
    return target


def _stratified_validation_indices(
    target: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    validation_parts: list[np.ndarray] = []
    for category in (0, 1):
        category_rows = np.flatnonzero(target == category)
        shuffled = rng.permutation(category_rows)
        validation_count = int(round(len(category_rows) * validation_fraction))
        validation_count = min(max(validation_count, 1), len(category_rows) - 1)
        validation_parts.append(shuffled[:validation_count])
    validation = np.sort(np.concatenate(validation_parts)).astype(np.int64, copy=False)
    training_mask = np.ones(len(target), dtype=bool)
    training_mask[validation] = False
    training = np.flatnonzero(training_mask).astype(np.int64, copy=False)
    return training, validation


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "ResidualBinaryMLP is optional and requires PyTorch. Install it with "
            "`pip install limiteddepkit[neural]` before calling fit()."
        ) from exc


def _make_network(
    torch: Any,
    *,
    n_features: int,
    hidden_width: int,
    n_blocks: int,
    dropout: float,
) -> Any:
    nn = torch.nn

    class _ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(hidden_width)
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_width, 2 * hidden_width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_width, hidden_width),
                nn.Dropout(dropout),
            )

        def forward(self, values: Any) -> Any:
            return values + self.feed_forward(self.normalization(values))

    class _ResidualTabularBinaryNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Sequential(
                nn.Linear(n_features, hidden_width),
                nn.LayerNorm(hidden_width),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.ModuleList(_ResidualBlock() for _ in range(n_blocks))
            self.output = nn.Sequential(
                nn.LayerNorm(hidden_width),
                nn.Linear(hidden_width, 1),
            )
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        def forward(self, values: Any) -> Any:
            hidden = self.input_projection(values)
            for block in self.blocks:
                hidden = block(hidden)
            return self.output(hidden).squeeze(-1)

    return _ResidualTabularBinaryNetwork()


def _bernoulli_log_loss(target: np.ndarray, logits: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - target * logits))


def _temperature_from_validation(
    logits: np.ndarray,
    target: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    baseline = _bernoulli_log_loss(target, logits)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return _bernoulli_log_loss(target, logits / temperature)

    optimized = minimize_scalar(
        objective,
        bounds=(float(np.log(bounds[0])), float(np.log(bounds[1]))),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not optimized.success or not np.isfinite(optimized.fun) or optimized.fun >= baseline:
        return 1.0, baseline
    return float(np.exp(optimized.x)), float(optimized.fun)


@dataclass(frozen=True)
class NeuralTrainingEpoch:
    """Finite optimization diagnostics recorded after one epoch."""

    epoch: int
    training_loss: float
    validation_loss: float
    maximum_gradient_norm: float


@dataclass(frozen=True)
class ResidualBinaryMLPResult:
    """Fitted residual binary MLP and leakage-safe training diagnostics."""

    feature_names: tuple[str, ...]
    feature_means: pd.Series
    feature_scales: pd.Series
    nobs: int
    internal_training_indices: np.ndarray
    internal_validation_indices: np.ndarray
    history: tuple[NeuralTrainingEpoch, ...]
    training_completed: bool
    converged: bool
    best_epoch: int
    stopped_early: bool
    best_validation_loss: float
    calibrated_validation_loss: float
    temperature: float
    temperature_scaled: bool
    hidden_width: int
    n_blocks: int
    dropout: float
    positive_class_weight: float
    random_state: int
    _network: Any = field(repr=False, compare=False)
    _torch: Any = field(repr=False, compare=False)

    @property
    def inference_valid(self) -> bool:
        """Return false: a neural prediction challenger has no ordinary inference."""
        return False

    @property
    def backend(self) -> str:
        return "pytorch-experimental"

    @property
    def covariance_type(self) -> str:
        return "none"

    @property
    def n_epochs(self) -> int:
        return len(self.history)

    def training_history_frame(self) -> pd.DataFrame:
        """Return one row per training epoch without exposing mutable internals."""
        return pd.DataFrame(
            [
                {
                    "epoch": item.epoch,
                    "training_loss": item.training_loss,
                    "validation_loss": item.validation_loss,
                    "maximum_gradient_norm": item.maximum_gradient_norm,
                }
                for item in self.history
            ]
        )

    def diagnostics(self) -> pd.Series:
        """Return compact convergence and calibration diagnostics."""
        return pd.Series(
            {
                "training_completed": self.training_completed,
                "converged": self.converged,
                "n_epochs": self.n_epochs,
                "best_epoch": self.best_epoch,
                "stopped_early": self.stopped_early,
                "best_validation_loss": self.best_validation_loss,
                "calibrated_validation_loss": self.calibrated_validation_loss,
                "temperature": self.temperature,
                "internal_training_n": len(self.internal_training_indices),
                "internal_validation_n": len(self.internal_validation_indices),
            },
            name="diagnostic",
        )

    def _standardized_tensor(self, X: Any) -> tuple[Any, pd.Index]:
        design, index = _prediction_design(X, self.feature_names)
        standardized = (
            design - self.feature_means.to_numpy(dtype=float)
        ) / self.feature_scales.to_numpy(dtype=float)
        parameter = next(self._network.parameters())
        return parameter.new_tensor(standardized), index

    def _event_probability(self, X: Any, *, training: bool) -> tuple[np.ndarray, pd.Index]:
        tensor, index = self._standardized_tensor(X)
        previous_mode = bool(self._network.training)
        self._network.train(training)
        try:
            with self._torch.no_grad():
                logits = self._network(tensor) / self.temperature
                probability = logits.sigmoid().detach().cpu().numpy().astype(float)
        finally:
            self._network.train(previous_mode)
        return probability, index

    def predict_proba(self, X: Any) -> pd.DataFrame:
        """Return calibrated class probabilities with input row labels preserved."""
        probability, index = self._event_probability(X, training=False)
        return pd.DataFrame({0: 1.0 - probability, 1: probability}, index=index)

    def predict(self, X: Any, *, threshold: float = 0.5) -> pd.Series:
        """Return thresholded labels with input row labels preserved."""
        threshold = _finite_number("threshold", threshold, minimum=0.0, strict=True)
        if threshold >= 1.0:
            raise ValueError("threshold must be strictly between zero and one.")
        probabilities = self.predict_proba(X)[1]
        return (probabilities >= threshold).astype(int).rename("prediction")

    def mc_dropout_probabilities(
        self,
        X: Any,
        *,
        n_draws: int = 100,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Return event-probability draws with dropout active at prediction time.

        These draws describe approximate model uncertainty conditional on the
        fitted weights.  They are not confidence intervals and do not include
        sampling, specification, or data uncertainty.
        """
        n_draws = _positive_integer("n_draws", n_draws)
        if random_state is None:
            seed = self.random_state
        elif isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise ValueError("random_state must be a non-negative integer or None.")
        else:
            seed = int(random_state)
            if seed < 0:
                raise ValueError("random_state must be a non-negative integer or None.")

        tensor, index = self._standardized_tensor(X)
        parameter = next(self._network.parameters())
        devices: list[int] = []
        if parameter.device.type == "cuda":
            devices = [
                int(parameter.device.index)
                if parameter.device.index is not None
                else int(self._torch.cuda.current_device())
            ]
        previous_mode = bool(self._network.training)
        draws = np.empty((len(index), n_draws), dtype=float)
        try:
            with self._torch.random.fork_rng(devices=devices, enabled=True):
                self._torch.manual_seed(seed)
                if devices:
                    self._torch.cuda.manual_seed_all(seed)
                self._network.train(True)
                with self._torch.no_grad():
                    for draw in range(n_draws):
                        logits = self._network(tensor) / self.temperature
                        draws[:, draw] = logits.sigmoid().detach().cpu().numpy()
        finally:
            self._network.train(previous_mode)
        return pd.DataFrame(
            draws,
            index=index,
            columns=pd.RangeIndex(n_draws, name="draw"),
        )

    def predict_proba_uncertainty(
        self,
        X: Any,
        *,
        n_draws: int = 100,
        level: float = 0.95,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Summarize Monte Carlo-dropout uncertainty for the event probability."""
        level = _finite_number("level", level, minimum=0.0, strict=True)
        if level >= 1.0:
            raise ValueError("level must be strictly between zero and one.")
        draws = self.mc_dropout_probabilities(
            X,
            n_draws=n_draws,
            random_state=random_state,
        )
        tail = (1.0 - level) / 2.0
        values = draws.to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "probability_mean": np.mean(values, axis=1),
                "probability_std": np.std(values, axis=1, ddof=0),
                "probability_lower": np.quantile(values, tail, axis=1),
                "probability_upper": np.quantile(values, 1.0 - tail, axis=1),
                "n_draws": values.shape[1],
            },
            index=draws.index,
        )


class ResidualBinaryMLP:
    """Small-sample residual MLP challenger for binary probability prediction.

    Every call to :meth:`fit` creates a deterministic, stratified internal
    validation split.  Standardization is estimated from the internal training
    rows only; validation rows are used for early stopping and, optionally,
    scalar temperature calibration.  Hyperparameter selection must still occur
    in an outer/nested cross-validation workflow.  The default unweighted loss
    targets event probabilities.  Opt-in class weighting changes the training
    target to a cost-weighted probability, and scalar temperature scaling need
    not undo the resulting prior/intercept shift.
    """

    def __init__(
        self,
        *,
        hidden_width: int = 32,
        n_blocks: int = 2,
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 300,
        validation_fraction: float = 0.2,
        patience: int = 30,
        min_delta: float = 1e-4,
        gradient_clip_norm: float = 5.0,
        positive_class_weight: str | float | None = None,
        temperature_scaling: bool = True,
        temperature_bounds: tuple[float, float] = (0.05, 20.0),
        random_state: int = 0,
        device: str = "cpu",
    ) -> None:
        self.hidden_width = _positive_integer("hidden_width", hidden_width)
        self.n_blocks = _positive_integer("n_blocks", n_blocks)
        self.dropout = _finite_number("dropout", dropout, minimum=0.0, strict=False)
        if self.dropout >= 1.0:
            raise ValueError("dropout must be smaller than one.")
        self.learning_rate = _finite_number(
            "learning_rate", learning_rate, minimum=0.0, strict=True
        )
        self.weight_decay = _finite_number(
            "weight_decay", weight_decay, minimum=0.0, strict=False
        )
        self.batch_size = _positive_integer("batch_size", batch_size)
        self.max_epochs = _positive_integer("max_epochs", max_epochs)
        self.validation_fraction = _finite_number(
            "validation_fraction", validation_fraction, minimum=0.0, strict=True
        )
        if self.validation_fraction >= 1.0:
            raise ValueError("validation_fraction must be strictly between zero and one.")
        self.patience = _positive_integer("patience", patience)
        self.min_delta = _finite_number("min_delta", min_delta, minimum=0.0, strict=False)
        self.gradient_clip_norm = _finite_number(
            "gradient_clip_norm", gradient_clip_norm, minimum=0.0, strict=True
        )
        if positive_class_weight is None:
            self.positive_class_weight: str | float | None = None
        elif isinstance(positive_class_weight, str):
            if positive_class_weight != "balanced":
                raise ValueError("positive_class_weight must be 'balanced', numeric, or None.")
            self.positive_class_weight = positive_class_weight
        else:
            self.positive_class_weight = _finite_number(
                "positive_class_weight",
                positive_class_weight,
                minimum=0.0,
                strict=True,
            )
        if not isinstance(temperature_scaling, bool):
            raise ValueError("temperature_scaling must be a boolean.")
        self.temperature_scaling = temperature_scaling
        if not isinstance(temperature_bounds, tuple) or len(temperature_bounds) != 2:
            raise ValueError("temperature_bounds must be a (lower, upper) tuple.")
        lower = _finite_number(
            "temperature_bounds[0]", temperature_bounds[0], minimum=0.0, strict=True
        )
        upper = _finite_number(
            "temperature_bounds[1]", temperature_bounds[1], minimum=lower, strict=True
        )
        self.temperature_bounds = (lower, upper)
        if isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise ValueError("random_state must be a non-negative integer.")
        self.random_state = int(random_state)
        if self.random_state < 0:
            raise ValueError("random_state must be a non-negative integer.")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty PyTorch device string.")
        self.device = device.strip()

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        entity: Any | None = None,
        time: Any | None = None,
    ) -> ResidualBinaryMLPResult:
        """Fit the independent-row challenger and return held-out diagnostics.

        The internal validation split is iid-stratified.  Panel/entity and time-
        ordered data are therefore rejected instead of being silently split in a
        leakage-prone way.  Such data need a future group/time-aware neural trainer.
        """
        supplied_metadata = [
            name for name, value in (("entity", entity), ("time", time)) if value is not None
        ]
        if supplied_metadata:
            names = " and ".join(supplied_metadata)
            raise ValueError(
                "ResidualBinaryMLP currently supports independent rows only; "
                f"{names} metadata cannot be used because its internal validation "
                "split is iid-stratified."
            )
        design, feature_names = _fit_design(X)
        target = _binary_outcome(y, nobs=len(design))
        training_rows, validation_rows = _stratified_validation_indices(
            target,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )
        means = np.mean(design[training_rows], axis=0)
        scales = np.std(design[training_rows], axis=0, ddof=0)
        scales = np.where(scales > np.finfo(float).eps, scales, 1.0)
        standardized = ((design - means) / scales).astype(np.float32, copy=False)

        torch = _load_torch()
        try:
            device = torch.device(self.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"Requested PyTorch device {self.device!r} is unavailable.")
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid or unavailable PyTorch device {self.device!r}.") from exc

        devices: list[int] = []
        if device.type == "cuda":
            devices = [
                int(device.index)
                if device.index is not None
                else int(torch.cuda.current_device())
            ]
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(self.random_state)
            if devices:
                torch.cuda.manual_seed_all(self.random_state)
            network = _make_network(
                torch,
                n_features=design.shape[1],
                hidden_width=self.hidden_width,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
            ).to(device)

            features = torch.as_tensor(standardized, dtype=torch.float32, device=device)
            outcomes = torch.as_tensor(target, dtype=torch.float32, device=device)
            if self.positive_class_weight == "balanced":
                training_target = target[training_rows]
                positive_weight = float(
                    np.count_nonzero(training_target == 0)
                    / np.count_nonzero(training_target == 1)
                )
            elif self.positive_class_weight is None:
                positive_weight = 1.0
            else:
                positive_weight = float(self.positive_class_weight)
            loss_function = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.as_tensor(positive_weight, dtype=torch.float32, device=device)
            )
            optimizer = torch.optim.AdamW(
                network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            rng = np.random.default_rng(self.random_state)
            history: list[NeuralTrainingEpoch] = []
            best_loss = np.inf
            best_epoch = 0
            epochs_without_improvement = 0
            patience_satisfied = False
            best_state: dict[str, Any] | None = None

            for epoch in range(1, self.max_epochs + 1):
                network.train(True)
                epoch_order = rng.permutation(training_rows)
                total_loss = 0.0
                total_rows = 0
                maximum_gradient_norm = 0.0
                for start in range(0, len(epoch_order), self.batch_size):
                    rows = epoch_order[start : start + self.batch_size]
                    row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = network(features.index_select(0, row_tensor))
                    loss = loss_function(logits, outcomes.index_select(0, row_tensor))
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Neural training produced a non-finite loss.")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        network.parameters(), self.gradient_clip_norm
                    )
                    numeric_gradient_norm = float(gradient_norm.detach().cpu())
                    if not np.isfinite(numeric_gradient_norm):
                        raise RuntimeError("Neural training produced non-finite gradients.")
                    maximum_gradient_norm = max(maximum_gradient_norm, numeric_gradient_norm)
                    optimizer.step()
                    total_loss += float(loss.detach().cpu()) * len(rows)
                    total_rows += len(rows)

                network.eval()
                validation_tensor = torch.as_tensor(
                    validation_rows, dtype=torch.long, device=device
                )
                with torch.no_grad():
                    validation_logits = network(
                        features.index_select(0, validation_tensor)
                    )
                    validation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        validation_logits,
                        outcomes.index_select(0, validation_tensor),
                    )
                numeric_validation_loss = float(validation_loss.detach().cpu())
                if not np.isfinite(numeric_validation_loss):
                    raise RuntimeError("Neural validation produced a non-finite loss.")
                history.append(
                    NeuralTrainingEpoch(
                        epoch=epoch,
                        training_loss=float(total_loss / total_rows),
                        validation_loss=numeric_validation_loss,
                        maximum_gradient_norm=maximum_gradient_norm,
                    )
                )
                if numeric_validation_loss < best_loss - self.min_delta:
                    best_loss = numeric_validation_loss
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().clone()
                        for name, value in network.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    patience_satisfied = True
                    break

            if best_state is None or best_epoch == 0:  # pragma: no cover - defensive
                raise RuntimeError("Neural training did not produce a finite validation model.")
            network.load_state_dict(best_state)
            network.eval()
            with torch.no_grad():
                best_validation_logits = (
                    network(features.index_select(0, validation_tensor))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )

        if self.temperature_scaling:
            temperature, calibrated_loss = _temperature_from_validation(
                best_validation_logits,
                target[validation_rows].astype(float),
                self.temperature_bounds,
            )
        else:
            temperature = 1.0
            calibrated_loss = _bernoulli_log_loss(
                target[validation_rows].astype(float), best_validation_logits
            )

        stopped_early = patience_satisfied and len(history) < self.max_epochs
        return ResidualBinaryMLPResult(
            feature_names=feature_names,
            feature_means=pd.Series(means, index=feature_names, name="mean"),
            feature_scales=pd.Series(scales, index=feature_names, name="scale"),
            nobs=len(design),
            internal_training_indices=training_rows.copy(),
            internal_validation_indices=validation_rows.copy(),
            history=tuple(history),
            training_completed=True,
            # ``converged`` is intentionally conservative because the generic
            # validation layer treats it as an eligibility gate.  A finite
            # checkpoint at the epoch limit remains usable for prediction, but
            # only the explicit patience criterion counts as stabilization.
            converged=patience_satisfied,
            best_epoch=best_epoch,
            stopped_early=stopped_early,
            best_validation_loss=float(best_loss),
            calibrated_validation_loss=float(calibrated_loss),
            temperature=float(temperature),
            temperature_scaled=self.temperature_scaling,
            hidden_width=self.hidden_width,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            positive_class_weight=positive_weight,
            random_state=self.random_state,
            _network=network,
            _torch=torch,
        )
