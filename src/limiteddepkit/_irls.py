"""Fast damped Newton/IRLS optimization for convex likelihoods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult


def damped_newton(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    information: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    *,
    maxiter: int,
    tolerance: float,
) -> OptimizeResult:
    """Minimize a smooth convex objective with an analytical Hessian.

    A backtracking Armijo step prevents the unconstrained Newton update from
    walking into overflow regions.  The return value follows SciPy's
    ``OptimizeResult`` contract so existing result objects retain their public
    optimizer diagnostics.
    """
    parameters = np.asarray(initial, dtype=float).copy()
    value = float(objective(parameters))
    nfev = 1
    njev = 0
    message = "Maximum iterations reached."
    success = False
    iterations = 0

    if not np.isfinite(value):
        return OptimizeResult(
            x=parameters,
            fun=value,
            jac=np.full_like(parameters, np.nan),
            success=False,
            status=2,
            message="Initial objective is non-finite.",
            nit=0,
            nfev=nfev,
            njev=njev,
        )

    for iteration in range(1, maxiter + 1):
        iterations = iteration
        score = np.asarray(gradient(parameters), dtype=float)
        njev += 1
        if not np.isfinite(score).all():
            message = "Analytical gradient became non-finite."
            break
        score_norm = float(np.linalg.norm(score, ord=np.inf))
        if score_norm <= tolerance:
            success = True
            message = "Converged: analytical score tolerance satisfied."
            break

        hessian = np.asarray(information(parameters), dtype=float)
        if not np.isfinite(hessian).all():
            message = "Analytical information matrix became non-finite."
            break
        hessian = 0.5 * (hessian + hessian.T)
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            message = "Analytical information matrix is singular."
            break
        if not np.isfinite(step).all():
            message = "Newton step became non-finite."
            break

        directional = float(score @ step)
        if not np.isfinite(directional) or directional <= 0.0:
            message = "Newton direction is not a finite descent direction."
            break

        scale = 1.0
        accepted = False
        candidate = parameters
        candidate_value = value
        for _ in range(40):
            candidate = parameters - scale * step
            candidate_value = float(objective(candidate))
            nfev += 1
            if np.isfinite(candidate_value) and candidate_value <= (
                value - 1e-4 * scale * directional
            ):
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            message = "Backtracking line search failed."
            break

        parameters = candidate
        value = candidate_value
        scaled_step = scale * step
        if float(np.linalg.norm(scaled_step, ord=np.inf)) <= tolerance * (
            1.0 + float(np.linalg.norm(parameters, ord=np.inf))
        ):
            final_score = np.asarray(gradient(parameters), dtype=float)
            njev += 1
            if np.isfinite(final_score).all() and float(
                np.linalg.norm(final_score, ord=np.inf)
            ) <= max(10.0 * tolerance, 1e-7):
                success = True
                message = "Converged: Newton step and score tolerances satisfied."
                break

    final_score = np.asarray(gradient(parameters), dtype=float)
    njev += 1
    if not success and np.isfinite(final_score).all() and float(
        np.linalg.norm(final_score, ord=np.inf)
    ) <= max(10.0 * tolerance, 1e-7):
        success = True
        message = "Converged: final analytical score tolerance satisfied."

    return OptimizeResult(
        x=parameters,
        fun=float(value),
        jac=final_score,
        success=success,
        status=0 if success else 1,
        message=message,
        nit=iterations,
        nfev=nfev,
        njev=njev,
        method="damped-newton-irls",
    )
