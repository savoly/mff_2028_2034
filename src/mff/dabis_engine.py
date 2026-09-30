from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DegressionBand:
    """One marginal degression band on gross DABIS per farm."""

    lower: float
    upper: float | None
    reduction_rate: float
    name: str | None = None

    def __post_init__(self) -> None:
        if self.lower < 0:
            raise ValueError("lower must be >= 0")
        if self.upper is not None and self.upper <= self.lower:
            raise ValueError("upper must be greater than lower")
        if not 0 <= self.reduction_rate <= 1:
            raise ValueError("reduction_rate must be in [0, 1]")

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.upper is None:
            return f"{self.lower:g}+"
        return f"{self.lower:g}-{self.upper:g}"


DEFAULT_DABIS_BANDS: tuple[DegressionBand, ...] = (
    DegressionBand(0, 20_000, 0.00, "0_20"),
    DegressionBand(20_000, 50_000, 0.25, "20_50"),
    DegressionBand(50_000, 75_000, 0.50, "50_75"),
    DegressionBand(75_000, None, 0.75, "75_plus"),
)


@dataclass(frozen=True)
class DabisPolicy:
    """Configuration of the farm-level DABIS degression mechanism."""

    bands: Sequence[DegressionBand] = DEFAULT_DABIS_BANDS
    payment_cap: float | None = 100_000.0

    def __post_init__(self) -> None:
        _validate_bands(self.bands)
        if self.payment_cap is not None and self.payment_cap < 0:
            raise ValueError("payment_cap must be >= 0 or None")


def _validate_bands(bands: Sequence[DegressionBand]) -> None:
    if not bands:
        raise ValueError("At least one degression band is required")

    ordered = sorted(bands, key=lambda b: b.lower)
    if list(bands) != ordered:
        raise ValueError("Degression bands must be sorted by lower bound")

    for previous, current in pairwise(bands):
        if previous.upper is None:
            raise ValueError("Only the final band may have upper=None")
        if not np.isclose(previous.upper, current.lower):
            raise ValueError("Degression bands must be contiguous")


DEFAULT_DABIS_POLICY = DabisPolicy()


def gross_at_payment_cap(policy: DabisPolicy = DEFAULT_DABIS_POLICY) -> float | None:
    """Return the gross DABIS amount where the post-degression payment reaches the cap.

    The result is the gross-support threshold that triggers capping.  It is derived
    from the configured marginal degression bands, so it changes automatically when
    either a reduction rate, a band boundary, or the payment cap changes.
    """
    if policy.payment_cap is None:
        return None

    target = float(policy.payment_cap)
    if target <= 0:
        return 0.0

    paid_before_band = 0.0

    for band in policy.bands:
        retention_rate = 1.0 - band.reduction_rate
        if band.upper is None:
            width = np.inf
        else:
            width = float(band.upper - band.lower)

        if retention_rate <= 0:
            # A 100% reduction band cannot increase the post-degression payment.
            if target <= paid_before_band:
                return float(band.lower)
            continue

        paid_capacity = width * retention_rate

        if target <= paid_before_band + paid_capacity:
            return float(band.lower + (target - paid_before_band) / retention_rate)

        paid_before_band += paid_capacity

    return None


def _safe_label(label: str) -> str:
    return (
        label.lower()
        .replace("–", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("+", "plus")
    )


def apply_degression(
    gross: pd.Series | np.ndarray | Sequence[float],
    policy: DabisPolicy = DEFAULT_DABIS_POLICY,
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Apply marginal degression and then an optional cap to gross DABIS.

    Returns one row per farm and an explicit breakdown of gross amounts and
    reductions by band.  The cap is applied to the payment *after* degression.
    """
    if isinstance(gross, pd.Series):
        idx = gross.index
        x = gross.fillna(0).to_numpy(dtype=float)
    else:
        x = np.asarray(gross, dtype=float)
        idx = index if index is not None else pd.RangeIndex(len(x))

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(x < 0):
        raise ValueError("gross DABIS cannot be negative")

    result = pd.DataFrame(index=idx)
    result["dabis_gross"] = x

    reduction_total = np.zeros_like(x, dtype=float)

    for band in policy.bands:
        upper = np.inf if band.upper is None else float(band.upper)
        amount_in_band = np.clip(np.minimum(x, upper) - band.lower, 0.0, None)
        reduction = amount_in_band * band.reduction_rate

        suffix = _safe_label(band.label)
        result[f"band_{suffix}"] = amount_in_band
        result[f"reduction_{suffix}"] = reduction
        reduction_total += reduction

    result["reduction_degression"] = reduction_total
    result["dabis_after_degression"] = x - reduction_total

    if policy.payment_cap is None:
        cap_reduction = np.zeros_like(x, dtype=float)
        paid = result["dabis_after_degression"].to_numpy(dtype=float)
    else:
        after = result["dabis_after_degression"].to_numpy(dtype=float)
        cap_reduction = np.clip(after - policy.payment_cap, 0.0, None)
        paid = np.minimum(after, policy.payment_cap)

    result["reduction_capping"] = cap_reduction
    result["reduction_total"] = reduction_total + cap_reduction
    result["dabis_paid"] = paid

    return result


def per_ha_amount(
    data: pd.DataFrame,
    rate: float,
    *,
    area_col: str = "area_biss_criss",
    eligible: pd.Series | np.ndarray | None = None,
    max_area: float | None = None,
) -> pd.Series:
    """Create one configurable per-hectare DABIS component."""
    area = data[area_col].fillna(0).clip(lower=0).astype(float)
    if max_area is not None:
        area = area.clip(upper=max_area)
    if eligible is not None:
        area = area.where(np.asarray(eligible, dtype=bool), 0.0)
    return area * float(rate)


def lump_sum_amount(
    data: pd.DataFrame,
    amount: float,
    *,
    eligible: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    """Create one configurable annual lump-sum DABIS component."""
    if eligible is None:
        mask = np.ones(len(data), dtype=bool)
    else:
        mask = np.asarray(eligible, dtype=bool)
    return pd.Series(np.where(mask, float(amount), 0.0), index=data.index)


def build_gross_dabis(
    data: pd.DataFrame,
    components: Mapping[str, pd.Series | np.ndarray | Sequence[float] | float],
) -> pd.DataFrame:
    """
    Combine arbitrary DABIS components into farm-level gross DABIS.

    Each component must already be expressed in EUR per farm.  This keeps the
    engine independent from the eligibility logic used by apps/notebooks.
    """
    result = pd.DataFrame(index=data.index)

    for name, values in components.items():
        if np.isscalar(values):
            component = np.full(len(data), float(values))
        elif isinstance(values, pd.Series):
            component = values.reindex(data.index).fillna(0).to_numpy(dtype=float)
        else:
            component = np.asarray(values, dtype=float)

        if len(component) != len(data):
            raise ValueError(f"Component {name!r} has incompatible length")
        if np.any(component < 0):
            raise ValueError(f"Component {name!r} contains negative values")

        result[f"dabis_component_{name}"] = component

    result["dabis_gross"] = result.sum(axis=1) if len(result.columns) else 0.0
    return result


def calculate_dabis(
    data: pd.DataFrame,
    components: Mapping[str, pd.Series | np.ndarray | Sequence[float] | float],
    policy: DabisPolicy = DEFAULT_DABIS_POLICY,
) -> pd.DataFrame:
    """Build gross DABIS, apply degression/capping, and return one joined table."""
    gross = build_gross_dabis(data, components)
    degression = apply_degression(gross["dabis_gross"], policy)
    return gross.join(degression.drop(columns="dabis_gross"))


def reduction_summary(result: pd.DataFrame) -> pd.Series:
    """Aggregate every reduction component dynamically, without hard-coded bands."""
    rows: dict[str, float] = {}
    reduction_cols = [
        c
        for c in result.columns
        if c.startswith("reduction_")
        and c not in {"reduction_degression", "reduction_capping", "reduction_total"}
    ]

    for col in reduction_cols:
        rows[col] = float(result[col].sum())

    rows["reduction_degression"] = float(result["reduction_degression"].sum())
    rows["reduction_capping"] = float(result["reduction_capping"].sum())
    rows["reduction_total"] = float(result["reduction_total"].sum())
    rows["dabis_gross"] = float(result["dabis_gross"].sum())
    rows["dabis_paid"] = float(result["dabis_paid"].sum())
    return pd.Series(rows)


def planned_average_per_ha(
    result: pd.DataFrame,
    data: pd.DataFrame,
    *,
    area_col: str = "area_biss_criss",
) -> float:
    """Gross/programming average per eligible hectare, before degression/capping."""
    area = float(data[area_col].fillna(0).clip(lower=0).sum())
    if area == 0:
        return float("nan")
    return float(result["dabis_gross"].sum() / area)


def paid_average_per_ha(
    result: pd.DataFrame,
    data: pd.DataFrame,
    *,
    area_col: str = "area_biss_criss",
) -> float:
    """Observed/effective paid DABIS per hectare; diagnostic, not the planning test."""
    area = float(data[area_col].fillna(0).clip(lower=0).sum())
    if area == 0:
        return float("nan")
    return float(result["dabis_paid"].sum() / area)


def legacy_components(
    data: pd.DataFrame,
    flat_rate: float,
    yfs_per_ha: float = 0.0,
    redist_per_ha: tuple[float, float] = (0.0, 0.0),
    *,
    area_col: str = "area_biss_criss",
    yfs_area_col: str = "area_yfs_cur_eligible",
) -> dict[str, pd.Series]:
    """Reproduce the old flat-rate + YFS + two-step redistributive gross formula."""
    area = data[area_col].fillna(0).clip(lower=0).astype(float)
    yfs_area = data[yfs_area_col].fillna(0).clip(lower=0).astype(float)

    redist1, redist2 = redist_per_ha
    redist = np.where(
        (area <= 1200) & (area != 0),
        redist1 * np.minimum(10, area)
        + redist2 * np.maximum(0, np.minimum(150, area) - 10),
        0.0,
    )

    return {
        "base": area * float(flat_rate),
        "young_farmer": yfs_area * float(yfs_per_ha),
        "redistributive": pd.Series(redist, index=data.index),
    }


def calculate_legacy_dabis(
    data: pd.DataFrame,
    flat_rate: float,
    yfs_per_ha: float = 0.0,
    redist_per_ha: tuple[float, float] = (0.0, 0.0),
    policy: DabisPolicy = DEFAULT_DABIS_POLICY,
) -> pd.DataFrame:
    """Drop-in semantic equivalent of the current legacy gross-support model."""
    return calculate_dabis(
        data,
        legacy_components(data, flat_rate, yfs_per_ha, redist_per_ha),
        policy,
    )


def find_base_rate_for_paid_budget(
    data: pd.DataFrame,
    budget_eur: float,
    *,
    component_builder,
    policy: DabisPolicy = DEFAULT_DABIS_POLICY,
    lower_rate: float = 0.0,
    upper_rate: float = 500.0,
    tolerance: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """
    Generic bisection solver for the base rate when the target is a paid budget.

    component_builder(rate) must return the EUR-per-farm component mapping used
    by calculate_dabis().  This makes the solver work with legacy and new designs.
    """
    if budget_eur < 0:
        raise ValueError("budget_eur must be >= 0")

    def paid(rate: float) -> float:
        result = calculate_dabis(data, component_builder(rate), policy)
        return float(result["dabis_paid"].sum())

    lo, hi = float(lower_rate), float(upper_rate)
    p_lo, p_hi = paid(lo), paid(hi)
    if budget_eur < p_lo - tolerance or budget_eur > p_hi + tolerance:
        raise ValueError(
            f"Target budget {budget_eur:,.2f} is outside the solvable interval "
            f"[{p_lo:,.2f}, {p_hi:,.2f}] for rates {lo:g}-{hi:g}."
        )

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p_mid = paid(mid)
        if abs(p_mid - budget_eur) <= tolerance:
            return mid
        if p_mid < budget_eur:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
