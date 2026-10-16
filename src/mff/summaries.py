from dataclasses import dataclass

import numpy as np
import pandas as pd

from mff.dabis_engine import gross_at_payment_cap
from mff.new_cap import (
    analyze_by_area_categories,
    compute_current_support,
    summarize_reduction_breakdown,
)
from mff.scenario import ScenarioConfig, ScenarioResults

YOUNG_FARMER_EXAMPLE_MAX_HA = 300
AREA_BINS = (0, 10, 150, 300, 1200, float("inf"))
EXAMPLE_AREAS = (5, 10, 50, 100, 150, 300, 500, 1200, 1500)


@dataclass(frozen=True, slots=True)
class YoungFarmerSummary:
    farms: int
    current: float
    new: float
    current_yfs: float
    new_yfs_gross: float

    @property
    def change(self) -> float:
        return self.new - self.current

    @property
    def change_pct(self) -> float:
        return 100 * self.change / self.current if self.current > 0 else float("nan")


def summarize_young_farmers(
    results: ScenarioResults, config: ScenarioConfig
) -> YoungFarmerSummary:
    current = results.current
    mask = current["area_yfs_cur_eligible"].fillna(0).gt(0)
    return YoungFarmerSummary(
        farms=int(mask.sum()),
        current=float(
            current.loc[mask, ["subs_biss", "subs_redist", "subs_yfs"]].sum().sum()
        ),
        new=float(results.new.loc[mask, "subs_capped"].sum()),
        current_yfs=float(current.loc[mask, "subs_yfs"].sum()),
        new_yfs_gross=float(
            current.loc[mask, "area_yfs_cur_eligible"].sum() * config.yfs_per_ha
        ),
    )


def summarize_example_farms(
    config: ScenarioConfig,
    is_young: bool,
    areas: tuple[int, ...] = EXAMPLE_AREAS,
) -> pd.DataFrame:
    """Illustrative farm support amounts; eligibility is assumed, not observed."""
    from mff.new_cap import compute_capped_subsidies

    sizes = np.asarray(areas)
    examples = pd.DataFrame(
        {
            "area_biss_criss": sizes,
            "area_yfs_cur_eligible": (
                np.minimum(sizes, YOUNG_FARMER_EXAMPLE_MAX_HA)
                if is_young
                else np.zeros(len(sizes), dtype=int)
            ),
        }
    )
    new = compute_capped_subsidies(
        examples,
        config.dabis_per_ha,
        config.yfs_per_ha,
        config.redistributive_per_ha,
        config.policy,
    )
    output = pd.DataFrame(
        {
            "Üzemméret (ha)": sizes,
            "Jelenlegi (€)": [
                compute_current_support(float(a), is_young) for a in sizes
            ],
            "Új (€)": new["subs_capped"].to_numpy(),
        }
    )
    output["Változás (€)"] = output["Új (€)"] - output["Jelenlegi (€)"]
    output["Változás (%)"] = 100 * output["Változás (€)"].div(
        output["Jelenlegi (€)"].replace(0, np.nan)
    )
    return output


def _hectares_at_gross(
    gross_limit: float, config: ScenarioConfig, young: bool
) -> float:
    """Illustrative inverse using the same first-300-ha assumption as the UI."""
    dabis = config.dabis_per_ha
    young_rate = config.yfs_per_ha if young and not config.exclude_young_farmer else 0.0
    first_rate = dabis + young_rate
    if first_rate <= 0:
        return float("inf")
    if gross_limit <= YOUNG_FARMER_EXAMPLE_MAX_HA * first_rate:
        return gross_limit / first_rate
    if dabis <= 0:
        return float("inf")
    return (
        YOUNG_FARMER_EXAMPLE_MAX_HA
        + (gross_limit - YOUNG_FARMER_EXAMPLE_MAX_HA * first_rate) / dabis
    )


def _ha_interval(
    low: float, high: float | None, config: ScenarioConfig, young: bool
) -> str:
    def fmt(value: float) -> str:
        if not np.isfinite(value):
            return "∞"
        return f"{value:,.1f}".replace(",", " ").replace(".", ",")

    return f"{fmt(_hectares_at_gross(low, config, young))}–{fmt(_hectares_at_gross(high, config, young)) if high is not None else '∞'}"


def calculate_band_statistics(
    data: pd.DataFrame, config: ScenarioConfig
) -> pd.DataFrame:
    """Gross degression-base volumes by *marginal* band.

    Bands are ordered as in ScenarioConfig.thresholds; the caller must ensure
    the existing breakdown uses that same order.
    """
    gross = config.degression_base(data)
    total = float(gross.sum())
    a, b, c = config.thresholds
    records = []
    for low, high in ((0, a), (a, b), (b, c), (c, None)):
        amount = (gross - low).clip(lower=0)
        if high is not None:
            amount = amount.clip(upper=high - low)
        volume = float(amount.sum())
        records.append(
            {
                "Sáv bruttó DABIS (M€)": volume / 1e6,
                "Bruttó DABIS aránya (%)": 100 * volume / total if total else 0.0,
                "Üzemszám": int((gross > low).sum()),
                "Nem fiatal (ha)": _ha_interval(low, high, config, False),
                "Fiatal (ha)": _ha_interval(low, high, config, True),
            }
        )
    return pd.DataFrame.from_records(records)


def build_reduction_table(
    data: pd.DataFrame,
    config: ScenarioConfig,
    breakdown: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Enrich the engine's band-by-band reduction table for presentation.

    The legacy summarizer is expected to return the four degression bands in
    threshold order, optionally followed by a capping row. Validate that
    structural assumption before assigning positional statistics.
    """
    if breakdown is None:
        breakdown = summarize_reduction_breakdown(
            data,
            config.dabis_per_ha,
            config.yfs_per_ha,
            config.redistributive_per_ha,
            config.policy,
        )
    result = breakdown.copy().reset_index(drop=True)
    bands = calculate_band_statistics(data, config)
    if len(result) not in (len(bands), len(bands) + 1):
        raise ValueError(
            "Unexpected breakdown structure: expected four bands and optional capping row"
        )

    for name in bands.columns:
        result[name] = np.nan if name not in ("Nem fiatal (ha)", "Fiatal (ha)") else "—"
        result.loc[: len(bands) - 1, name] = bands[name].to_numpy()

    cap_trigger = gross_at_payment_cap(config.policy)
    if len(result) > len(bands) and cap_trigger is not None:
        result.at[len(bands), "Nem fiatal (ha)"] = _ha_interval(
            cap_trigger, None, config, False
        )
        result.at[len(bands), "Fiatal (ha)"] = _ha_interval(
            cap_trigger, None, config, True
        )

    columns = list(result.columns)
    columns.remove("Nem fiatal (ha)")
    columns.remove("Fiatal (ha)")
    columns[1:1] = ["Nem fiatal (ha)", "Fiatal (ha)"]
    return result[columns]


def summarize_by_area_class(
    results: ScenarioResults,
    category: str,
    bins: tuple[float, ...] = AREA_BINS,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize already-calculated scenario results by farm size."""

    if labels is None:
        from mff.new_cap import generate_labels_from_bins

        labels = generate_labels_from_bins(list(bins))

    current = results.current
    data = results.new.copy()

    if not data.index.equals(current.index):
        raise ValueError("Current and scenario results must have matching indexes.")

    data["subs_cur"] = current[["subs_biss", "subs_redist", "subs_yfs"]].sum(axis=1)

    if category == "csak fiatal gazda":
        data = data.loc[current["subs_yfs"] > 0]

    elif category == "nem fiatal gazda":
        data = data.loc[current["subs_yfs"] == 0]

    elif category != "összes gazda":
        raise ValueError(f"Unknown farmer category: {category}")

    return analyze_by_area_categories(
        data,
        bins=list(bins),
        labels=labels,
    )
