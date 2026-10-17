from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from mff.dabis_engine import apply_degression, gross_at_payment_cap
from mff.new_cap import (
    analyze_by_area_categories,
    compute_current_support,
    summarize_reduction_breakdown,
)
from mff.scenario import ScenarioConfig, ScenarioResults

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
            config.young_area(current).loc[mask].sum() * config.yfs_per_ha
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
                np.minimum(sizes, config.yfs_max_ha)
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


def _gross_for_area(
    area: float, config: ScenarioConfig, young: bool, *, with_redist: bool
) -> float:
    """Gross reduction base for an illustrative fully eligible farm."""
    yfs_rate = config.yfs_per_ha if young and not config.exclude_young_farmer else 0.0
    gross = area * config.dabis_per_ha + min(area, config.yfs_max_ha) * yfs_rate
    if with_redist and 0 < area <= 1200:
        r1, r2 = config.redistributive_per_ha
        gross += r1 * min(area, 10) + r2 * max(0, min(area, 150) - 10)
    return gross


def _ha_interval(
    low: float, high: float | None, config: ScenarioConfig, young: bool
) -> str:
    """All area intervals whose reduction base lies in the requested band.

    Redistributive support stops above 1,200 ha; as a result the mapping
    need not be monotonic. Return separate intervals rather than a false
    single continuous range. Assumes all hectares are eligible for DABIS.
    """
    points = sorted({0.0, 10.0, 150.0, min(float(config.yfs_max_ha), 1200.0), 1200.0})
    sections = [(a, b, True) for a, b in pairwise(points)]
    after = [1200.0]
    if config.yfs_max_ha > 1200:
        after.append(float(config.yfs_max_ha))
    after.append(float("inf"))
    sections.extend((a, b, False) for a, b in pairwise(after))
    intervals: list[tuple[float, float]] = []

    for left, right, with_redist in sections:
        if with_redist:
            gl = _gross_for_area(left, config, young, with_redist=True)
            gr = _gross_for_area(right, config, young, with_redist=True)
            slope = (gr - gl) / (right - left)
        else:
            gl = _gross_for_area(left, config, young, with_redist=False)
            slope = config.dabis_per_ha
            if young and not config.exclude_young_farmer and left < config.yfs_max_ha:
                # Already split at the YFS boundary in sections above.
                slope += config.yfs_per_ha

        if slope < 0:
            raise ValueError("Negative marginal gross support rate is unsupported.")
        if slope == 0:
            if gl >= low and (high is None or gl <= high):
                intervals.append((left, right))
            continue

        begin = max(left, left + (low - gl) / slope)
        end = min(right, float("inf") if high is None else left + (high - gl) / slope)
        if begin <= end and end >= left and begin <= right:
            intervals.append((begin, end))

    if not intervals:
        return "—"

    # Merge touching intervals; retain gaps created by the 1,200 ha discontinuity.
    merged: list[tuple[float, float]] = []
    for begin, end in intervals:
        if merged and abs(merged[-1][1] - begin) < 1e-8:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))

    def fmt(value: float) -> str:
        return f"{value:,.1f}".replace(",", " ").replace(".", ",")

    return "; ".join(
        f"{fmt(a)} felett" if not np.isfinite(b) else f"{fmt(a)}–{fmt(b)}"
        for a, b in merged
    )


def _cap_reductions(
    scenario: pd.DataFrame | None,
    gross: pd.Series,
    config: ScenarioConfig,
) -> pd.Series:
    if scenario is None:
        return apply_degression(gross, config.policy)["reduction_capping"]
    if not scenario.index.equals(gross.index):
        raise ValueError("The reduction table and scenario have different indexes.")
    return scenario["subs_capping_reduction"].astype(float)


def calculate_band_statistics(
    data: pd.DataFrame,
    config: ScenarioConfig,
    scenario: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Marginal gross and finally paid amounts for each degression band.

    Post-degression capping is allocated from the highest marginal bands down.
    Net band totals include support exempted from reductions. The full
    gross DABIS (general + young farmer + redistributive support) is unchanged
    when the young-farmer exemption is toggled.
    """
    gross = config.degression_base(data)
    if scenario is not None and not data.index.equals(scenario.index):
        raise ValueError("The reduction table and scenario have different indexes.")

    area = data["area_biss_criss"].fillna(0).clip(lower=0).astype(float)
    # The exclusive classification is based on total gross DABIS, including
    # YFS even when young-farmer payments are exempt from reductions.
    # The existing "Üzemszám" continues to use the degression base.
    gross_dabis = (
        scenario["subs_eur"].astype(float)
        if scenario is not None
        else gross
        + (
            config.young_area(data) * config.yfs_per_ha
            if config.exclude_young_farmer
            else 0.0
        )
    )
    # Use the actual farm-level total loss, not a hypothetical band maximum.
    total_reduction = (
        scenario["subs_total_reduction"].astype(float)
        if scenario is not None
        else apply_degression(gross, config.policy)["reduction_total"]
    )
    # The full gross support is the reference for gross/net reporting.
    # The narrower degression base is used only to calculate reductions.
    total = float(gross_dabis.sum())
    cap_remaining = _cap_reductions(scenario, gross, config).copy()

    records = []
    retained_slices = []
    exempt_slices = []
    for band in config.policy.bands:
        volume_per_farm = (gross - band.lower).clip(lower=0)

        if band.upper is not None:
            volume_per_farm = volume_per_farm.clip(upper=band.upper - band.lower)

        retained_slices.append(volume_per_farm * (1 - band.reduction_rate))

        # Teljes bruttó DABIS, beleértve a YFS-t és CRISS-t
        gross_volume = (gross_dabis - band.lower).clip(lower=0)

        if band.upper is not None:
            gross_volume = gross_volume.clip(upper=band.upper - band.lower)

        # Gross minus reduction-base marginal volume is exempt support.
        # Its full amount is paid and must be included in net support.
        exempt_slices.append(gross_volume - volume_per_farm)
        volume = float(gross_volume.sum())
        # Mutually exclusive groups; exclude zero-gross-support records.
        in_gross_band = gross_dabis.gt(band.lower)
        if band.upper is not None:
            in_gross_band &= gross_dabis.le(band.upper)
        gross_in_group = float(gross_dabis.loc[in_gross_band].sum())
        loss_in_group = float(total_reduction.loc[in_gross_band].sum())
        farms_in_group = int(in_gross_band.sum())
        records.append(
            {
                "Bruttó DABIS (M€)": volume / 1e6,
                "Bruttó DABIS aránya (%)": 100 * volume / total if total else 0.0,
                "Sávval érintett gazdaságok (db)": int((gross > band.lower).sum()),
                "Sávba tartozó gazdaságok (db)": farms_in_group,
                "Sávba tartozó gazdaságok területe (ha)": float(
                    area.loc[in_gross_band].sum()
                ),
                "Sávba tartozók bruttó támogatása (M€)": gross_in_group / 1e6,
                "Sávba tartozók teljes elvonása (M€)": loss_in_group / 1e6,
                "Sávba tartozók nettó támogatása (M€)": (gross_in_group - loss_in_group)
                / 1e6,
                "Elvonással érintett gazdaságok (db)": int(
                    total_reduction.loc[in_gross_band].gt(0).sum()
                ),
                "Átlagos teljes elvonás (€/gazdaság)": (
                    loss_in_group / farms_in_group if farms_in_group else np.nan
                ),
                # Group-level (gross-support-weighted) effective reduction rate.
                "Teljes elvonás aránya (%)": (
                    100 * loss_in_group / gross_in_group if gross_in_group else np.nan
                ),
                "Nem fiatal (ha)": _ha_interval(band.lower, band.upper, config, False),
                "Fiatal (ha)": _ha_interval(band.lower, band.upper, config, True),
            }
        )

    for i in range(len(retained_slices) - 1, -1, -1):
        retained = retained_slices[i]
        cap_from_band = np.maximum(
            np.minimum(retained, cap_remaining),
            0.0,
        )
        records[i]["Nettó DABIS (M€)"] = (
            float((retained - cap_from_band + exempt_slices[i]).sum()) / 1e6
        )
        cap_remaining -= cap_from_band

    if not np.allclose(cap_remaining.to_numpy(), 0.0, atol=0.01):
        raise ValueError("Cannot allocate capping reductions to degression bands.")

    band_stats = pd.DataFrame.from_records(records)
    # The ordinary bands must partition FULL gross and FULL net support.
    if not np.isclose(band_stats["Bruttó DABIS (M€)"].sum() * 1e6, total, atol=0.05):
        raise ValueError(
            "Marginal gross DABIS does not reconcile with full gross support."
        )

    full_net = total - float(total_reduction.sum())
    if not np.isclose(band_stats["Nettó DABIS (M€)"].sum() * 1e6, full_net, atol=0.05):
        raise ValueError("Marginal net DABIS does not reconcile with full net support.")
    return band_stats


def build_reduction_table(
    data: pd.DataFrame,
    config: ScenarioConfig,
    breakdown: pd.DataFrame | None = None,
    scenario: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Enrich the engine's marginal reduction table for presentation.

    The capping row is an OVERLAPPING diagnostic, not an additional band.
    Its affected farms are already represented in the degression bands.
    The only additive cap-row financial measure is its 'Elvonás (M€)'.
    """
    if breakdown is None:
        scenario_data = data.assign(area_yfs_cur_eligible=config.young_area(data))
        breakdown = summarize_reduction_breakdown(
            scenario_data,
            config.dabis_per_ha,
            config.yfs_per_ha,
            config.redistributive_per_ha,
            config.policy,
        )
    result = breakdown.copy().reset_index(drop=True)
    bands = calculate_band_statistics(data, config, scenario=scenario)
    if len(result) not in (len(bands), len(bands) + 1):
        raise ValueError(
            "Unexpected breakdown structure: expected four bands and optional capping row"
        )

    for name in bands.columns:
        result[name] = np.nan if name not in ("Nem fiatal (ha)", "Fiatal (ha)") else "—"
        result.loc[: len(bands) - 1, name] = bands[name].to_numpy()

    if len(result) > len(bands):
        cap_idx = len(bands)
        trigger = gross_at_payment_cap(config.policy)
        gross = config.degression_base(data)
        cap_reductions = _cap_reductions(scenario, gross, config)
        capped = cap_reductions > 0
        if scenario is not None:
            gross_total = scenario["subs_eur"].astype(float)
            total_reduction = scenario["subs_total_reduction"].astype(float)
        else:
            gross_total = gross + (
                config.young_area(data) * config.yfs_per_ha
                if config.exclude_young_farmer
                else 0.0
            )
            total_reduction = apply_degression(gross, config.policy)["reduction_total"]
        capped_farms = int(capped.sum())
        capped_loss = float(total_reduction.loc[capped].sum())
        capped_gross = float(gross_total.loc[capped].sum())
        gross_above_cap = (
            float((gross.loc[capped] - trigger).clip(lower=0).sum())
            if trigger is not None
            else 0.0
        )
        result.at[cap_idx, "Sávval érintett gazdaságok (db)"] = capped_farms
        # There is no exclusive gross-support band for capping.
        result.at[cap_idx, "Sávba tartozó gazdaságok területe (ha)"] = float(
            data.loc[capped, "area_biss_criss"].fillna(0).clip(lower=0).sum()
        )
        # For capping, these group-level fields describe *affected* farms,
        # not an exclusive band; they overlap the other group statistics.
        result.at[cap_idx, "Sávba tartozók bruttó támogatása (M€)"] = capped_gross / 1e6
        result.at[cap_idx, "Sávba tartozók teljes elvonása (M€)"] = capped_loss / 1e6
        result.at[cap_idx, "Sávba tartozók nettó támogatása (M€)"] = (
            capped_gross - capped_loss
        ) / 1e6
        result.at[cap_idx, "Elvonással érintett gazdaságok (db)"] = capped_farms
        result.at[cap_idx, "Átlagos teljes elvonás (€/gazdaság)"] = (
            capped_loss / capped_farms if capped_farms else np.nan
        )
        result.at[cap_idx, "Teljes elvonás aránya (%)"] = (
            100 * capped_loss / capped_gross if capped_gross else np.nan
        )
        # These are not an independent marginal band: do not show zero net
        # or gross figures that readers might add to band subtotals.
        for name in (
            "Bruttó DABIS (M€)",
            "Nettó DABIS (M€)",
            "Bruttó DABIS aránya (%)",
        ):
            result.at[cap_idx, name] = np.nan
        result.at[cap_idx, "Plafon feletti bruttó alap (M€)"] = gross_above_cap / 1e6
        result.at[cap_idx, "Plafon miatti elvonás (M€)"] = (
            float(cap_reductions.loc[capped].sum()) / 1e6
        )
        if trigger is not None:
            result.at[cap_idx, "Nem fiatal (ha)"] = _ha_interval(
                trigger, None, config, False
            )
            result.at[cap_idx, "Fiatal (ha)"] = _ha_interval(
                trigger, None, config, True
            )

    for column in (
        "Sávval érintett gazdaságok (db)",
        "Sávba tartozó gazdaságok (db)",
        "Elvonással érintett gazdaságok (db)",
    ):
        if column in result.columns:
            # The capping row is an overlapping diagnostic, not a gross band.
            result[column] = pd.to_numeric(result[column]).astype("Int64")

    preferred = [
        "Tétel",
        "Nem fiatal (ha)",
        "Fiatal (ha)",
        "Elvonás (€)",
        "Elvonás (M€)",
        "Elvonási ráta (%)",
        "Bruttó DABIS aránya (%)",
        "Bruttó DABIS (M€)",
        "Nettó DABIS (M€)",
        "Sávba tartozó gazdaságok (db)",
        "Sávval érintett gazdaságok (db)",
        "Sávba tartozó gazdaságok területe (ha)",
        "Sávba tartozók bruttó támogatása (M€)",
        "Sávba tartozók teljes elvonása (M€)",
        "Sávba tartozók nettó támogatása (M€)",
        "Elvonással érintett gazdaságok (db)",
        "Átlagos teljes elvonás (€/gazdaság)",
        "Teljes elvonás aránya (%)",
        "Plafon feletti bruttó alap (M€)",
        "Plafon miatti elvonás (M€)",
    ]
    columns = [name for name in preferred if name in result]
    columns.extend(str(name) for name in result if name not in columns)
    output = result[columns]
    output.attrs["note"] = (
        "A degressziós sávok pénzügyi oszlopai marginális összegek; "
        "a sávba tartozók mutatói az adott csoport teljes támogatására és elvonására vonatkoznak. "
        "A plafon sora átfedő diagnosztikai sor: gazdaságszáma, területe és csoportmutatói "
        "nem adhatók hozzá a degressziós sávokhoz. "
        "A hektárküszöbök az elvonási alapból számított tájékoztató értékek; "
        "YFS-mentesség esetén eltérhetnek a teljes bruttó támogatás szerint "
        "képzett gazdaságcsoportok határaitól."
    )
    return output


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

    young_mask = current["area_yfs_cur_eligible"].fillna(0).gt(0)

    if category == "csak fiatal gazda":
        data = data.loc[young_mask]

    elif category == "nem fiatal gazda":
        data = data.loc[~young_mask]

    elif category != "összes gazda":
        raise ValueError(f"Unknown farmer category: {category}")

    return analyze_by_area_categories(
        data,
        bins=list(bins),
        labels=labels,
    )


def summarize_reductions(
    results: ScenarioResults,
    config: ScenarioConfig,
) -> dict[str, pd.DataFrame]:
    """Summarize reductions for all, young and non-young farmers."""

    current = results.current
    scenario = results.new

    if not current.index.equals(scenario.index):
        raise ValueError("Current and scenario indexes must match.")

    # Reuse the engine's exact per-band and capping reductions; recomputing
    # the gross base separately could omit redistributive support.
    cap_reduction = scenario["subs_capping_reduction"]

    # Consistent definition with summarize_young_farmers().
    young = current["area_yfs_cur_eligible"].fillna(0).gt(0)

    masks = {
        "all": pd.Series(True, index=current.index),
        "young": young,
        "non_young": ~young,
    }

    summaries = {}

    for group, mask in masks.items():
        records = []

        for band in config.policy.bands:
            # Matches dabis_engine.apply_degression() naming.
            suffix = (
                band.label.lower()
                .replace("–", "_")
                .replace("-", "_")
                .replace(" ", "_")
                .replace("+", "plus")
            )

            column = f"reduction_{suffix}"

            reduction = float(scenario.loc[mask, column].sum())

            lower = f"{band.lower:,.0f}".replace(",", " ")
            if band.upper is None:
                label = f"{lower} € felett"
            else:
                upper = f"{band.upper:,.0f}".replace(",", " ")
                label = f"{lower}–{upper} €"

            records.append(
                {
                    "Tétel": label,
                    "Elvonás (€)": reduction,
                    "Elvonás (M€)": reduction / 1e6,
                    "Elvonási ráta (%)": band.reduction_rate * 100,
                }
            )

        cap_amount = float(cap_reduction.loc[mask].sum())

        records.append(
            {
                "Tétel": (
                    f"Kifizetési plafon ({config.policy.payment_cap:,.0f} €)".replace(
                        ",", " "
                    )
                    if config.policy.payment_cap is not None
                    else "Capping (kikapcsolva)"
                ),
                "Elvonás (€)": cap_amount,
                "Elvonás (M€)": cap_amount / 1e6,
                "Elvonási ráta (%)": 100.0
                if config.policy.payment_cap is not None
                else np.nan,
            }
        )

        summaries[group] = pd.DataFrame(records)

    # Validate the reconciliation with the calculated scenario.
    total = summaries["all"]

    if not np.isclose(
        total["Elvonás (€)"].sum(),
        results.total_reduction,
        rtol=1e-9,
        atol=0.01,
    ):
        raise ValueError("Reduction breakdown does not match scenario totals.")

    return summaries
