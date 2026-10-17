from dataclasses import dataclass

import pandas as pd

from mff.dabis_engine import DabisPolicy, DegressionBand, legacy_components
from mff.new_cap import compute_capped_subsidies


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Parameters defining a DABIS policy scenario."""

    year: int = 2024

    dabis_per_ha: float = 90.0
    yfs_per_ha: float = 50.0
    yfs_max_ha: float = 300.0
    redistributive_per_ha: tuple[float, float] = (0.0, 0.0)

    thresholds: tuple[float, float, float] = (
        20_000.0,
        50_000.0,
        75_000.0,
    )
    rates: tuple[float, float, float] = (
        0.25,
        0.50,
        0.75,
    )

    payment_cap: float | None = 100_000.0
    exclude_young_farmer: bool = False

    def __post_init__(self) -> None:
        a, b, c = self.thresholds

        if not 0 < a < b < c:
            raise ValueError("Degression thresholds must be strictly increasing.")

        if any(not 0 <= rate <= 1 for rate in self.rates):
            raise ValueError("Degression rates must be between 0 and 1.")

        if self.payment_cap is not None and self.payment_cap <= 0:
            raise ValueError("Payment cap must be positive.")

        if self.dabis_per_ha < 0 or self.yfs_per_ha < 0:
            raise ValueError("Support rates cannot be negative.")

        if self.yfs_max_ha <= 0:
            raise ValueError("YFS maximum area must be positive.")

    def young_area(self, data: pd.DataFrame) -> pd.Series:
        return (
            data["area_biss_criss"]
            .where(data["area_yfs_cur_eligible"].gt(0), 0.0)
            .fillna(0.0)
            .clip(lower=0.0, upper=self.yfs_max_ha)
        )

    @property
    def policy(self) -> DabisPolicy:
        """Construct the degression and capping policy."""
        a, b, c = self.thresholds
        r1, r2, r3 = self.rates

        return DabisPolicy(
            bands=(
                DegressionBand(0, a, 0.0, "band_1"),
                DegressionBand(a, b, r1, "band_2"),
                DegressionBand(b, c, r2, "band_3"),
                DegressionBand(c, None, r3, "band_4"),
            ),
            payment_cap=self.payment_cap,
            exclude_young_farmer=self.exclude_young_farmer,
        )

    def degression_base(self, data: pd.DataFrame) -> pd.Series:
        """Gross reduction base, consistent with the scenario calculation engine."""
        eligible = data.assign(area_yfs_cur_eligible=self.young_area(data))
        components = legacy_components(
            eligible,
            flat_rate=self.dabis_per_ha,
            yfs_per_ha=self.yfs_per_ha,
            redist_per_ha=self.redistributive_per_ha,
        )
        base = components["base"] + components["redistributive"]
        if not self.exclude_young_farmer:
            base = base + components["young_farmer"]
        return base


@dataclass(slots=True)
class ScenarioResults:
    """Farm-level results and basic scenario aggregates."""

    current: pd.DataFrame
    new: pd.DataFrame

    @property
    def total_current(self) -> float:
        columns = ("subs_biss", "subs_redist", "subs_yfs")
        return float(self.current[list(columns)].sum().sum())

    @property
    def total_gross(self) -> float:
        return float(self.new["subs_eur"].sum())

    @property
    def total_paid(self) -> float:
        return float(self.new["subs_capped"].sum())

    @property
    def total_reduction(self) -> float:
        return float(self.new["subs_total_reduction"].sum())

    @property
    def degression_reduction(self) -> float:
        return float(self.new["subs_degression_reduction"].sum())

    @property
    def capping_reduction(self) -> float:
        return float(self.new["subs_capping_reduction"].sum())

    @property
    def capped_farms(self) -> int:
        return int((self.new["subs_capping_reduction"] > 0).sum())

    @property
    def eligible_farms(self) -> int:
        return int(self.current["area_biss_criss"].fillna(0).gt(0).sum())

    @property
    def eligible_area(self) -> float:
        return float(self.current["area_biss_criss"].fillna(0).clip(lower=0).sum())


def calculate_scenario(
    data: pd.DataFrame,
    config: ScenarioConfig,
) -> ScenarioResults:
    """Calculate a DABIS scenario for a farm-level dataset."""

    scenario_data = data.copy()

    scenario_data["area_yfs_cur_eligible"] = config.young_area(data)

    new = compute_capped_subsidies(
        scenario_data,
        config.dabis_per_ha,
        config.yfs_per_ha,
        config.redistributive_per_ha,
        config.policy,
    )

    return ScenarioResults(
        current=data,
        new=new,
    )
