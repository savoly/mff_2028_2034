import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal
from collections.abc import Callable, Iterable, Mapping
from sqlalchemy.engine import Engine

from mff.new_cap import apply_reductions, cal_redist


def get_data(engine: Engine, year: int) -> pd.DataFrame:
    with open(r"sql\fadn_dabis.sql", encoding="utf-8") as f:
        sql = f.read()

    data = pd.read_sql(sql, con=engine)
    data = data[data["ev"].eq(year)]
    return data


NOT_DISPLAYED_COLS: dict[str, str] = {
    "n_farms": "Üzemek száma - súlyozatlan (db)",
    "n_farms_weighted": "Üzemek száma - súlyozott (db)",
    "n_farms_current_neg": "Vesztéseges üzemek száma - tényadat - súlyozatlan (db)",
    "n_farms_current_pos": "Nyereséges üzemek száma - tényadat - súlyozatlan (db)",
    "n_farms_scenario_neg": "Vesztéseges üzemek száma - szcenárió - súlyozatlan (db)",
    "n_farms_scenario_pos": "Nyereséges üzemek száma - szcenárió - súlyozatlan (db)",
    "n_farms_current_neg_weighted": "Vesztéseges üzemek száma - tényadat - súlyozott (db)",
    "n_farms_current_pos_weighted": "Nyereséges üzemek száma - tényadat - súlyozott (db)",
    "n_farms_scenario_neg_weighted": "Vesztéseges üzemek száma - szcenárió - súlyozott (db)",
    "n_farms_scenario_pos_weighted": "Nyereséges üzemek száma - szcenárió - súlyozott (db)",
}

DISPLAY_COLS: dict[str, str] = {
    "n_farms_ratio_neg_current_non_weighted": "Vesztéseges üzemek aránya - tényadat - súlyozatlan (%)",
    "n_farms_ratio_pos_current_non_weighted": "Nyereséges üzemek aránya - tényadat - súlyozatlan (%)",
    "n_farms_ratio_neg_scenario_non_weighted": "Vesztéseges üzemek száma - szcenárió - súlyozatlan (%)",
    "n_farms_ratio_pos_scenario_non_weighted": "Nyereséges üzemek aránya - szcenárió - súlyozatlan (%)",
    "n_farms_ratio_neg_current_weighted": "Vesztéseges üzemek aránya - tényadat - súlyozott (%)",
    "n_farms_ratio_pos_current_weighted": "Nyereséges üzemek aránya - tényadat - súlyozott (%)",
    "n_farms_ratio_neg_scenario_weighted": "Vesztéseges üzemek száma - szcenárió - súlyozott (%)",
    "n_farms_ratio_pos_scenario_weighted": "Nyereséges üzemek aránya - szcenárió - súlyozott (%)",
    "n_farms_ratio_is_less_non_weighted": "EBIDTA-csökkenéses üzemek aránya - súlyozatlan (%)",
    "n_farms_ratio_is_not_less_non_weighted": "EBIDTA-nem-csökkenés aránya száma - súlyozatlan (%)",
    "n_farms_ratio_is_less_weighted": "EBIDTA-csökkenés üzemek aránya - súlyozott (%)",
    "n_farms_ratio_is_not_less_weighted": "EBIDTA-nem-csökkenés aránya száma - súlyozott (%)",
    "val_brutto_termelesi_ertek_ratio_weighted": "Bruttó termelési érték arány - súlyozott (%)",
    "val_brutto_termelesi_ertek_ratio_non_weighted": "Bruttó termelési érték arány - súlyozatlan (%)",
}

ROW_ORDER: list[str] = [
    "novterm",
    "zoldsegsz",
    "zoldsegn",
    "gyumolcs",
    "szolo",
    "legel",
    "tej",
    "sertes",
    "baromfi",
    "vegyes",
    "teljes_mgi",
]


@dataclass(frozen=True)
class FADNPolicyParams:
    year: int
    exchange_rate: float
    base_payment_per_ha: float
    yfs_per_ha: float
    redist_params: tuple[float, float]


def create_base_data(engine: Engine, params: FADNPolicyParams) -> pd.DataFrame:
    data = get_data(engine, params.year)

    mask_excl = data["mezogazdasagi_terulet_ha"].eq(0) & data["tip_m10ste"].isin(
        ["zoldsegn", "legel"]
    )
    data = data[~mask_excl].copy()

    dabis_eur = (
        data["biss_ha"] * params.base_payment_per_ha
        + data["yfs_ha"] * params.yfs_per_ha
        + data["biss_ha"].apply(lambda x: cal_redist(x, params.redist_params))
    ).apply(apply_reductions)

    data["dabis_ft"] = params.exchange_rate * dabis_eur

    old_subs_titles = ["biss_ft", "criss_ft", "yfs_ft"]
    data["cur_payments_ft"] = data[old_subs_titles].fillna(0).sum(axis=1)

    data["tamogatasok_osszesen_ft_dabis"] = (
        data["tamogatasok_osszesen_ft"] + data["dabis_ft"] - data["cur_payments_ft"]
    )

    delta_support = (
        data["tamogatasok_osszesen_ft_dabis"] - data["tamogatasok_osszesen_ft"]
    )

    adjust_cols = [
        "adozott_eredmeny_ft",
        "ebitda_ft",
        "netto_hozzadott_ertek_ft",
        "brutto_hozzadott_ertek_ft",
    ]

    for col in adjust_cols:
        data[f"{col}_dabis"] = data[col] + delta_support

    data["young_farmer"] = np.where(data["yfs_ha"] > 0, 1, 0)

    return data


@dataclass(frozen=True)
class Interval:
    end_points: tuple[float, float]
    interval_type: Literal["left_closed", "right_closed", "closed", "open"]


def interval_mask(
    df: pd.DataFrame, interval: Interval | None, col: str = "mezogazdasagi_terulet_ha"
) -> pd.Series:
    if interval is None:
        # keep all rows
        return pd.Series(True, index=df.index)

    start, end = interval.end_points
    if interval.interval_type == "left_closed":
        return (start <= df[col]) & (df[col] < end)
    elif interval.interval_type == "right_closed":
        return (start < df[col]) & (df[col] <= end)
    elif interval.interval_type == "closed":
        return (start <= df[col]) & (df[col] <= end)
    elif interval.interval_type == "open":
        return (start < df[col]) & (df[col] < end)

    # fallback: no filtering
    return pd.Series(True, index=df.index)


def filter_interval(
    df: pd.DataFrame, interval: Interval | None, col: str = "mezogazdasagi_terulet_ha"
) -> pd.DataFrame:
    return df[interval_mask(df, interval, col=col)]


def _prepare_flags_and_weights(tmp: pd.DataFrame, metric: str, md: str) -> pd.DataFrame:
    tmp = tmp.copy()
    tmp["weight"] = tmp["weights_ste"]

    tmp["ebidta_arany"] = 100 * (tmp["ebitda_ft"] / tmp["ebitda_ft_dabis"] - 1)
    tmp["weighted_ebidta_arany"] = 100 * tmp["ebidta_arany"] * tmp["weight"]

    tmp["brutto_termelesi_ertek_ft_wgt"] = (
        tmp["brutto_termelesi_ertek_ft"] * tmp["weight"]
    )
    tmp["brutto_termelesi_ertek_eves_no_wgt"] = tmp.groupby("ev")[
        "brutto_termelesi_ertek_ft"
    ].transform("sum")
    tmp["brutto_termelesi_ertek_eves_wgt"] = tmp.groupby("ev")[
        "brutto_termelesi_ertek_ft_wgt"
    ].transform("sum")

    tmp["is_neg_current"] = (tmp[metric] < 0).astype(int)
    tmp["is_neg_scenario"] = (tmp[md] < 0).astype(int)
    tmp["is_pos_current"] = (tmp[metric] >= 0).astype(int)
    tmp["is_pos_scenario"] = (tmp[md] >= 0).astype(int)
    tmp["is_less"] = ((tmp[md] - tmp[metric]) < 0).astype(int)
    tmp["is_not_less"] = ((tmp[md] - tmp[metric]) >= 0).astype(int)

    for col in [
        "is_less",
        "is_not_less",
        "is_neg_current",
        "is_neg_scenario",
        "is_pos_current",
        "is_pos_scenario",
    ]:
        tmp[f"weighted_{col}"] = tmp["weight"] * tmp[col]

    return tmp


def prepare_base_df(
    data: pd.DataFrame,
    metric: str,
    group_cols: Iterable[str],
    interval: Interval | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    md = f"{metric}_dabis"
    required = set(group_cols) | {
        metric,
        md,
        "cur_payments_ft",
        "dabis_ft",
        "young_farmer",
        "weights_ste",
        "brutto_termelesi_ertek_ft",
        "mezogazdasagi_terulet_ha",
    }
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Hiányzó oszlop(ok): {missing}")

    tmp = data[
        list(group_cols)
        + ["weights_ste"]
        + [
            metric,
            md,
            "cur_payments_ft",
            "dabis_ft",
            "young_farmer",
            "brutto_termelesi_ertek_ft",
            "biss_ha",
            "mezogazdasagi_terulet_ha",
        ]
    ].copy()

    tmp = _prepare_flags_and_weights(tmp, metric, md)
    tmp = filter_interval(tmp, interval)
    if row_filter is not None:
        mask = row_filter(tmp)
        tmp = tmp[mask]

    return tmp


def compute_data(
    data: pd.DataFrame,
    metric: str,
    group_cols: Iterable[str],
    interval: Interval | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    tmp = prepare_base_df(
        data=data,
        metric=metric,
        group_cols=group_cols,
        interval=interval,
        row_filter=row_filter,
    )

    grp = tmp.groupby(list(group_cols), as_index=False).agg(
        n_farms=(metric, "size"),
        n_farms_current_neg=("is_neg_current", "sum"),
        n_farms_scenario_neg=("is_neg_scenario", "sum"),
        n_farms_current_pos=("is_pos_current", "sum"),
        n_farms_scenario_pos=("is_pos_scenario", "sum"),
        n_farms_is_less=("is_less", "sum"),
        n_farms_is_not_less=("is_not_less", "sum"),
        n_farms_weighted=("weight", "sum"),
        n_farms_current_neg_weighted=("weighted_is_neg_current", "sum"),
        n_farms_scenario_neg_weighted=("weighted_is_neg_scenario", "sum"),
        n_farms_current_pos_weighted=("weighted_is_pos_current", "sum"),
        n_farms_scenario_pos_weighted=("weighted_is_pos_scenario", "sum"),
        n_farms_is_less_weighted=("weighted_is_less", "sum"),
        n_farms_is_not_less_weighted=("weighted_is_not_less", "sum"),
        val_brutto_termelesi_ertek_non_weighted=("brutto_termelesi_ertek_ft", "sum"),
        val_brutto_termelesi_ertek_weighted=("brutto_termelesi_ertek_ft_wgt", "sum"),
        brutto_termelesi_ertek_eves_wgt=("brutto_termelesi_ertek_eves_wgt", "max"),
        brutto_termelesi_ertek_eves_no_wgt=(
            "brutto_termelesi_ertek_eves_no_wgt",
            "max",
        ),
    )

    grp["n_farms_ratio_neg_current_weighted"] = 100 * (
        grp["n_farms_current_neg_weighted"] / grp["n_farms_weighted"]
    )
    grp["n_farms_ratio_pos_current_weighted"] = 100 * (
        grp["n_farms_current_pos_weighted"] / grp["n_farms_weighted"]
    )
    grp["n_farms_ratio_neg_scenario_weighted"] = 100 * (
        grp["n_farms_scenario_neg_weighted"] / grp["n_farms_weighted"]
    )
    grp["n_farms_ratio_pos_scenario_weighted"] = 100 * (
        grp["n_farms_scenario_pos_weighted"] / grp["n_farms_weighted"]
    )
    grp["n_farms_ratio_pos_current_non_weighted"] = 100 * (
        grp["n_farms_current_pos"] / grp["n_farms"]
    )
    grp["n_farms_ratio_pos_scenario_non_weighted"] = 100 * (
        grp["n_farms_scenario_pos"] / grp["n_farms"]
    )

    grp["n_farms_ratio_neg_current_non_weighted"] = 100 * (
        grp["n_farms_current_neg"] / grp["n_farms"]
    )
    grp["n_farms_ratio_neg_scenario_non_weighted"] = 100 * (
        grp["n_farms_scenario_neg"] / grp["n_farms"]
    )

    grp["n_farms_ratio_is_less_weighted"] = 100 * (
        grp["n_farms_is_less_weighted"] / grp["n_farms_weighted"]
    )
    grp["n_farms_ratio_is_not_less_weighted"] = 100 * (
        grp["n_farms_is_not_less_weighted"] / grp["n_farms_weighted"]
    )
    grp["n_farms_ratio_is_less_non_weighted"] = 100 * (
        grp["n_farms_is_less"] / grp["n_farms"]
    )
    grp["n_farms_ratio_is_not_less_non_weighted"] = 100 * (
        grp["n_farms_is_not_less"] / grp["n_farms"]
    )

    grp["val_brutto_termelesi_ertek_ratio_weighted"] = 100 * (
        grp["val_brutto_termelesi_ertek_weighted"]
        / grp["brutto_termelesi_ertek_eves_wgt"]
    )
    grp["val_brutto_termelesi_ertek_ratio_non_weighted"] = 100 * (
        grp["val_brutto_termelesi_ertek_non_weighted"]
        / grp["brutto_termelesi_ertek_eves_no_wgt"]
    )

    return grp


def summarize_metric_pivot_with_interval(
    data: pd.DataFrame,
    metric: str,
    interval: Interval,
    *,
    pretty_metric_name: str | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    data = data.copy()

    grp0 = compute_data(data, metric, ("ev", "tip_m10ste"), interval, row_filter)
    grp1 = compute_data(data, metric, ("ev",), interval, row_filter)
    grp1.insert(1, "tip_m10ste", "teljes_mgi")

    grp = pd.concat([grp0, grp1], ignore_index=True)

    pivot = (
        grp.pivot(index="tip_m10ste", columns="ev", values=list(DISPLAY_COLS.keys()))
        .rename(columns=DISPLAY_COLS, level=0)
        .reindex(list(DISPLAY_COLS.values()), axis=1, level=0)
        .reindex(ROW_ORDER)
    )
    return pivot


def summarize_multi_metrics_pivot_with_interval(
    data: pd.DataFrame,
    metrics: Iterable[str],
    interval: Interval,
    *,
    group_cols: Iterable[str] = ("ev", "tip_m10ste"),
    pretty_names: Mapping[str, str] | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    pivots = []
    keys = []
    for metric in metrics:
        pretty = (pretty_names or {}).get(metric)
        piv = summarize_metric_pivot_with_interval(
            data, metric, interval, pretty_metric_name=pretty, row_filter=row_filter
        )
        pivots.append(piv)
        keys.append(
            pretty
            or {
                "adozott_eredmeny_ft": "Adózott eredmény (Ft)",
                "ebitda_ft": "EBITDA (Ft)",
                "netto_hozzadott_ertek": "Nettó hozzáadott érték (Ft)",
                "brutto_hozzadott_ertek": "Bruttó hozzáadott érték (Ft)",
            }.get(metric, metric)
        )

    out = pd.concat(pivots, axis=1, keys=keys)
    out = out.sort_index(axis=1, level=list(range(out.columns.nlevels)))
    return out


def summarize_metric_pivot(
    data: pd.DataFrame,
    metric: str,
    *,
    pretty_metric_name: str | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    grp0 = compute_data(data, metric, ("ev", "tip_m10ste"), None, row_filter)
    grp1 = compute_data(data, metric, ("ev",), None, row_filter)
    grp1.insert(1, "tip_m10ste", "teljes_mgi")

    grp = pd.concat([grp0, grp1], ignore_index=True)

    pivot = (
        grp.pivot(index="tip_m10ste", columns="ev", values=list(DISPLAY_COLS.keys()))
        .rename(columns=DISPLAY_COLS, level=0)
        .reindex(list(DISPLAY_COLS.values()), axis=1, level=0)
        .loc[ROW_ORDER]
    )
    return pivot


def summarize_multi_metrics_pivot(
    data: pd.DataFrame,
    metrics: Iterable[str],
    *,
    group_cols: Iterable[str] = ("ev", "tip_m10ste"),
    pretty_names: Mapping[str, str] | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    pivots = []
    keys = []
    for metric in metrics:
        pretty = (pretty_names or {}).get(metric)
        piv = summarize_metric_pivot(
            data, metric, pretty_metric_name=pretty, row_filter=row_filter
        )
        pivots.append(piv)
        keys.append(
            pretty
            or {
                "adozott_eredmeny_ft": "Adózott eredmény (Ft)",
                "ebitda_ft": "EBITDA (Ft)",
                "netto_hozzadott_ertek": "Nettó hozzáadott érték (Ft)",
                "brutto_hozzadott_ertek": "Bruttó hozzáadott érték (Ft)",
            }.get(metric, metric)
        )

    out = pd.concat(pivots, axis=1, keys=keys)
    out = out.sort_index(axis=1, level=list(range(out.columns.nlevels)))
    return out


def plot_support_difference_vs_biss_area(data: pd.DataFrame) -> None:
    y = data["tamogatasok_osszesen_ft_dabis"] - data["tamogatasok_osszesen_ft"]
    X = data["biss_ha"]

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.6)
    plt.xlabel("BISS (ha)")
    plt.ylabel("Támogatások különbsége (Dabis - nem Dabis) [Ft]")
    plt.title("A BISS terület és a támogatáskülönbség kapcsolata")
    plt.grid(True)
    plt.show()
