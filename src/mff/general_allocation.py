import geopandas as gpd
import mapclassify
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from mff.new_cap import c_round
from shapely.geometry import box

from typing import cast

# data sources: * NUTS 3 population data: https://ec.europa.eu/eurostat/databrowser/view/demo_r_pjanaggr3/default/table
#               * Available budget of Cohesion Policy 2021-2027: https://ec.europa.eu/regional_policy/funding/available-budget_en
#               * NUTS 3 level GDP data: nama_10r_3gdp
#               * PEA 2022: https://agriculture.ec.europa.eu/document/download/8707d160-1fed-45b1-aba8-bc9a6eca9846_en?filename=summary-report-implementation-direct-payements-claim-2022_en.pdf


def read_data_country_lvl() -> pl.DataFrame:
    return pl.read_excel("input/database_country.xlsx")


def read_data_nuts3_lvl() -> pl.DataFrame:
    return (
        pl.read_excel("input/database_nuts3.xlsx")
        .filter(pl.col("geo_codes").str.len_chars() == 5)
        .with_columns(pl.col("geo_codes").str.slice(0, 2).alias("country_code"))
    )


def calculate_agri_prosperity_gap(data_country: pl.DataFrame) -> pl.DataFrame:
    dp_per_ha_eu = data_country.select(pl.sum("dp_2027") / pl.sum("pea_2022")).item()

    dp_per_ha_country = pl.col("dp_2027") / pl.col("pea_2022")
    pea_share = pl.col("pea_2022") / pl.col("dp_2027")
    gap_component = (0.9 * dp_per_ha_eu) - dp_per_ha_country

    result = data_country.with_columns(
        [
            dp_per_ha_country.alias("dp_per_ha_country"),
            (gap_component.clip(lower_bound=0) * pea_share).alias(
                "agri_prosperity_gap"
            ),
        ]
    )

    return result.select("geo_codes", "agri_prosperity_gap")


def calculate_regional_prosperity_gap(data_nuts3: pl.DataFrame) -> pl.DataFrame:
    eu_avg_gdp = 35767

    return (
        data_nuts3.with_columns(
            (
                (0.75 - (pl.col("gdp_pc_pps_nuts3") / eu_avg_gdp)).clip(lower_bound=0)
                * (pl.col("pop_nuts3") / pl.sum("pop_nuts3").over("country_code"))
            ).alias("regional_prosperity_gap")
        )
        .group_by("country_code")
        .agg(pl.sum("regional_prosperity_gap"))
        .rename({"country_code": "geo_codes"})
    )


def calculate_gni_multiplier(data_country: pl.DataFrame) -> pl.DataFrame:
    gni_pc_pps_2023_eu_avg = 38093

    return data_country.select(
        [
            pl.col("geo_codes"),
            (gni_pc_pps_2023_eu_avg / pl.col("gni_pc_pps_2023")).alias(
                "gni_multiplier"
            ),
        ]
    )


def calculate_product_part1(data_country: pl.DataFrame) -> pl.DataFrame:
    return data_country.select(
        [
            pl.col("geo_codes"),
            (
                (
                    data_country["population_2024"]
                    / data_country["population_2024"].sum()
                    + data_country["arope_ra_1000_pop_2024"]
                    / data_country["arope_ra_1000_pop_2024"].sum()
                )
                / 2
            ).alias("product_part1"),
        ]
    )


def calculate_general_allocation(
    agri_prosperity_gap: pl.DataFrame,
    regional_prosperity_gap: pl.DataFrame,
    gni_multiplier: pl.DataFrame,
    product_part1: pl.DataFrame,
) -> pl.DataFrame:
    result_pl = (
        agri_prosperity_gap.join(regional_prosperity_gap, on="geo_codes", how="left")
        .join(gni_multiplier, on="geo_codes", how="left")
        .join(product_part1, on="geo_codes", how="left")
    )

    product_part2_expr = (
        pl.col("gni_multiplier")
        * (1 + pl.col("regional_prosperity_gap") + pl.col("agri_prosperity_gap")) ** 2
    ) ** 2

    result_pl = result_pl.with_columns(product_part2_expr.alias("product_part2"))

    result_pl = result_pl.with_columns(
        (pl.col("product_part1") * pl.col("product_part2")).alias("product")
    )

    # normalize a_i
    result_pl = result_pl.with_columns(
        (result_pl["product"] * 1 / result_pl["product"].sum()).alias("a_i")
    )
    result_pl = result_pl.with_columns(
        (748.9 * result_pl["a_i"]).alias("allocation_general")
    )

    result_pd: pd.DataFrame = result_pl.to_pandas()

    countries: gpd.GeoDataFrame = gpd.read_file(
        "input/CNTR_RG_20M_2024_3035.gpkg", columns=["CNTR_ID", "geometry"]
    )
    countries = countries.rename(columns={"CNTR_ID": "geo_codes"})

    result_gdf: gpd.GeoDataFrame = countries.merge(
        result_pd, on="geo_codes", how="right"
    )

    if result_gdf.crs is None:
        result_gdf = result_gdf.set_crs(epsg=3035)

    xmin, xmax = 2500000, 6500000
    ymin, ymax = 1400000, 5500000

    bbox = gpd.GeoDataFrame(
        {"geometry": [box(xmin, ymin, xmax, ymax)]}, crs=result_gdf.crs
    )

    result_gdf = gpd.clip(result_gdf, bbox)

    res: pd.DataFrame = pd.DataFrame(
        result_gdf[
            [
                "geo_codes",
                "agri_prosperity_gap",
                "regional_prosperity_gap",
                "a_i",
                "allocation_general",
            ]
        ]
        .copy()
        .drop(columns="geometry", errors="ignore")
    )

    res["a_i"] = 100 * res["a_i"]
    res = res.rename(columns={"a_i": "a_i_pct"})

    com_data = pd.read_excel("input/general_allocation.xlsx", engine="calamine")
    res = res.merge(com_data, on="geo_codes")

    loc_a_i_pct = cast(int, res.columns.get_loc("a_i_pct"))

    res.insert(
        loc_a_i_pct + 1,
        "a_i_pct_corrected",
        100
        * res["general_allocation_corrected"]
        / res["general_allocation_corrected"].sum(),
    )

    res = res[
        [
            "geo_codes",
            "agri_prosperity_gap",
            "regional_prosperity_gap",
            "a_i_pct",
            "allocation_general",
            "a_i_pct_corrected",
            "general_allocation_corrected",
        ]
    ]

    res["a_i_pct_point_diff"] = res["a_i_pct_corrected"] - res["a_i_pct"]

    cols = res.select_dtypes(include="number").columns
    res[cols] = res[cols].map(lambda x: c_round(x, 2))
    return pl.from_pandas(res)


def highlight_diff(row: pd.Series) -> list[str]:
    color = "palegreen" if row["a_i_pct_point_diff"] > 0 else "lightcoral"
    return [f"background-color: {color}" for _ in row]


def gradient_row(row: pd.Series) -> list[str]:
    val = row["a_i_pct_point_diff"]
    if pd.isna(val):
        return [""] * len(row)
    if val > 0:
        intensity = min(255, 255 - int(val * 200))
        color = f"background-color: rgb({intensity}, 255, {intensity})"
    else:
        intensity = min(255, 255 - int(abs(val) * 200))
        color = f"background-color: rgb(255, {intensity}, {intensity})"
    return [color] * len(row)


def plot_agri_prosperity_gap(result) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    result.plot(
        column="agri_prosperity_gap",
        cmap="OrRd",
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        missing_kwds={"color": "lightgrey"},
    )  # optional for NaN

    if (result["agri_prosperity_gap"] == 0).any():
        zeros_agri = result["agri_prosperity_gap"] == 0
        result[zeros_agri].plot(
            ax=ax,
            facecolor="white",
            edgecolor="blue",
            linewidth=1.2,
        )

    for idx, row in result.iterrows():
        if not np.isnan(row["agri_prosperity_gap"]):
            centroid = row.geometry.centroid
            label_text = (
                f"{c_round(row['agri_prosperity_gap'], 2)}\n({row['geo_codes']})"
            )
            pe = [withStroke(linewidth=1, foreground="white")]
            fontsize = 6
            ax.text(
                centroid.x,
                centroid.y,
                label_text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black",
                path_effects=pe,
            )

    ax.set_title("Agri Prosperity Gap")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("output/agri_prosperity_gap.svg", bbox_inches="tight")
    plt.show()


def plot_regional_prosperity_gap(result) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    result.plot(
        column="regional_prosperity_gap",
        cmap="OrRd",
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        missing_kwds={"color": "lightgrey"},
    )  # optional for NaN

    if (result["regional_prosperity_gap"] == 0).any():
        zeros_agri = result["regional_prosperity_gap"] == 0
        result[zeros_agri].plot(
            ax=ax,
            facecolor="white",
            edgecolor="blue",
            linewidth=1.2,
        )

    for idx, row in result.iterrows():
        if not np.isnan(row["regional_prosperity_gap"]):
            centroid = row.geometry.centroid
            label_text = (
                f"{c_round(row['regional_prosperity_gap'], 2)}\n({row['geo_codes']})"
            )
            pe = [withStroke(linewidth=1, foreground="white")]
            fontsize = 6
            ax.text(
                centroid.x,
                centroid.y,
                label_text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black",
                path_effects=pe,
            )

    ax.set_title("Regional Prosperity Gap")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("output/regional_prosperity_gap.svg", bbox_inches="tight")
    plt.show()


def make_jenks_bins(series: pd.Series, k: int = 3):
    nonzero = series.dropna()
    classifier = mapclassify.NaturalBreaks(nonzero, k=k)
    edges = [-np.inf] + list(classifier.bins)
    labels = [f"{edges[i]:.2f} – {edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True), labels, edges


def plot_agri_and_regional_prosperity_gap(result):
    if not isinstance(result, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result, geometry="geometry")

    if result.crs is None:
        result.set_crs(epsg=3035, inplace=True)

    # --- mask zeros ---
    m_agri = result["agri_prosperity_gap"].where(result["agri_prosperity_gap"] != 0)
    m_reg = result["regional_prosperity_gap"].where(
        result["regional_prosperity_gap"] != 0
    )

    # --- Jenks binning with readable labels ---
    result["agri_bins"], agri_labels, agri_edges = make_jenks_bins(m_agri, k=3)
    result["regional_bins"], reg_labels, reg_edges = make_jenks_bins(m_reg, k=3)

    # discrete colormaps
    cmap_agri = mpl.colormaps["YlOrRd"].resampled(len(agri_labels))
    cmap_reg = mpl.colormaps["YlGnBu"].resampled(len(reg_labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Agri ---
    result.plot(
        column="agri_bins",
        cmap=cmap_agri,
        legend=False,
        ax=axes[0],
        edgecolor="black",
        linewidth=0.5,
        missing_kwds={"color": "lightgrey"},
    )

    # Highlight zeros with hatch
    zeros_agri = result["agri_prosperity_gap"] == 0
    result[zeros_agri].plot(
        ax=axes[0], facecolor="lightgrey", edgecolor="blue", linewidth=1.2, hatch="///"
    )

    # Custom legend
    agri_patches = [
        Patch(facecolor=cmap_agri(i), label=lab, edgecolor="black")
        for i, lab in enumerate(agri_labels)
    ]
    axes[0].legend(
        handles=agri_patches,
        title="Gap Range",
        loc="upper left",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    axes[0].set_title("Agri Prosperity Gap (Jenks)")
    axes[0].axis("off")
    axes[0].set_xlim(2500000, 6000000)
    axes[0].set_ylim(1400000, 5500000)

    # --- Regional ---
    result.plot(
        column="regional_bins",
        cmap=cmap_reg,
        legend=False,
        ax=axes[1],
        edgecolor="black",
        linewidth=0.5,
        missing_kwds={"color": "lightgrey"},
    )

    zeros_regional = result["regional_prosperity_gap"] == 0
    result[zeros_regional].plot(
        ax=axes[1], facecolor="lightgrey", edgecolor="blue", linewidth=1.2, hatch="///"
    )

    # Custom legend
    reg_patches = [
        Patch(facecolor=cmap_reg(i), label=lab, edgecolor="black")
        for i, lab in enumerate(reg_labels)
    ]
    axes[1].legend(
        handles=reg_patches,
        title="Gap Range",
        loc="upper left",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    axes[1].set_title("Regional Prosperity Gap (Jenks)")
    axes[1].axis("off")
    axes[1].set_xlim(2500000, 6000000)
    axes[1].set_ylim(1400000, 5500000)

    plt.tight_layout()
    plt.show()


# # Compute bounds
# lower_bound = 0.8 * result["a_i_ref"]
# upper_bound = 1.05 * result["a_i_ref"]
#
# # Start with clipping
# result = result.with_columns(
#     [
#         pl.col("a_i")
#         .clip(lower_bound=pl.lit(lower_bound), upper_bound=pl.lit(upper_bound))
#         .alias("a_i_clipped")
#     ]
# )
#
# # Function to iteratively rescale within bounds
#
#
# def bounded_normalize(
#     df, value_col, lower_bound, upper_bound, target_sum=1.0, tol=1e-10, max_iter=100
# ):
#     values = df[value_col].to_numpy().copy()  # Make it writable
#     fixed = np.zeros(len(values), dtype=bool)
#
#     for _ in range(max_iter):
#         free = ~fixed
#         total_free = values[free].sum()
#
#         if total_free == 0:
#             break
#
#         scale = (target_sum - values[fixed].sum()) / total_free
#         values[free] *= scale
#
#         # Clip and fix values that exceed bounds
#         exceeded_upper = values > upper_bound
#         below_lower = values < lower_bound
#
#         values = np.clip(values, lower_bound, upper_bound)
#         fixed |= exceeded_upper | below_lower
#
#         if abs(values.sum() - target_sum) < tol:
#             break
#
#     return values
#
#
# # Apply iterative bounded normalization
# final_a_i = bounded_normalize(
#     result,
#     "a_i_clipped",
#     lower_bound=lower_bound.to_numpy(),
#     upper_bound=upper_bound.to_numpy(),
# )
#
# # Add the final result to the DataFrame
# result = result.with_columns(pl.Series("a_i_final", final_a_i))
