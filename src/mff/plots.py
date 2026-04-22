from collections.abc import Iterable
from dataclasses import dataclass
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter, MultipleLocator

from utils import c_round
from mff.new_cap import (
    maximize_ratio,
    calc_ratio_subs,
    apply_reductions,
    cal_redist,
    compute_current_support,
    calc_thresholds,
    find_cur_new_equal_root,
    compute_capped_subsidies,
    analyze_by_area_categories,
)

formatter = FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))


@dataclass(frozen=True)
class PolicyParams:
    base_payment_per_ha: float
    yfs_per_ha: float
    redist_params: tuple[float, float]
    yfs_cap_ha: float = 300.0


def plot_per_ha(
    policy: PolicyParams,
    label1: str,
    label2: str,
    title: str,
    output_path: str,
) -> None:
    ha_upper_l = 255000 / policy.base_payment_per_ha
    x_values = np.linspace(0.01, ha_upper_l, 100 * int(ha_upper_l / 10) + 1)

    y_values_capping_per_ha = [
        apply_reductions(
            policy.base_payment_per_ha * x
            + policy.yfs_per_ha * min(policy.yfs_cap_ha, x)
            + cal_redist(x, policy.redist_params)
        )
        / x
        for x in x_values
    ]

    y_values_current_per_ha = [
        compute_current_support(x, policy.yfs_per_ha) / x for x in x_values
    ]

    zero_cross_x = c_round(
        find_cur_new_equal_root(
            policy.base_payment_per_ha, policy.redist_params, policy.yfs_per_ha
        ),
        2,
    )

    thresholds = [20_000, 50_000, 75_000, 255_000]
    thresholds_ha = calc_thresholds(
        thresholds, policy.base_payment_per_ha, policy.redist_params, policy.yfs_per_ha
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        x_values, y_values_capping_per_ha, label=label1, color="#0072B2", linewidth=2.5
    )
    ax.plot(
        x_values,
        y_values_current_per_ha,
        label=label2,
        color="#D55E00",
        linestyle="--",
        linewidth=2.5,
    )

    for t, th_ha in zip(thresholds, thresholds_ha):
        ax.axvline(x=th_ha, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
        label = (
            f"Capping határ: 255 000 €\n({c_round(th_ha, 0):,.0f} ha)".replace(",", " ")
            if t == 255_000
            else f"{t:,.0f} €\n({c_round(th_ha, 0):,.0f} ha)".replace(",", " ")
        )
        ax.text(
            th_ha + 5,
            max(y_values_capping_per_ha) * 0.42,
            label,
            rotation=45,
            verticalalignment="bottom",
            color="red",
            fontsize=10,
        )

    ax.axvline(
        x=zero_cross_x,
        color="#0072B2",
        linestyle="--",
        linewidth=1.5,
    )
    label = f"Egyenlőségi pont\n({zero_cross_x:,.0f} ha)"

    ax.text(
        zero_cross_x + 5,
        max(y_values_capping_per_ha) * 0.85,
        label,
        rotation=45,
        verticalalignment="bottom",
        color="#0072B2",
        fontsize=10,
    )

    ax.set_xlabel("Üzemméret (ha)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Támogatás hektáronként (€)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.text(
        x_values[-1],
        y_values_capping_per_ha[-1],
        f"{y_values_capping_per_ha[-1]:,.0f} €/ha",
        fontsize=10,
        color="#0072B2",
        va="bottom",
    )
    ax.text(
        x_values[-1],
        y_values_current_per_ha[-1],
        f"{y_values_current_per_ha[-1]:,.0f} €/ha",
        fontsize=10,
        color="#D55E00",
        va="bottom",
    )

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.legend(loc="upper right", fontsize=11, frameon=True)
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_total(
    policy: PolicyParams,
    label1: str,
    label2: str,
    title: str,
    output_path: str,
) -> None:
    ha_upper_l = 255000 / policy.base_payment_per_ha
    x_values = np.linspace(0.01, ha_upper_l, 100 * int(ha_upper_l / 10) + 1)

    y_values_capping = [
        apply_reductions(
            policy.base_payment_per_ha * x
            + policy.yfs_per_ha * min(policy.yfs_cap_ha, x)
            + cal_redist(x, policy.redist_params)
        )
        for x in x_values
    ]
    y_values_current = [compute_current_support(x, policy.yfs_per_ha) for x in x_values]

    zero_cross_x = c_round(
        find_cur_new_equal_root(
            policy.base_payment_per_ha, policy.redist_params, policy.yfs_per_ha
        ),
        2,
    )

    thresholds = [20_000, 50_000, 75_000, 255_000]
    thresholds_ha = calc_thresholds(
        thresholds, policy.base_payment_per_ha, policy.redist_params, policy.yfs_per_ha
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(x_values, y_values_capping, label=label1, color="#0072B2", linewidth=2.5)
    ax.plot(
        x_values,
        y_values_current,
        label=label2,
        color="#D55E00",
        linestyle="--",
        linewidth=2.5,
    )

    for t, th_ha in zip(thresholds, thresholds_ha):
        ax.axvline(x=th_ha, color="red", linestyle="--", linewidth=1, alpha=0.6)
        label = (
            f"Capping határ: 255 000 €\n({c_round(th_ha, 0):,.0f} ha)".replace(",", " ")
            if t == 255_000
            else f"{t:,.0f} €\n({c_round(th_ha, 0):,.0f} ha)".replace(",", " ")
        )
        ax.text(
            th_ha + 5,
            max(y_values_capping) * 0.05,
            label,
            rotation=45,
            verticalalignment="bottom",
            color="red",
            fontsize=10,
        )

    ax.axvline(
        x=zero_cross_x,
        color="#0072B2",
        linestyle="--",
        linewidth=1.5,
    )
    label = f"Egyenlőségi pont\n({zero_cross_x:,.0f} ha)"

    ax.text(
        zero_cross_x + 5,
        max(y_values_capping) * 0.85,
        label,
        rotation=45,
        verticalalignment="bottom",
        color="#0072B2",
        fontsize=10,
    )

    ax.set_xlabel("Üzemméret (ha)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Támogatás (€)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.text(
        x_values[-1],
        y_values_capping[-1],
        f"{y_values_capping[-1]:,.0f}".replace(",", " ") + " €",
        fontsize=10,
        color="#0072B2",
        va="bottom",
    )
    ax.text(
        x_values[-1],
        y_values_current[-1],
        f"{y_values_current[-1]:,.0f}".replace(",", " ") + " €",
        fontsize=10,
        color="#D55E00",
        va="bottom",
    )

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.legend(loc="upper left", fontsize=11, frameon=True)
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_diff_dual_axis(
    policy: PolicyParams,
    title_name: str,
    scen_name: str,
    farm_areas: list[float] | np.ndarray,
    output_path: str,
    cum_mode: str = "farms",
):
    ha_upper_l = 255000 / policy.base_payment_per_ha
    x_values = np.linspace(0.01, ha_upper_l, 100 * int(ha_upper_l / 10) + 1)

    y_values_capping = [
        apply_reductions(
            policy.base_payment_per_ha * x
            + policy.yfs_per_ha * min(policy.yfs_cap_ha, x)
            + cal_redist(x, policy.redist_params)
        )
        for x in x_values
    ]
    y_values_current = [compute_current_support(x, policy.yfs_per_ha) for x in x_values]

    diff_pct = [
        (cap - curr) / curr * 100 if curr != 0 else 0
        for cap, curr in zip(y_values_capping, y_values_current)
    ]

    peak_x = maximize_ratio(
        50, policy.base_payment_per_ha, policy.yfs_per_ha, policy.redist_params
    )
    peak_x_rounded = int(c_round(peak_x, 0))
    peak_y = 100 * calc_ratio_subs(
        peak_x, policy.base_payment_per_ha, policy.yfs_per_ha, policy.redist_params
    )
    peak_y_rounded = c_round(peak_y, 2)

    zero_cross_x = find_cur_new_equal_root(
        policy.base_payment_per_ha, policy.redist_params, policy.yfs_per_ha
    )

    fa = np.asarray(farm_areas, dtype=float)
    fa = fa[(fa > 0) & (fa <= ha_upper_l)]
    fa_sorted = np.sort(fa)

    n = len(fa_sorted)
    cum_farm_pct = np.arange(1, n + 1) / n * 100  # 1..n / n * 100
    x_cum = fa_sorted

    num_farms = (fa_sorted <= zero_cross_x).sum()
    pct_farms = num_farms / len(fa_sorted) * 100

    total_area = fa_sorted.sum()
    area_below = fa_sorted[fa_sorted <= zero_cross_x].sum()
    pct_area = area_below / total_area * 100

    cum_area_pct = np.cumsum(fa_sorted) / total_area * 100

    textbox_text = f"Az üzemek {pct_farms:.1f}%-a és a terület {pct_area:.1f}%-a van {zero_cross_x:.2f} ha alatt."

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_axisbelow(True)

    ax1.plot(
        x_values,
        diff_pct,
        color="#009E73",
        linewidth=2.2,
        label=scen_name,
    )
    ax1.set_ylabel("Eltérés (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Üzemméret (ha)", fontsize=12, fontweight="bold")
    ax1.axhline(0, linestyle="--", color="black", linewidth=1)

    ax1.yaxis.set_major_locator(MultipleLocator(10))

    step = 10

    values_for_ylim = diff_pct + [peak_y]

    data_min = min(values_for_ylim)
    data_max = max(values_for_ylim)

    ymin = step * np.floor(data_min / step)
    ymax = step * np.ceil(1.05 * data_max / step)

    ax1.set_ylim(ymin, ymax)

    ax1.axvline(peak_x, color="red", linestyle="--", linewidth=1.5, label="Maximum")
    ax1.axvline(
        x=zero_cross_x,
        color="#56B4E9",
        linestyle="--",
        linewidth=1.5,
        label="Egyenlőségi pont",
    )

    x_offset = 60
    y_offset = -5 if peak_y > max(diff_pct) * 0.8 else 5

    ax1.annotate(
        f"{peak_y_rounded}% ({peak_x_rounded} ha)",
        xy=(peak_x, peak_y),
        xytext=(peak_x + x_offset, peak_y + y_offset),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=11,
        fontweight="bold",
        color="red",
    )

    ax1.annotate(
        f"{int(c_round(zero_cross_x, 0))} ha",
        xy=(zero_cross_x, 0),
        xytext=(zero_cross_x + x_offset, 0 + y_offset),
        arrowprops=dict(arrowstyle="->", color="#56B4E9"),
        fontsize=11,
        fontweight="bold",
        color="#56B4E9",
    )

    ax2 = ax1.twinx()

    if cum_mode == "area":
        y2 = cum_area_pct
        y2_label = "Kumulált területarány (%)"
    else:
        y2 = cum_farm_pct
        y2_label = "Kumulált üzemszámarány (%)"

    ax2.plot(
        x_cum,
        y2,
        color="0.3",
        linewidth=2,
        linestyle="-",
        label=y2_label,
    )
    ax2.set_ylabel(y2_label, fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(PercentFormatter())

    ax1.set_title(
        title_name,
        fontsize=14,
        fontweight="bold",
    )

    try:
        ax1.xaxis.set_major_formatter(formatter)
        ax1.yaxis.set_major_formatter(formatter)
    except NameError:
        pass

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    x_ref = x_values[int(len(x_values) * 0.8)]

    idx2 = np.searchsorted(x_cum, x_ref, side="right") - 1
    idx2 = np.clip(idx2, 0, len(x_cum) - 1)
    grey_y = y2[idx2]

    grey_disp = ax2.transData.transform((x_ref, grey_y))
    grey_axes = ax1.transAxes.inverted().transform(grey_disp)
    y_grey_frac = grey_axes[1]

    candidates = [0.9, 0.8, 0.7, 0.6]
    text_y_pos = next(
        (c for c in candidates if abs(c - y_grey_frac) > 0.12),
        0.85,
    )

    _ = ax1.text(
        0.98,
        text_y_pos,
        textbox_text,
        transform=ax1.transAxes,
        fontsize=11,
        color="black",
        ha="right",
        va="top",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            boxstyle="round,pad=0.4",
            alpha=1.0,
        ),
        zorder=10,
        clip_on=False,
    )

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_reduction(
    policy: PolicyParams,
    title_name: str,
    scen_name: str,
    output_path: str,
) -> None:
    ha_upper_l = 255000 / policy.base_payment_per_ha
    x_values = np.linspace(0.01, ha_upper_l, 100 * int(ha_upper_l / 10) + 1)

    y_values_before_capping = [
        policy.base_payment_per_ha * x
        + policy.yfs_per_ha * min(policy.yfs_cap_ha, x)
        + cal_redist(x, policy.redist_params)
        for x in x_values
    ]
    y_values_after_capping = [apply_reductions(y) for y in y_values_before_capping]

    percent_diff = [
        cap / no_cap * 100 if no_cap != 0 else 0
        for cap, no_cap in zip(y_values_after_capping, y_values_before_capping)
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        x_values,
        percent_diff,
        label=scen_name,
        color="#009E73",
        linewidth=2.5,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Üzemméret (ha)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Megmaradó támogatás aránya (%)", fontsize=12, fontweight="bold")
    ax.set_title(title_name, fontsize=14, fontweight="bold")

    ax.legend(loc="best", fontsize=11, frameon=True)

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_diff_pct(
    policy: PolicyParams,
    title_name: str,
    scen_name: str,
    output_path: str,
) -> None:
    ha_upper_l = 255000 / policy.base_payment_per_ha
    x_values = np.linspace(0.01, ha_upper_l, 100 * int(ha_upper_l / 10) + 1)

    y_values_capping = [
        apply_reductions(
            policy.base_payment_per_ha * x
            + policy.yfs_per_ha * min(policy.yfs_cap_ha, x)
            + cal_redist(x, policy.redist_params)
        )
        for x in x_values
    ]
    y_values_current = [compute_current_support(x, policy.yfs_per_ha) for x in x_values]

    percent_diff = [
        (cap - curr) / curr * 100 if curr != 0 else 0
        for cap, curr in zip(y_values_capping, y_values_current)
    ]

    peak_x = maximize_ratio(
        50, policy.base_payment_per_ha, policy.yfs_per_ha, policy.redist_params
    )
    peak_x_rounded = int(c_round(peak_x, 0))
    peak_y = 100 * calc_ratio_subs(
        peak_x, policy.base_payment_per_ha, policy.yfs_per_ha, policy.redist_params
    )
    peak_y_rounded = c_round(peak_y, 2)

    zero_cross_x = find_cur_new_equal_root(
        policy.base_payment_per_ha, policy.redist_params, policy.yfs_per_ha
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        x_values,
        percent_diff,
        label=scen_name,
        color="#009E73",
        linewidth=2.5,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(peak_x, color="red", linestyle="--", linewidth=1.5, label="Maxmimum")
    ax.axvline(
        x=zero_cross_x, color="#56B4E9", linestyle="--", label="Egyenlőségi pont"
    )

    x_offset = 60
    y_offset = -5 if peak_y > max(percent_diff) * 0.8 else 5

    ax.annotate(
        f"{peak_y_rounded}% ({peak_x_rounded} ha)",
        xy=(peak_x, peak_y),
        xytext=(peak_x + x_offset, peak_y + y_offset),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=11,
        fontweight="bold",
        color="red",
    )

    ax.annotate(
        f"{int(c_round(zero_cross_x, 0))} ha",
        xy=(zero_cross_x, 0),
        xytext=(zero_cross_x + x_offset, 0 + y_offset),
        arrowprops=dict(arrowstyle="->", color="#56B4E9"),
        fontsize=11,
        fontweight="bold",
        color="#56B4E9",
    )

    ax.set_xlabel("Üzemméret (ha)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Eltérés (%)", fontsize=12, fontweight="bold")
    ax.set_title(title_name, fontsize=14, fontweight="bold")

    ax.legend(loc="best", fontsize=11, frameon=True)

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_allocation_with_fixed_rate(
    payment_rate: float,
    subs_per_acre_results: list[float],
    alloc_results: list[float],
    values: list[float],
):
    fig, ax = plt.subplots(figsize=(5, 3))

    x_values = subs_per_acre_results
    ax.plot(
        x_values,
        alloc_results,
        color="royalblue",
        linewidth=3,
        label="Total Allocation Needed",
        zorder=2,
    )
    ax.set_xlabel("Support per Hectare (EUR)", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Total Allocation Needed (Million EUR)",
        color="royalblue",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelcolor="royalblue")

    ax = ax.twinx()
    ax.plot(
        x_values,
        values,
        color="darkorange",
        linewidth=3,
        linestyle="--",
        label="Payment Rate (EUR/ha)",
        zorder=2,
    )
    ax.set_ylabel(
        "Payment Rate (EUR/ha)", color="darkorange", fontsize=12, fontweight="bold"
    )
    ax.tick_params(axis="y", labelcolor="darkorange")

    payment_rate_fixed = payment_rate
    ax.axhline(
        payment_rate_fixed,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Fixed Payment Rate",
        zorder=3,
    )

    idx = (np.abs(np.array(values) - payment_rate_fixed)).argmin()
    intersection_x = round(x_values[idx], 2)
    intersection_y1 = round(alloc_results[idx], 2)
    intersection_y2 = round(values[idx], 2)

    if abs(intersection_y2 - payment_rate) < 0.05:
        intersection_y2 = payment_rate

    textstr = (
        f"{'Support per Hectare:':<20}{intersection_x:>8.2f} EUR/ha\n"
        f"{'Allocation Needed:':<20}{intersection_y1:>8.2f} M EUR\n"
        f"{'Payment Rate:':<20}{intersection_y2:>8.2f} EUR/ha"
    )

    x_rel = (intersection_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
    y_rel = (intersection_y1 - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])

    fig.text(
        x_rel + 0.02,
        y_rel + 0.02,
        textstr,
        fontsize=11,
        ha="left",
        va="bottom",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.95,
        ),
        zorder=50,
    )

    ax.grid(color="lightgrey", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.title(
        "Impact of Support Rate on Total Allocation (Fixed Payment Rate Highlighted)",
        fontsize=14,
        fontweight="bold",
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plt.tight_layout()
    plt.show()


def add_bar_labels(
    ax: Axes,
    bars: BarContainer | Iterable[Rectangle],
    fmt: str = "{:,}",
    percent: bool = False,
) -> None:
    max_val = float("-inf")
    min_val = float("inf")

    for bar in bars:
        h = bar.get_height()
        label = f"{h:.1f} %" if percent else fmt.format(int(h)).replace(",", " ")

        # lefelé kerül a negatív felirat
        offset = 5 if h >= 0 else -5

        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9,
        )
        max_val = max(max_val, h)
        min_val = min(min_val, h)

    data_range = max_val - min_val if max_val != min_val else abs(max_val)
    margin = data_range * 0.155 if data_range > 0 else abs(max_val) * 0.155

    if min_val >= 0:
        ax.set_ylim(0, max_val + margin)
    else:
        ax.set_ylim(min(0, min_val - margin), max_val + margin)


def plot_per_ha_support_comparison_by_area_class(
    data: pd.DataFrame,
    dabis_per_ha: float,
    redist_per_ha: tuple[float, float],
    bins: list[float],
    labels: list[str],
    allocation: float,
) -> None:
    data_with_subs = compute_capped_subsidies(data, dabis_per_ha, 0, redist_per_ha)
    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"] + data_with_subs["subs_redist"]
    )

    plot_data = analyze_by_area_categories(
        data_with_subs, bins=bins, labels=labels
    ).loc[:, ["avg_subs_per_ha_cur", "avg_subs_per_ha_capped"]]

    x = np.arange(len(plot_data))  # number of categories
    width = 0.35  # width of bars

    plt.figure(figsize=(8, 5))

    color_current = "#0072B2"
    color_capped = "#009E73"

    bars1 = plt.bar(
        x - width / 2,
        plot_data["avg_subs_per_ha_cur"],
        width,
        label="Alaptámogatás + Redisztribúció - 2024",
        color=color_current,
    )

    bars2 = plt.bar(
        x + width / 2,
        plot_data["avg_subs_per_ha_capped"],
        width,
        label="Alaptámogatás                  - új MFF",
        color=color_capped,
    )

    plt.ylabel("Fajlagos támogatás (EUR/ha)", fontsize=11)
    plt.xlabel("Méretkategória", fontsize=11)
    plt.title(
        f"Átlagos fajlagos támogatás gazdálkodói méretkategóriák szerint\n{allocation / 1e6:.1f} millió EUR borítékkal számolva".replace(
            ".", ","
        ),
        fontsize=12,
    )
    plt.xticks(x, plot_data.index.astype(str).tolist(), rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(prop={"family": "monospace"})

    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{c_round(height, 2)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars2:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{c_round(height, 2)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(f"output/abra_fajlagos_tam_{allocation / 1e6:.1f}.png", dpi=300)
    plt.show()


def plot_support_summary_by_area_class(
    data: pd.DataFrame,
    dabis_per_ha: float,
    redist_per_ha: tuple[float, float],
    bins: list[float],
    labels: list[str],
    allocation: float,
) -> None:
    data_with_subs = compute_capped_subsidies(data, dabis_per_ha, 0, redist_per_ha)
    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"] + data_with_subs["subs_redist"]
    )

    grouped_result = analyze_by_area_categories(
        data_with_subs, bins=bins, labels=labels
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    bars1 = ax1.bar(
        grouped_result.index,
        grouped_result["total_farmers"],
        color="#0072B2",
        alpha=0.85,
    )
    ax1.set_ylabel("Gazdálkodók száma (db)", fontsize=12)
    ax1.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
    )
    add_bar_labels(ax1, bars1)

    bars2 = ax2.bar(
        grouped_result.index, grouped_result["total_area"], color="#56B4E9", alpha=0.85
    )
    ax2.set_ylabel("Összes terület (ha)", fontsize=12)
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
    )
    add_bar_labels(ax2, bars2)

    bars3 = ax3.bar(
        grouped_result.index,
        grouped_result["avg_change"],
        color=[
            "#009E73" if v >= 0 else "#B22222" for v in grouped_result["avg_change"]
        ],
        alpha=0.85,
    )
    ax3.set_ylabel("Fajlagos támogatás változása (%)", fontsize=12)
    ax3.axhline(0, color="gray", linestyle="--", linewidth=1)
    add_bar_labels(ax3, bars3, percent=True)

    ax3.set_xlabel("Méretkategória (ha)", fontsize=12)

    fig.suptitle(
        f"Támogatások alakulása gazdálkodói méretkategóriák szerint\n{allocation / 1e6:.1f} millió EUR borítékkal számolva".replace(
            ".", ","
        ),
        fontsize=14,
        fontweight="bold",
        x=0.5,
        y=0.95,
    )

    fig.align_ylabels([ax1, ax2, ax3])
    plt.subplots_adjust(left=0.12, hspace=0.2)
    plt.savefig(f"output/abra_osszetett_{allocation / 1e6:.1f}.png", dpi=400)
    plt.show()


def plot_avg_change_vs_farmer_count_by_area_class(
    data: pd.DataFrame,
    dabis_per_ha: float,
    redist_per_ha: tuple[float, float],
    bins: list[float],
    labels: list[str],
) -> None:
    data_with_subs = compute_capped_subsidies(data, dabis_per_ha, 0, redist_per_ha)
    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"] + data_with_subs["subs_redist"]
    )

    grouped_result = analyze_by_area_categories(
        data_with_subs, bins=bins, labels=labels
    )
    fig, ax = plt.subplots(figsize=(9, 5))

    # Oszlopdiagram
    bars = ax.bar(
        grouped_result.index,
        grouped_result["total_farmers"],
        alpha=0.6,
        color="steelblue",
    )
    ax.set_ylabel("Gazdálkodók száma (db)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
    )

    # Oszlop feliratok
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}".replace(",", " "),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="blue",
        )

    ax = ax.twinx()
    ax.plot(
        grouped_result.index,
        grouped_result["avg_change"],
        color="red",
        marker="o",
        linewidth=2,
    )
    ax.set_ylabel("Átlagos változás (%)", color="red")
    ax.tick_params(axis="y", labelcolor="red")

    y_values = grouped_result["avg_change"].values
    for i, (x, y) in enumerate(zip(grouped_result.index, y_values)):
        prev_y = y_values[i - 1] if i > 0 else y
        next_y = y_values[i + 1] if i < len(y_values) - 1 else y

        # Ha a szomszédos pont magasabban van, tegyük lejjebb a feliratot, különben feljebb
        if prev_y > y and next_y > y:
            offset_y = -15
        else:
            offset_y = 15

        ax.annotate(
            f"{y:.1f}%",
            xy=(x, y),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="red",
        )

    plt.title("Átlagos változás és gazdálkodók száma területkategóriánként")
    plt.tight_layout()
    plt.show()


def plot_total_capped_loss_by_area_class(
    data: pd.DataFrame,
    dabis_per_ha: float,
    redist_per_ha: tuple[float, float],
    bins: list[float],
    labels: list[str],
) -> None:
    data_with_subs = compute_capped_subsidies(data, dabis_per_ha, 0, redist_per_ha)
    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"] + data_with_subs["subs_redist"]
    )

    grouped_result = analyze_by_area_categories(
        data_with_subs, bins=bins, labels=labels
    )

    plt.figure(figsize=(8, 5))
    ax = grouped_result["total_capped_loss"].plot(
        kind="bar", color="salmon", title="A capping összege területkategóriánként"
    )
    plt.ylabel("Megvágott összeg (EUR)", fontsize=11)
    plt.xlabel("Területkategória (ha)", fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    grouped_result["avg_subs_per_ha_capped"].plot(
        kind="bar",
        color="seagreen",
        title="Átlagos fajlagos támogatás hektáronként capping után",
    )
    plt.ylabel("Fajlagos támogatás (EUR/ha)", fontsize=11)
    plt.xlabel("Területkategória (ha)", fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    ax = grouped_result["total_capped_loss"].plot(
        kind="bar",
        color="royalblue",
        title="Támogatás változása a jelenlegi és az új (cappingelt) rendszer között",
    )
    plt.ylabel("Eltérés összege (EUR)", fontsize=11)
    plt.xlabel("Területkategória (ha)", fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    ax = grouped_result["avg_change"].plot(
        kind="bar",
        color="darkorange",
        title="Relatív támogatásváltozás a jelenlegihez képest (%)",
    )
    plt.ylabel("Támogatásváltozás (%)", fontsize=11)
    plt.xlabel("Területkategória (ha)", fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_avg_subsidy_per_ha_current_by_area_class(
    data: pd.DataFrame,
    dabis_per_ha: float,
    redist_per_ha: tuple[float, float],
    bins: list[float],
    labels: list[str],
) -> None:
    data_with_subs = compute_capped_subsidies(data, dabis_per_ha, 0, redist_per_ha)
    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"] + data_with_subs["subs_redist"]
    )

    grouped_result = analyze_by_area_categories(
        data_with_subs, bins=bins, labels=labels
    )

    plt.figure(figsize=(8, 5))
    grouped_result["avg_subs_per_ha_cur"].plot(
        kind="bar",
        color="seagreen",
        title="Átlagos fajlagos támogatás hektáronként capping után",
    )
    plt.ylabel("Fajlagos támogatás (EUR/ha)", fontsize=11)
    plt.xlabel("Területkategória (ha)", fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def build_rate_area_matrix(
    data: pd.DataFrame,
    redist_per_ha: tuple[float, float],
    bins: list[float],
    labels: list[str],
    subsidy_rates: NDArray[np.float64],
) -> pd.DataFrame:
    results = []

    for rate in subsidy_rates:
        df_subs = compute_capped_subsidies(data, rate, 0, redist_per_ha)
        df_subs["subs_cur"] = df_subs["subs_biss"] + df_subs["subs_redist"]

        grouped = analyze_by_area_categories(df_subs, bins=bins, labels=labels)
        for cat in grouped.index:
            results.append(
                {
                    "subs_per_ha": rate,
                    "area_class": cat,
                    "avg_subs_per_ha": grouped.loc[cat, "avg_subs_per_ha_cur"],
                    "perc_diff": grouped.loc[cat, "avg_change"],
                }
            )

    result_df = pd.DataFrame(results)
    return result_df


def plot_subsidy_rate_sweep_by_area_class(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    sns.lineplot(
        data=data,
        x="subs_per_ha",
        y="avg_subs_per_ha",
        hue="area_class",
        marker="o",
        linewidth=2.2,
        markersize=6,
        ax=axs[0],
    )
    axs[0].set_title(
        "Support and Relative Change by Area Class across Support Levels",
        fontsize=14,
        weight="bold",
    )
    axs[0].set_ylabel("Átlagos fajlagos támogatás (EUR/ha)", fontsize=12)
    axs[0].grid(True, axis="y", linestyle="--", alpha=0.6)
    axs[0].legend(
        title="Területkategória (ha)",
        title_fontsize=11,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    axs[0].yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )

    sns.lineplot(
        data=data,
        x="subs_per_ha",
        y="perc_diff",
        hue="area_class",
        marker="o",
        linewidth=2.2,
        markersize=6,
        ax=axs[1],
        legend=False,
    )
    axs[1].set_title(
        "Relatív támogatásváltozás (%) különböző támogatási szinteken",
        fontsize=14,
        weight="bold",
    )
    axs[1].set_ylabel("Relatív támogatásváltozás (%)", fontsize=12)
    axs[1].set_xlabel("Hektáronkénti támogatási szint (EUR)", fontsize=12)
    axs[1].grid(True, axis="y", linestyle="--", alpha=0.6)
    axs[1].yaxis.set_major_formatter(PercentFormatter(decimals=0))
    axs[1].xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )

    plt.tight_layout()
    plt.show()


def plot_subsidy_rate_sweep_overview(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for area_class in data["area_class"].unique():
        subset = data[data["area_class"].eq(area_class)]
        ax.plot(
            subset["subs_per_ha"],
            subset["avg_subs_per_ha"],
            marker="o",
            label=area_class,
        )

    ax.set_title(
        "Átlagos fajlagos támogatás (EUR/ha) különböző támogatási szinteken",
        fontsize=13,
        weight="bold",
    )
    ax.set_xlabel("Hektáronkénti támogatási szint (EUR)", fontsize=11)
    ax.set_ylabel("Átlagos fajlagos támogatás (EUR/ha)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Területkategória", loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(10, 6))
    for area_class in data["area_class"].unique():
        subset = data[data["area_class"].eq(area_class)]
        ax.plot(
            subset["subs_per_ha"], subset["perc_diff"], marker="o", label=area_class
        )

    ax.set_title(
        "Relatív támogatásváltozás (%) különböző támogatási szinteken",
        fontsize=13,
        weight="bold",
    )
    ax.set_xlabel("Hektáronkénti támogatási szint (EUR)", fontsize=11)
    ax.set_ylabel("Relatív támogatásváltozás (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Területkategória", loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.tight_layout()

    pivot_table = data.pivot(
        index="area_class", columns="subs_per_ha", values="perc_diff"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    vlim = np.nanmax(np.abs(pivot_table.values))
    c = ax.imshow(
        pivot_table.values,
        cmap="coolwarm",
        vmin=-vlim,
        vmax=+vlim,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_xticklabels([f"{x:.0f} €" for x in pivot_table.columns], rotation=45)

    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_yticklabels(pivot_table.index)

    ax.set_title("Relatív támogatásváltozás (%) hőtérképe", fontsize=13, weight="bold")
    cbar = fig.colorbar(c, ax=ax, label="Támogatásváltozás (%)")
    cbar.formatter = PercentFormatter()
    cbar.update_ticks()
    plt.tight_layout()
    plt.show()


def plot_perc_diff_heatmap_by_rate_and_area_class(data: pd.DataFrame) -> None:
    vlim = np.nanmax(np.abs(data["perc_diff"].to_numpy(dtype="float")))
    matrix = data.pivot(index="area_class", columns="subs_per_ha", values="perc_diff")

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vlim, vmax=+vlim)

    xticks = np.arange(len(matrix.columns))
    xtick_labels = [
        f"{x:.0f} €" if i % 10 == 0 else "" for i, x in enumerate(matrix.columns)
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45)

    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_title("Relatív támogatásváltozás (%) hőtérképe", fontsize=13, weight="bold")
    fig.colorbar(im, ax=ax, label="Támogatásváltozás (%)")
    plt.tight_layout()
    plt.show()


def plot_yfs_distribution(data: pd.DataFrame, threshold: float) -> None:
    s = data["area_yfs_cur_eligible"]
    s = s[s > 0]

    count_below = (s <= threshold).sum()
    area_below = s[s <= threshold].sum()

    pct_count = 100 * count_below / len(s)
    pct_area = 100 * area_below / s.sum()

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=20, color="skyblue", edgecolor="black")
    plt.axvline(
        133, color="red", linestyle="--", linewidth=2, label=f"{threshold} ha threshold"
    )

    plt.text(
        142,
        plt.ylim()[1] * 0.8,
        f"Below {threshold} ha:\n{pct_count:.1f}% of farms\n{pct_area:.1f}% of area",
        color="red",
        fontsize=10,
    )

    plt.xlabel("YFS-eligible area (ha)")
    plt.ylabel("Number of farms")
    plt.title("Distribution of YFS-eligible area")
    plt.legend()
    plt.tight_layout()
    plt.show()


def add_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.010,  # offset above bar
            f"{h * 100:.1f}%",  # format as percent
            ha="center",
            va="bottom",
            fontsize=6,
        )


def plot_support_summary_by_area_class_01(
    data: pd.DataFrame,
    coupled_payments: dict[str, dict[str, float]],
    dabis_per_ha: float,
    redist_per_ha: tuple[float, float],
    bins: list[float],
    labels: list[str],
    allocation: float,
    cis_ratio: float = 1,
    flat_rate: float = 0,
) -> pd.DataFrame:
    data_with_subs = compute_capped_subsidies(data, dabis_per_ha, 90, redist_per_ha)

    cis_columns = [
        "subs_tk_cukorrepa",
        "subs_tk_szemes_feherjenoveny",
        "subs_tk_szalas_feherjenoveny",
        "subs_tk_extenziv_gyumolcs",
        "subs_tk_intenziv_gyumolcs",
        "subs_tk_ipari_olajnoveny",
        "subs_tk_ipari_zoldsegnoveny",
        "subs_tk_zoldsegnoveny",
        "subs_tk_rizs",
        "subs_tk_hizottbika",
        "subs_tk_anyatehen",
        "subs_tk_tejhasznu_tehen",
        "subs_tk_anyajuh",
    ]

    cis_new_columns = [
        "subs_new_tk_cukorrepa",
        "subs_new_tk_szemes_feherjenoveny",
        "subs_new_tk_szalas_feherjenoveny",
        "subs_new_tk_extenziv_gyumolcs",
        "subs_new_tk_intenziv_gyumolcs",
        "subs_new_tk_ipari_olajnoveny",
        "subs_new_tk_ipari_zoldsegnoveny",
        "subs_new_tk_zoldsegnoveny",
        "subs_new_tk_rizs",
        "subs_new_tk_hizottbika",
        "subs_new_tk_anyatehen",
        "subs_new_tk_tejhasznu_tehen",
        "subs_new_tk_anyajuh",
    ]

    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"]
        + data_with_subs["subs_redist"]
        + data_with_subs["subs_yfs"]
        + data_with_subs[cis_columns].fillna(0).sum(axis=1)
        + data_with_subs["subs_aop"].fillna(0)
        + data_with_subs["subs_vp_akg_2021"].fillna(0)
    )

    tk_feh = sum(
        coupled_payments[key]["budget"]
        for key in coupled_payments
        if key in ["tk_szalas_feherjenoveny", "tk_szemes_feherjenoveny"]
    )

    tk_nem_feh = sum(
        coupled_payments[key]["budget"]
        for key in coupled_payments
        if key not in ["tk_szalas_feherjenoveny", "tk_szemes_feherjenoveny"]
    )

    CIS_prot_ratio = (5 / 25 * 202_110_350) / tk_feh
    CIS_non_prot_ratio = (20 / 25 * 202_110_350) / tk_nem_feh

    for key in coupled_payments.keys():
        if key in ["tk_szalas_feherjenoveny", "tk_szemes_feherjenoveny"]:
            data_with_subs[f"subs_new_{key}"] = (
                CIS_prot_ratio * data_with_subs[f"subs_{key}"]
            )
        else:
            data_with_subs[f"subs_new_{key}"] = (
                CIS_non_prot_ratio * data_with_subs[f"subs_{key}"]
            )

    data_with_subs["subs_capped"] += (
        cis_ratio * data_with_subs[cis_new_columns].fillna(0).sum(axis=1)
        + flat_rate * data_with_subs["area_aop"]
    )

    grouped_result = analyze_by_area_categories(
        data_with_subs, bins=bins, labels=labels
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    bars1 = ax1.bar(
        grouped_result.index,
        grouped_result["total_farmers"],
        color="#0072B2",
        alpha=0.85,
    )
    ax1.set_ylabel("Gazdálkodók száma (db)", fontsize=12)
    ax1.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
    )
    add_bar_labels(ax1, bars1)

    bars2 = ax2.bar(
        grouped_result.index, grouped_result["total_area"], color="#56B4E9", alpha=0.85
    )
    ax2.set_ylabel("Összes terület (ha)", fontsize=12)
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
    )
    add_bar_labels(ax2, bars2)

    bars3 = ax3.bar(
        grouped_result.index,
        grouped_result["avg_change"],
        color=[
            "#009E73" if v >= 0 else "#B22222" for v in grouped_result["avg_change"]
        ],
        alpha=0.85,
    )
    ax3.set_ylabel("Fajlagos támogatás változása (%)", fontsize=12)
    ax3.axhline(0, color="gray", linestyle="--", linewidth=1)
    add_bar_labels(ax3, bars3, percent=True)

    ax3.set_xlabel("Méretkategória (ha)", fontsize=12)

    # fig.suptitle(
    #     f"Támogatások alakulása gazdálkodói méretkategóriák szerint\n{allocation / 1e6:.1f} millió EUR borítékkal DABIS borítékkal számolva".replace(
    #         ".", ","
    #     ),
    #     fontsize=14,
    #     fontweight="bold",
    #     x=0.5,
    #     y=0.95,
    # )

    fig.align_ylabels([ax1, ax2, ax3])
    plt.subplots_adjust(left=0.12, hspace=0.2)
    plt.savefig(f"output/abra_osszetett_{allocation / 1e6:.1f}.png", dpi=300)
    plt.show()
    return data_with_subs
