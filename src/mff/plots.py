from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
from numpy.typing import NDArray

from mff.new_cap import (
    analyze_by_area_categories,
    apply_reductions,
    cal_redist,
    calc_ratio_subs,
    calc_thresholds,
    compute_capped_subsidies,
    compute_current_support,
    find_cur_new_equal_roots,
    maximize_ratio,
)
from mff.utils import c_round

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
    ha_upper_l = 255_000 / policy.base_payment_per_ha
    x_values = np.linspace(
        0.01,
        ha_upper_l,
        100 * int(ha_upper_l / 10) + 1,
    )

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

    zero_crossings = find_cur_new_equal_roots(
        policy.base_payment_per_ha,
        policy.redist_params,
        policy.yfs_per_ha,
    )

    # Only plot equality points that are inside the plotted range
    zero_crossings = [x for x in zero_crossings if x <= ha_upper_l]

    thresholds = [20_000, 50_000, 75_000, 255_000]
    thresholds_ha = calc_thresholds(
        thresholds,
        policy.base_payment_per_ha,
        policy.redist_params,
        policy.yfs_per_ha,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        x_values,
        y_values_capping_per_ha,
        label=label1,
        color="#0072B2",
        linewidth=2.5,
    )

    ax.plot(
        x_values,
        y_values_current_per_ha,
        label=label2,
        color="#D55E00",
        linestyle="--",
        linewidth=2.5,
    )

    y_max = max(y_values_capping_per_ha)

    # Degression / capping thresholds
    for threshold, threshold_ha in zip(thresholds, thresholds_ha):
        ax.axvline(
            x=threshold_ha,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        )

        label = (
            f"Capping határ: 255 000 €\n({c_round(threshold_ha, 0):,.0f} ha)"
            if threshold == 255_000
            else f"{threshold:,.0f} €\n({c_round(threshold_ha, 0):,.0f} ha)"
        ).replace(",", " ")

        ax.text(
            threshold_ha + 5,
            y_max * 0.42,
            label,
            rotation=45,
            verticalalignment="bottom",
            color="red",
            fontsize=10,
        )

    # Current vs new support equality points
    for i, zero_cross_x in enumerate(zero_crossings):
        ax.axvline(
            x=zero_cross_x,
            color="#0072B2",
            linestyle="--",
            linewidth=1.5,
        )

        label = f"Egyenlőségi pont\n({zero_cross_x:,.0f} ha)"

        # Slight vertical staggering if there are multiple roots
        y_position = y_max * (0.85 - i * 0.12)

        ax.text(
            zero_cross_x + 5,
            y_position,
            label,
            rotation=45,
            verticalalignment="bottom",
            color="#0072B2",
            fontsize=10,
        )

    ax.set_xlabel(
        "Üzemméret (ha)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Támogatás hektáronként (€)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
    )

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

    ax.legend(
        loc="upper right",
        fontsize=11,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_total(
    policy: PolicyParams,
    label1: str,
    label2: str,
    title: str,
    output_path: str,
) -> None:
    ha_upper_l = 255_000 / policy.base_payment_per_ha
    x_values = np.linspace(
        0.01,
        ha_upper_l,
        100 * int(ha_upper_l / 10) + 1,
    )

    y_values_capping = [
        apply_reductions(
            policy.base_payment_per_ha * x
            + policy.yfs_per_ha * min(policy.yfs_cap_ha, x)
            + cal_redist(x, policy.redist_params)
        )
        for x in x_values
    ]

    y_values_current = [compute_current_support(x, policy.yfs_per_ha) for x in x_values]

    zero_crossings = find_cur_new_equal_roots(
        policy.base_payment_per_ha,
        policy.redist_params,
        policy.yfs_per_ha,
    )

    zero_crossings = [x for x in zero_crossings if x <= ha_upper_l]

    thresholds = [20_000, 50_000, 75_000, 255_000]
    thresholds_ha = calc_thresholds(
        thresholds,
        policy.base_payment_per_ha,
        policy.redist_params,
        policy.yfs_per_ha,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        x_values,
        y_values_capping,
        label=label1,
        color="#0072B2",
        linewidth=2.5,
    )

    ax.plot(
        x_values,
        y_values_current,
        label=label2,
        color="#D55E00",
        linestyle="--",
        linewidth=2.5,
    )

    y_max = max(y_values_capping)

    for threshold, threshold_ha in zip(thresholds, thresholds_ha):
        ax.axvline(
            x=threshold_ha,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )

        label = (
            f"Capping határ: 255 000 €\n({c_round(threshold_ha, 0):,.0f} ha)"
            if threshold == 255_000
            else f"{threshold:,.0f} €\n({c_round(threshold_ha, 0):,.0f} ha)"
        ).replace(",", " ")

        ax.text(
            threshold_ha + 5,
            y_max * 0.05,
            label,
            rotation=45,
            verticalalignment="bottom",
            color="red",
            fontsize=10,
        )

    for i, zero_cross_x in enumerate(zero_crossings):
        ax.axvline(
            x=zero_cross_x,
            color="#0072B2",
            linestyle="--",
            linewidth=1.5,
        )

        y_position = y_max * (0.85 - i * 0.12)

        ax.text(
            zero_cross_x + 5,
            y_position,
            f"Egyenlőségi pont\n({zero_cross_x:,.0f} ha)",
            rotation=45,
            verticalalignment="bottom",
            color="#0072B2",
            fontsize=10,
        )

    ax.set_xlabel(
        "Üzemméret (ha)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Támogatás (€)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        x_values[-1],
        y_values_capping[-1],
        f"{y_values_capping[-1]:,.0f} €".replace(",", " "),
        fontsize=10,
        color="#0072B2",
        va="bottom",
    )

    ax.text(
        x_values[-1],
        y_values_current[-1],
        f"{y_values_current[-1]:,.0f} €".replace(",", " "),
        fontsize=10,
        color="#D55E00",
        va="bottom",
    )

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_diff_dual_axis(
    policy: PolicyParams,
    title_name: str,
    scen_name: str,
    farm_areas: list[float] | np.ndarray,
    output_path: str,
    cum_mode: str = "farms",
) -> None:
    ha_upper_l = 255_000 / policy.base_payment_per_ha
    x_values = np.linspace(
        0.01,
        ha_upper_l,
        100 * int(ha_upper_l / 10) + 1,
    )

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

    # Maximum
    peak_x = maximize_ratio(
        50,
        policy.base_payment_per_ha,
        policy.yfs_per_ha,
        policy.redist_params,
    )

    peak_x_rounded = int(c_round(peak_x, 0))

    peak_y = 100 * calc_ratio_subs(
        peak_x,
        policy.base_payment_per_ha,
        policy.yfs_per_ha,
        policy.redist_params,
    )

    peak_y_rounded = c_round(peak_y, 2)

    # Equality points
    zero_crossings = find_cur_new_equal_roots(
        policy.base_payment_per_ha,
        policy.redist_params,
        policy.yfs_per_ha,
    )

    zero_crossings = [x for x in zero_crossings if x <= ha_upper_l]

    # Farm-size distribution
    fa = np.asarray(farm_areas, dtype=float)

    fa = fa[np.isfinite(fa) & (fa > 0) & (fa <= ha_upper_l)]

    if len(fa) == 0:
        raise ValueError("No valid farm areas in the plotted range.")

    fa_sorted = np.sort(fa)

    cum_farm_pct = np.arange(1, len(fa_sorted) + 1) / len(fa_sorted) * 100

    total_area = fa_sorted.sum()
    cum_area_pct = np.cumsum(fa_sorted) / total_area * 100

    # Textbox
    if len(zero_crossings) >= 2:
        lower, upper = zero_crossings[:2]

        mask_between = (fa_sorted >= lower) & (fa_sorted <= upper)

        pct_farms = mask_between.mean() * 100
        pct_area = fa_sorted[mask_between].sum() / total_area * 100

        textbox_text = (
            f"A két egyenlőségi pont ({lower:.0f}–{upper:.0f} ha) közé\n"
            f"az üzemek {pct_farms:.1f}%-a és "
            f"a terület {pct_area:.1f}%-a esik."
        )

    elif len(zero_crossings) == 1:
        zero_cross_x = zero_crossings[0]

        mask_below = fa_sorted <= zero_cross_x

        pct_farms = mask_below.mean() * 100
        pct_area = fa_sorted[mask_below].sum() / total_area * 100

        textbox_text = (
            f"{zero_cross_x:.0f} ha alatt az üzemek "
            f"{pct_farms:.1f}%-a és a terület "
            f"{pct_area:.1f}%-a található."
        )

    else:
        textbox_text = "Nincs egyenlőségi pont a vizsgált tartományban."

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_axisbelow(True)

    ax1.plot(
        x_values,
        diff_pct,
        color="#009E73",
        linewidth=2.2,
        label=scen_name,
    )

    ax1.axhline(
        0,
        linestyle="--",
        color="black",
        linewidth=1,
    )

    ax1.set_xlabel(
        "Üzemméret (ha)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel(
        "Eltérés (%)",
        fontsize=12,
        fontweight="bold",
    )

    ax1.yaxis.set_major_locator(MultipleLocator(10))

    step = 10
    values_for_ylim = [*diff_pct, peak_y]

    data_min = min(values_for_ylim)
    data_max = max(values_for_ylim)

    ymin = step * np.floor(data_min / step)
    ymax = step * np.ceil(1.05 * data_max / step)

    ax1.set_ylim(ymin, ymax)

    # ------------------------------------------------------------
    # Secondary axis
    # Must be created BEFORE annotations that use ax2/y2
    # ------------------------------------------------------------
    ax2 = ax1.twinx()

    if cum_mode == "area":
        y2 = cum_area_pct
        y2_label = "Kumulált területarány (%)"

    elif cum_mode == "farms":
        y2 = cum_farm_pct
        y2_label = "Kumulált üzemszámarány (%)"

    else:
        raise ValueError("cum_mode must be 'farms' or 'area'.")

    ax2.plot(
        fa_sorted,
        y2,
        color="0.3",
        linewidth=2,
        label=y2_label,
    )

    ax2.set_ylabel(
        y2_label,
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(PercentFormatter())

    # ------------------------------------------------------------
    # Vertical lines
    # ------------------------------------------------------------
    ax1.axvline(
        peak_x,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Maximum",
    )

    for i, zero_cross_x in enumerate(zero_crossings):
        ax1.axvline(
            zero_cross_x,
            color="#56B4E9",
            linestyle="--",
            linewidth=1.5,
            label="Egyenlőségi pont" if i == 0 else None,
        )

    # ------------------------------------------------------------
    # Maximum annotation
    # ------------------------------------------------------------
    cum_y_at_peak = np.interp(
        peak_x,
        fa_sorted,
        y2,
    )

    peak_display_y = ax1.transData.transform((peak_x, peak_y))[1]

    cum_display_y = ax2.transData.transform((peak_x, cum_y_at_peak))[1]

    distance_px = abs(peak_display_y - cum_display_y)

    peak_y_offset = -42 if distance_px < 60 else -28

    if peak_x < ha_upper_l * 0.65:
        peak_x_offset = 45
        peak_ha = "left"
    else:
        peak_x_offset = -45
        peak_ha = "right"

    ax1.annotate(
        f"{peak_y_rounded}% ({peak_x_rounded} ha)",
        xy=(peak_x, peak_y),
        xytext=(peak_x_offset, peak_y_offset),
        textcoords="offset points",
        ha=peak_ha,
        va="top",
        arrowprops={
            "arrowstyle": "->",
            "color": "red",
            "linewidth": 1.2,
        },
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
            "pad": 0.25,
        },
        fontsize=11,
        fontweight="bold",
        color="red",
        zorder=20,
    )

    # ------------------------------------------------------------
    # Equality-point annotations
    # ------------------------------------------------------------
    zero_offsets = [
        (-35, 18),
        (35, 18),
    ]

    for i, zero_cross_x in enumerate(zero_crossings):
        if i < len(zero_offsets):
            offset = zero_offsets[i]
        else:
            offset = (35, 18 + 16 * i)

        ax1.annotate(
            f"{zero_cross_x:.0f} ha",
            xy=(zero_cross_x, 0),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom",
            arrowprops={
                "arrowstyle": "->",
                "color": "#56B4E9",
                "linewidth": 1.1,
            },
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 0.2,
            },
            fontsize=11,
            fontweight="bold",
            color="#56B4E9",
            zorder=20,
        )

    # ------------------------------------------------------------
    # Title / formatting
    # ------------------------------------------------------------
    ax1.set_title(
        title_name,
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)

    # ------------------------------------------------------------
    # Textbox placement
    # ------------------------------------------------------------
    x_ref = x_values[int(len(x_values) * 0.8)]

    idx2 = (
        np.searchsorted(
            fa_sorted,
            x_ref,
            side="right",
        )
        - 1
    )

    idx2 = np.clip(
        idx2,
        0,
        len(fa_sorted) - 1,
    )

    grey_y = y2[idx2]

    grey_disp = ax2.transData.transform((x_ref, grey_y))

    grey_axes = ax1.transAxes.inverted().transform(grey_disp)

    y_grey_frac = grey_axes[1]

    candidates = [0.9, 0.8, 0.7, 0.6]

    text_y_pos = next(
        (candidate for candidate in candidates if abs(candidate - y_grey_frac) > 0.12),
        0.85,
    )

    ax1.text(
        0.98,
        text_y_pos,
        textbox_text,
        transform=ax1.transAxes,
        fontsize=11,
        ha="right",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.8,
            "boxstyle": "round,pad=0.4",
            "alpha": 1.0,
        },
        zorder=10,
        clip_on=False,
    )

    # ------------------------------------------------------------
    # Combined legend
    # ------------------------------------------------------------
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        fontsize=11,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

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

    _, ax = plt.subplots(figsize=(12, 6))

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
    ha_upper_l = 255_000 / policy.base_payment_per_ha
    x_values = np.linspace(
        0.01,
        ha_upper_l,
        100 * int(ha_upper_l / 10) + 1,
    )

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
        50,
        policy.base_payment_per_ha,
        policy.yfs_per_ha,
        policy.redist_params,
    )
    peak_x_rounded = int(c_round(peak_x, 0))
    peak_y = 100 * calc_ratio_subs(
        peak_x,
        policy.base_payment_per_ha,
        policy.yfs_per_ha,
        policy.redist_params,
    )
    peak_y_rounded = c_round(peak_y, 2)

    zero_crossings = find_cur_new_equal_roots(
        policy.base_payment_per_ha,
        policy.redist_params,
        policy.yfs_per_ha,
    )

    zero_crossings = [x for x in zero_crossings if x <= ha_upper_l]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        x_values,
        percent_diff,
        label=scen_name,
        color="#009E73",
        linewidth=2.5,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(
        peak_x,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Maximum",
    )

    for i, zero_cross_x in enumerate(zero_crossings):
        ax.axvline(
            zero_cross_x,
            color="#56B4E9",
            linestyle="--",
            linewidth=1.5,
            label="Egyenlőségi pont" if i == 0 else None,
        )

    # Maximum
    ax.annotate(
        f"{peak_y_rounded}% ({peak_x_rounded} ha)",
        xy=(peak_x, peak_y),
        xytext=(35, -30),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "->",
            "color": "red",
        },
        fontsize=11,
        fontweight="bold",
        color="red",
    )

    # Egyenlőségi pontok
    offsets = [
        (-45, 18),
        (15, 18),
    ]

    for i, zero_cross_x in enumerate(zero_crossings):
        offset = offsets[i] if i < len(offsets) else (15, 18 + 18 * i)

        ax.annotate(
            f"{zero_cross_x:.0f} ha",
            xy=(zero_cross_x, 0),
            xytext=offset,
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "->",
                "color": "#56B4E9",
            },
            fontsize=11,
            fontweight="bold",
            color="#56B4E9",
            ha="center",
        )

    ax.set_xlabel("Üzemméret (ha)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Eltérés (%)", fontsize=12, fontweight="bold")
    ax.set_title(title_name, fontsize=14, fontweight="bold")

    ax.legend(loc="best", fontsize=11, frameon=True)

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_allocation_with_fixed_rate(
    payment_rate: float,
    subs_per_ha_results: list[float],
    alloc_results: list[float],
    values: list[float],
) -> None:
    x = np.asarray(subs_per_ha_results, dtype=float)
    allocation = np.asarray(alloc_results, dtype=float)
    rates = np.asarray(values, dtype=float)

    if not (len(x) == len(allocation) == len(rates)):
        raise ValueError("All input arrays must have the same length.")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # ------------------------------------------------------------
    # Curves
    # ------------------------------------------------------------
    line1 = ax1.plot(
        x,
        allocation,
        linewidth=2.5,
        label="Szükséges keret",
    )[0]

    line2 = ax2.plot(
        x,
        rates,
        linewidth=2.5,
        linestyle="--",
        label="Kifizetési arány",
    )[0]

    fixed_line = ax2.axhline(
        payment_rate,
        linestyle=":",
        linewidth=2,
        label=f"Rögzített kifizetési arány ({payment_rate:.2f} EUR/ha)",
    )

    # ------------------------------------------------------------
    # Find intersection by linear interpolation
    # ------------------------------------------------------------
    diff = rates - payment_rate
    crossing_idx = np.flatnonzero(diff[:-1] * diff[1:] <= 0)

    if len(crossing_idx) == 0:
        raise ValueError(
            f"The fixed payment rate ({payment_rate}) does not intersect "
            "the calculated payment-rate curve."
        )

    i = crossing_idx[0]

    x1, x2 = x[i], x[i + 1]
    r1, r2 = rates[i], rates[i + 1]

    if r1 == r2:
        intersection_x = x1
    else:
        intersection_x = x1 + ((payment_rate - r1) / (r2 - r1) * (x2 - x1))

    intersection_allocation = np.interp(
        intersection_x,
        x,
        allocation,
    )

    # ------------------------------------------------------------
    # Mark intersection
    # ------------------------------------------------------------
    ax1.axvline(
        intersection_x,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )

    ax1.scatter(
        intersection_x,
        intersection_allocation,
        s=45,
        zorder=5,
    )

    ax2.scatter(
        intersection_x,
        payment_rate,
        s=45,
        zorder=5,
    )

    # ------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------
    text = (
        f"{'Támogatás hektáronként:':<27}"
        f"{intersection_x:>8.2f} EUR/ha\n"
        f"{'Szükséges keret:':<27}"
        f"{intersection_allocation:>8.2f} millió EUR\n"
        f"{'Kifizetési arány:':<27}"
        f"{payment_rate:>8.2f} EUR/ha"
    )

    ax1.annotate(
        text,
        xy=(intersection_x, intersection_allocation),
        xytext=(25, 25),
        textcoords="offset points",
        fontsize=10,
        family="monospace",
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.95,
        },
        arrowprops={
            "arrowstyle": "->",
        },
    )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
    ax1.set_xlabel(
        "Támogatás hektáronként (EUR/ha)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel(
        "Szükséges keret (millió EUR)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel(
        "Kifizetési arány (EUR/ha)",
        fontsize=12,
        fontweight="bold",
    )

    ax1.set_title(
        "A támogatási egységösszeg és a szükséges keret kapcsolata",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax1.grid(
        axis="both",
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )

    # Combined legend
    lines = [line1, line2, fixed_line]
    ax1.legend(
        lines,
        [line.get_label() for line in lines],
        loc="best",
        frameon=True,
    )

    fig.tight_layout()
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
    yfs_per_ha: float,
    cal_type: str,
    bins: list[float],
    labels: list[str],
) -> None:
    data_with_subs = compute_capped_subsidies(
        data, dabis_per_ha, yfs_per_ha, redist_per_ha
    )
    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"]
        + data_with_subs["subs_redist"]
        + data_with_subs["subs_yfs"]
    )

    if cal_type == "csak fiatal gazda":
        data_with_subs = data_with_subs[data_with_subs["subs_yfs"] > 0]
    if cal_type == "nem fiatal gazda":
        data_with_subs = data_with_subs[data_with_subs["subs_yfs"] == 0]

    plot_data = analyze_by_area_categories(
        data_with_subs, bins=bins, labels=labels
    ).loc[:, ["avg_subs_per_ha_cur", "avg_subs_per_ha_capped"]]

    plot_data = plot_data[
        plot_data["avg_subs_per_ha_cur"].notna()
        & plot_data["avg_subs_per_ha_capped"].notna()
    ]

    x = np.arange(len(plot_data))  # number of categories
    width = 0.35  # width of bars

    plt.figure(figsize=(8, 5))

    color_current = "#0072B2"
    color_capped = "#009E73"

    bars1 = plt.bar(
        x - width / 2,
        plot_data["avg_subs_per_ha_cur"],
        width,
        label="Jelenlegi (alaptámogatás + redisztribúció + fiatal gazda támogatás)",
        color=color_current,
    )

    bars2 = plt.bar(
        x + width / 2,
        plot_data["avg_subs_per_ha_capped"],
        width,
        label="Új (degresszív alaptámogatás)",
        color=color_capped,
    )

    plt.ylabel("Fajlagos támogatás (EUR/ha)", fontsize=11)
    plt.xlabel("Méretkategória", fontsize=11)
    plt.suptitle(
        f"Átlagos fajlagos támogatás gazdálkodói méretkategóriák szerint\n({cal_type})",
        fontweight="bold",
    )
    plt.xticks(x, plot_data.index.astype(str).tolist(), rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(prop={"size": 9}, frameon=True)

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
    plt.savefig(f"output/abra_fajlagos_tam_{cal_type}.png", dpi=300)
    plt.show()


def plot_nfarmer_by_area_class(
    data: pd.DataFrame,
    dabis_per_ha: float,
    redist_per_ha: tuple[float, float],
    yfs_per_ha: float,
    cal_type: str,
    bins: list[float],
    labels: list[str],
) -> None:
    data_with_subs = compute_capped_subsidies(
        data, dabis_per_ha, yfs_per_ha, redist_per_ha
    )
    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"]
        + data_with_subs["subs_redist"]
        + data_with_subs["subs_yfs"]
    )

    if cal_type == "csak fiatal gazda":
        data_with_subs = data_with_subs[data_with_subs["subs_yfs"] > 0]
    if cal_type == "nem fiatal gazda":
        data_with_subs = data_with_subs[data_with_subs["subs_yfs"] == 0]

    grouped_result = analyze_by_area_categories(
        data_with_subs, bins=bins, labels=labels
    )

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

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

    fig.suptitle(
        f"Üzemméret alakulása gazdálkodói méretkategóriák szerint\n({cal_type})".replace(
            ".", ","
        ),
        # fontsize=12,
        fontweight="bold",
        # x=0.5,
        # y=0.95,
    )

    ax1.grid(False)

    plt.savefig(f"output/uzemszam_{cal_type}.png", dpi=400)
    plt.show()


def calc_support_summary_by_area_class(
    data: pd.DataFrame,
    dabis_per_ha: float,
    redist_per_ha: tuple[float, float],
    yfs_per_ha: float,
    cal_type: str,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    data_with_subs = compute_capped_subsidies(
        data, dabis_per_ha, yfs_per_ha, redist_per_ha
    )
    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"]
        + data_with_subs["subs_redist"]
        + data_with_subs["subs_yfs"]
    )

    if cal_type == "csak fiatal gazda":
        data_with_subs = data_with_subs[data_with_subs["subs_yfs"] > 0]
    if cal_type == "nem fiatal gazda":
        data_with_subs = data_with_subs[data_with_subs["subs_yfs"] == 0]

    return analyze_by_area_categories(data_with_subs, bins=bins, labels=labels)


def plot_support_summary_by_area_class(
    data: pd.DataFrame,
    dabis_per_ha: float,
    redist_per_ha: tuple[float, float],
    yfs_per_ha: float,
    cal_type: str,
    bins: list[float],
    labels: list[str],
) -> None:
    data_with_subs = compute_capped_subsidies(
        data, dabis_per_ha, yfs_per_ha, redist_per_ha
    )
    data_with_subs["subs_cur"] = (
        data_with_subs["subs_biss"]
        + data_with_subs["subs_redist"]
        + data_with_subs["subs_yfs"]
    )

    if cal_type == "csak fiatal gazda":
        data_with_subs = data_with_subs[data_with_subs["subs_yfs"] > 0]
    if cal_type == "nem fiatal gazda":
        data_with_subs = data_with_subs[data_with_subs["subs_yfs"] == 0]

    grouped_result = analyze_by_area_categories(
        data_with_subs, bins=bins, labels=labels
    )

    fig, (ax2, ax3) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # bars1 = ax1.bar(
    #     grouped_result.index,
    #     grouped_result["total_farmers"],
    #     color="#0072B2",
    #     alpha=0.85,
    # )
    # ax1.set_ylabel("Gazdálkodók száma (db)", fontsize=12)
    # ax1.yaxis.set_major_formatter(
    #     FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
    # )
    # add_bar_labels(ax1, bars1)

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
        f"Támogatások alakulása gazdálkodói méretkategóriák szerint\n({cal_type})".replace(
            ".", ","
        ),
        # fontsize=12,
        fontweight="bold",
        # x=0.5,
        # y=0.95,
    )

    ax2.grid(False)
    ax3.grid(False)

    fig.align_ylabels([ax2, ax3])
    plt.subplots_adjust(left=0.12, hspace=0.2)
    plt.savefig(f"output/abra_osszetett_{cal_type}.png", dpi=400)
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
    _, ax = plt.subplots(figsize=(9, 5))

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

    _, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

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

    for key in coupled_payments:
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

    fig.align_ylabels([ax1, ax2, ax3])
    plt.subplots_adjust(left=0.12, hspace=0.2)
    plt.savefig(f"output/abra_osszetett_{allocation / 1e6:.1f}.png", dpi=300)
    plt.show()
    return data_with_subs


def plot_cis_claimant_distribution_by_area_class(
    data: pd.DataFrame,
    param: str,
) -> None:
    dict_param = {
        "count_tk_tejhasznu_tehen": "Termeléshez kötött támogatás - Tejhasznú igénylők",
        "count_tk_anyajuh": "Termeléshez kötött támogatás - anyajuh igénylők",
        "count_tk_hizottbika": "Termeléshez kötött támogatás - hízottbika igénylők",
        "count_tk_anyatehen": "Termeléshez kötött támogatás - anyatehén igénylők",
        "area_tk_cukorrepa": "Termeléshez kötött támogatás - cukorrépa igénylők",
        "area_tk_ipari_zoldsegnoveny": "Termeléshez kötött támogatás - ipari zöldségnövény igénylők",
        "area_tk_zoldsegnoveny": "Termeléshez kötött támogatás - zöldségnövény igénylők",
        "area_tk_extenziv_gyumolcs": "Termeléshez kötött támogatás - extenzív gyümölcs igénylők",
        "area_tk_intenziv_gyumolcs": "Termeléshez kötött támogatás - intenzív gyümölcs igénylők",
        "area_tk_szemes_feherjenoveny": "Termeléshez kötött támogatás - szemes fehérjenövény igénylők",
    }
    mask = data[param] > 0
    share = data[mask].groupby("area_class", observed=False).size() / mask.sum() * 100

    plt.style.use("seaborn-v0_8-whitegrid")

    _, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.Blues(np.linspace(0.2, 0.9, len(share)))
    bars = []
    for i, (cat, v) in enumerate(zip(share.index, share.values)):
        if cat == "Nincs mezőgazdasági területe":
            bar = ax.bar(
                cat,
                v,
                color="lightgrey",
                edgecolor="black",
                hatch="//",  # csíkozás
            )
        else:
            bar = ax.bar(
                cat,
                v,
                color=colors[i],
                edgecolor="black",
            )
        bars.append(bar[0])

    ax.set_ylabel("Üzemszám arány (%)")
    ax.set_title(
        f"{dict_param[param]} százalékos megoszlása\nbirtokméret kategóriák szerint"
    )
    plt.xticks(rotation=45)

    for bar, v in zip(bars, share.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


def plot_cis_quantity_distribution_by_area_class(
    data: pd.DataFrame,
    param: str,
) -> None:
    dict_param = {
        "count_tk_tejhasznu_tehen": "Termeléshez kötött támogatás - Tejhasznú tehénállomány",
        "count_tk_anyajuh": "Termeléshez kötött támogatás - anyajuh állomány",
        "count_tk_hizottbika": "Termeléshez kötött támogatás - hízottbika állomány",
        "count_tk_anyatehen": "Termeléshez kötött támogatás - anyatehén állomány",
        "area_tk_cukorrepa": "Termeléshez kötött támogatás - cukorrépa terület",
        "area_tk_ipari_zoldsegnoveny": "Termeléshez kötött támogatás - ipari zöldségnövény terület",
        "area_tk_zoldsegnoveny": "Termeléshez kötött támogatás - zöldségnövény terület",
        "area_tk_extenziv_gyumolcs": "Termeléshez kötött támogatás - extenzív gyümölcsterület",
        "area_tk_intenziv_gyumolcs": "Termeléshez kötött támogatás - intenzív gyümölcsterület",
        "area_tk_szemes_feherjenoveny": "Termeléshez kötött támogatás - szemes fehérjenövény terület",
    }

    total = data[param].sum()
    share = data.groupby("area_class", observed=False)[param].sum() / total * 100

    plt.style.use("seaborn-v0_8-whitegrid")

    _, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.Blues(np.linspace(0.2, 0.9, len(share)))
    bars = []
    for i, (cat, v) in enumerate(zip(share.index, share.values)):
        if cat == "Nincs mezőgazdasági területe":
            bar = ax.bar(
                cat,
                v,
                color="lightgrey",
                edgecolor="black",
                hatch="//",  # csíkozás
            )
        else:
            bar = ax.bar(
                cat,
                v,
                color=colors[i],
                edgecolor="black",
            )
        bars.append(bar[0])

    ax.set_ylabel("Arány (%)")
    ax.set_title(
        f"{dict_param[param]} százalékos megoszlása\nbirtokméret kategóriák szerint"
    )
    plt.xticks(rotation=45)

    for bar, v in zip(bars, share.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()
