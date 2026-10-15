import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mff.dabis_engine import DabisPolicy, DegressionBand, gross_at_payment_cap
from mff.new_cap import (
    compute_capped_subsidies,
    compute_current_support,
    compute_dabis_support_summary,
    find_gross_flat_rate,
    generate_labels_from_bins,
    read_extended_base_data,
    summarize_reduction_breakdown,
)
from mff.plots import (
    PolicyParams,
    calc_support_summary_by_area_class,
    plot_diff_pct,
    plot_per_ha,
    plot_reduction,
    plot_total,
)


def color_row(row):
    value = row["Változás (%)"]

    if pd.isna(value):
        return [""] * len(row)

    if value > 0:
        color = "background-color: rgba(0, 128, 0, 0.06);"
    elif value < 0:
        color = "background-color: rgba(220, 0, 0, 0.05);"
    else:
        color = ""

    return [color] * len(row)


def color_change(value):
    if pd.isna(value):
        return ""

    if value > 0:
        return "background-color: rgba(0, 128, 0, 0.14);"
    if value < 0:
        return "background-color: rgba(220, 0, 0, 0.12);"

    return ""


st.set_page_config(page_title="KAP hatáselemző", page_icon="🌾", layout="wide")
st.markdown(
    """
<style>
    .stApp {background:#f5f8f8;color:#19313a}
    [data-testid="stSidebar"] {background:#102934}
    [data-testid="stSidebar"] * {color:#e3f2ee}
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {color:#19313a}
    .block-container {padding-top:1.6rem;max-width:1450px}
    [data-testid="stMetric"] {background:white;border:1px solid #dfe9e8;border-radius:12px;padding:16px 19px}
    .eyebrow {color:#168668;font-size:.75rem;font-weight:800;letter-spacing:.13em}
    .title {font-size:clamp(2rem,3vw,3rem);line-height:1.15;letter-spacing:-.04em;font-weight:800;margin:.35rem 0}
    .subtitle {color:#698388;font-size:.9rem;margin:0 0 1.2rem}
</style>
""",
    unsafe_allow_html=True,
)


def number(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


@st.cache_data(show_spinner="Üzemsoros állomány betöltése…")
def load_local_data(year: int) -> pd.DataFrame:
    # Pontosan ugyanaz az adatbetöltő, mint a notebook első futtatott cellájában.
    return read_extended_base_data(year)


@st.cache_data(show_spinner="Ábra készítése a plots.py függvényeivel…")
def notebook_plot(
    kind: str,
    dabis: float,
    yfs: float,
    redist: tuple[float, float],
    is_young: bool,
    band_1: float,
    band_2: float,
    band_3: float,
    rate_20_50: float,
    rate_50_75: float,
    rate_75_plus: float,
    payment_cap: float | None,
) -> bytes:
    """A plots.py függvényének mentett PNG-jét adja át a Streamlitnek."""
    dabis_policy = DabisPolicy(
        bands=(
            DegressionBand(0, band_1, 0.0, "band_1"),
            DegressionBand(band_1, band_2, rate_20_50, "band_2"),
            DegressionBand(band_2, band_3, rate_50_75, "band_3"),
            DegressionBand(band_3, None, rate_75_plus, "band_4"),
        ),
        payment_cap=payment_cap,
    )
    policy = PolicyParams(
        base_payment_per_ha=dabis,
        yfs_per_ha=yfs if is_young else 0,
        redist_params=redist,
        dabis_policy=dabis_policy,
    )
    group = "fiatal gazdálkodó" if is_young else "nem fiatal gazdálkodó"
    old_label = "BISS+CRISS+CIS-YF" if is_young else "BISS+CRISS"
    fd, output_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.close("all")
    try:
        if kind == "Fajlagos támogatás":
            plot_per_ha(
                policy,
                label1="DABIS",
                label2=old_label,
                title=f"Hektáronkénti támogatás ({group})",
                output_path=output_path,
            )
        elif kind == "Összes támogatás":
            plot_total(
                policy,
                label1="DABIS",
                label2=old_label,
                title=f"Teljes támogatás ({group})",
                output_path=output_path,
            )
        elif kind == "Százalékos eltérés":
            plot_diff_pct(
                policy,
                f"DABIS és jelenlegi támogatás eltérése ({group})",
                "Eltérés (%)",
                output_path=output_path,
            )
        elif kind == "Degresszió":
            plot_reduction(
                policy,
                f"Degresszió és plafon hatása ({group})",
                "Megmaradó támogatás (%)",
                output_path=output_path,
            )
        else:
            raise ValueError(f"Ismeretlen ábra: {kind}")
        return Path(output_path).read_bytes()
    finally:
        plt.close("all")
        Path(output_path).unlink(missing_ok=True)


if "pending_dabis" in st.session_state:
    st.session_state["dabis"] = st.session_state.pop("pending_dabis")

with st.sidebar:
    st.header("Forgatókönyv")

    base_year = st.selectbox(
        "Bázisév",
        options=[2024, 2025, 2026],
        index=0,
    )
    st.caption("Kiinduló értékek: 90 €/ha általános DABIS + 50 €/ha fiatal gazda")

    dabis = st.number_input(
        "DABIS általános (€/ha)",
        min_value=90.0,
        max_value=240.0,
        value=90.0,
        step=5.0,
        key="dabis",
    )
    yfs = st.number_input(
        "DABIS · fiatal gazda (€/ha)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=5.0,
    )

    with st.expander("Degresszió és capping", expanded=True):
        st.caption("A sávhatárok bruttó DABIS összegek, az elvonás marginális.")

        band_1 = st.number_input(
            "1. sávhatár (€)",
            min_value=1_000.0,
            max_value=500_000.0,
            value=20_000.0,
            step=5_000.0,
        )
        band_2 = st.number_input(
            "2. sávhatár (€)",
            min_value=1_000.0,
            max_value=500_000.0,
            value=50_000.0,
            step=5_000.0,
        )
        band_3 = st.number_input(
            "3. sávhatár (€)",
            min_value=1_000.0,
            max_value=1_000_000.0,
            value=75_000.0,
            step=5_000.0,
        )

        rate_20_50_pct = st.slider("2. sáv elvonása", 0, 100, 25, 5, format="%d%%")
        rate_50_75_pct = st.slider("3. sáv elvonása", 0, 100, 50, 5, format="%d%%")
        rate_75_plus_pct = st.slider("4. sáv elvonása", 0, 100, 75, 5, format="%d%%")

        cap_enabled = st.toggle("Kifizetési plafon", value=True)
        cap_value = st.number_input(
            "Plafon (€ / üzem)",
            min_value=1_000.0,
            max_value=1_000_000.0,
            value=100_000.0,
            step=5_000.0,
            disabled=not cap_enabled,
        )

        if not 0 < band_1 < band_2 < band_3:
            st.error("A sávhatároknak növekvő sorrendben kell lenniük.")
            st.stop()

        rate_20_50 = rate_20_50_pct / 100
        rate_50_75 = rate_50_75_pct / 100
        rate_75_plus = rate_75_plus_pct / 100
        payment_cap = cap_value if cap_enabled else None

        dabis_policy = DabisPolicy(
            bands=(
                DegressionBand(0, band_1, 0.0, "band_1"),
                DegressionBand(band_1, band_2, rate_20_50, "band_2"),
                DegressionBand(band_2, band_3, rate_50_75, "band_3"),
                DegressionBand(band_3, None, rate_75_plus, "band_4"),
            ),
            payment_cap=payment_cap,
        )

        cap_trigger = gross_at_payment_cap(dabis_policy)
        if payment_cap is None:
            st.caption("A capping ki van kapcsolva.")
        elif cap_trigger is None:
            st.caption(
                "A megadott degressziós ráták mellett ez a plafon nem válik kötővé."
            )
        else:
            st.caption(
                f"A plafon kb. {number(cap_trigger)} € bruttó DABIS-nál kezd kötni."
            )

    st.divider()
    st.caption(
        f"A {base_year}-es BISS + CRISS + fiatal gazda támogatáshoz hasonlít. "
        "Az AÖP és más támogatási elemek itt nem szerepelnek."
    )


redist = (0.0, 0.0)
bins = [0, 10, 150, 300, 1200, float("inf")]
labels = generate_labels_from_bins(bins)
required = {
    "area_biss_criss",
    "area_yfs_cur_eligible",
    "subs_biss",
    "subs_redist",
    "subs_yfs",
}
policy_key = (
    yfs,
    redist,
    band_1,
    band_2,
    band_3,
    rate_20_50,
    rate_50_75,
    rate_75_plus,
    payment_cap,
)

st.markdown(
    f'<div class="eyebrow">{base_year}-ES BÁZIS · ÚJ KAP FORGATÓKÖNYV</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="title">KAP Hatáselemző</div>', unsafe_allow_html=True)

farm_tab, data_tab = st.tabs(
    [
        "Üzemméret szerinti hatás",
        "Üzemsoros eredmények",
    ]
)

with farm_tab:
    a, b = st.columns(2)
    with a:
        area = st.number_input(
            "Üzemméret (ha)", min_value=0.1, max_value=10000.0, value=100.0, step=1.0
        )
    with b:
        farmer_type = st.segmented_control(
            "Gazdálkodói státusz",
            ["Nem fiatal", "Fiatal"],
            default="Nem fiatal",
            selection_mode="single",
            width="stretch",
        )

        is_young = farmer_type == "Fiatal"

    # Az új támogatást a notebook compute_capped_subsidies függvénye számolja.
    farm = pd.DataFrame(
        {
            "area_biss_criss": [area],
            "area_yfs_cur_eligible": [min(area, 300) if is_young else 0],
        }
    )
    farm_new = compute_capped_subsidies(farm, dabis, yfs, redist, dabis_policy).iloc[0]
    farm_current = compute_current_support(area, is_young)
    m1, m2, m3 = st.columns(3)
    m1.metric("Jelenlegi támogatás", f"{number(farm_current)} €")
    m2.metric("Új támogatás", f"{number(farm_new.subs_capped)} €")
    m3.metric(
        "Változás",
        f"{number(farm_new.subs_capped - farm_current)} €",
        delta=f"{number(100 * (farm_new.subs_capped / farm_current - 1), 1)} %"
        if farm_current
        else None,
    )

    chart_kind = st.radio(
        "Ábra",
        ["Fajlagos támogatás", "Összes támogatás", "Százalékos eltérés", "Degresszió"],
        horizontal=True,
    )
    if is_young and yfs == 0 and chart_kind != "Degresszió":
        st.warning(
            "A plots.py az új fiatal gazda ráta előjeléből következtet a jelenlegi fiatal státuszra. Nulla új ráta mellett ez az ábra félrevezető lenne; az üzemszintű kártyák továbbra is helyesek."
        )
    else:
        try:
            st.image(
                notebook_plot(
                    chart_kind,
                    dabis,
                    yfs,
                    redist,
                    is_young,
                    band_1,
                    band_2,
                    band_3,
                    rate_20_50,
                    rate_50_75,
                    rate_75_plus,
                    payment_cap,
                ),
                width="stretch",
            )
        except ValueError as exc:
            st.warning(f"Az ábra ezekkel a paraméterekkel nem készíthető el: {exc}")

    with st.expander("Jellemző üzemméretek"):
        areas = np.array([5, 10, 50, 100, 150, 300, 500, 1200, 1500])

        examples = pd.DataFrame(
            {
                "area_biss_criss": areas,
                "area_yfs_cur_eligible": np.minimum(areas, 300) if is_young else 0,
            }
        )

        examples_new = compute_capped_subsidies(
            examples,
            dabis,
            yfs,
            redist,
            dabis_policy,
        )

        overview = pd.DataFrame(
            {
                "Üzemméret (ha)": areas,
                "Jelenlegi (€)": [
                    compute_current_support(area, is_young) for area in areas
                ],
                "Új (€)": examples_new["subs_capped"].to_numpy(),
            }
        )

        overview["Változás (€)"] = overview["Új (€)"] - overview["Jelenlegi (€)"]

        overview["Változás (%)"] = (
            100 * overview["Változás (€)"] / overview["Jelenlegi (€)"]
        )

        styled_overview = (
            overview.style.format(
                {
                    "Üzemméret (ha)": lambda x: f"{x:,.0f}".replace(",", " "),
                    "Jelenlegi (€)": lambda x: f"{x:,.0f} €".replace(",", " "),
                    "Új (€)": lambda x: f"{x:,.0f} €".replace(",", " "),
                    "Változás (€)": lambda x: f"{x:+,.0f} €".replace(",", " "),
                    "Változás (%)": lambda x: f"{x:+.1f} %",
                }
            )
            .apply(color_row, axis=1)
            .map(
                color_change,
                subset=["Változás (%)"],
            )
        )

        st.dataframe(
            styled_overview,
            hide_index=True,
            width="stretch",
        )

with data_tab:
    st.subheader(f"{base_year}-es üzemsoros állomány")

    try:
        data = load_local_data(base_year)
    except (FileNotFoundError, OSError, ImportError, ValueError) as exc:
        data = None
        st.info(f"A {base_year}-es üzemsoros állomány nem tölthető be ({exc}).")

    if data is not None:
        missing = required.difference(data.columns)
        if missing:
            st.error(
                f"A notebook API-jához szükséges oszlopok hiányoznak: {', '.join(sorted(missing))}"
            )
        else:
            total_current = float(
                data[["subs_biss", "subs_redist", "subs_yfs"]].sum().sum()
            )
            scenario = compute_capped_subsidies(data, dabis, yfs, redist, dabis_policy)
            total_gross = float(scenario["subs_eur"].sum())
            total_new = float(scenario["subs_capped"].sum())
            total_reduction = float(scenario["subs_total_reduction"].sum())

            eligible_farms = data["area_biss_criss"].fillna(0).gt(0)
            n_farms = int(eligible_farms.sum())
            eligible_area = float(data.loc[eligible_farms, "area_biss_criss"].sum())

            st.caption(f"{base_year}-es bázis · helyi állomány")

            # Bázis
            st.markdown(f"#### {base_year}-es bázis")
            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Üzemszám",
                f"{number(n_farms)} db",
            )

            k2.metric(
                "Jogosult terület",
                f"{number(eligible_area)} ha",
            )

            k3.metric(
                f"{base_year}-es támogatás",
                f"{number(total_current / 1e6, 2)} M€",
            )

            # Forgatókönyv
            st.markdown("#### DABIS-forgatókönyv")
            k4, k5, k6 = st.columns(3)

            k4.metric(
                "Bruttó DABIS",
                f"{number(total_gross / 1e6, 2)} M€",
            )

            k5.metric(
                "Kifizetett DABIS",
                f"{number(total_new / 1e6, 2)} M€",
            )

            k6.metric(
                "Összes elvonás",
                f"{number(total_reduction / 1e6, 2)} M€",
            )

            # ------------------------------------------------------------
            # Fiatal gazdálkodók: jelenlegi vs. új rendszer
            # ------------------------------------------------------------
            young_mask = data["area_yfs_cur_eligible"].fillna(0).gt(0)
            n_young = int(young_mask.sum())

            young_current = float(
                data.loc[
                    young_mask,
                    ["subs_biss", "subs_redist", "subs_yfs"],
                ]
                .sum()
                .sum()
            )

            young_new = float(
                scenario.loc[
                    young_mask,
                    "subs_capped",
                ].sum()
            )

            young_change = young_new - young_current
            young_change_pct = (
                100 * young_change / young_current if young_current > 0 else np.nan
            )

            st.markdown("#### Fiatal gazdálkodók")
            st.caption(
                "Azok a gazdaságok tartoznak ide, amelyeknél a "
                "fiatalgazda-jogosult terület nagyobb mint 0 ha."
            )

            y1, y2, y3, y4 = st.columns(4)

            y1.metric(
                "Fiatal gazdák száma",
                f"{number(n_young)} db",
            )

            y2.metric(
                "Jelenlegi támogatás",
                f"{number(young_current / 1e6, 2)} M€",
            )

            y3.metric(
                "Új DABIS",
                f"{number(young_new / 1e6, 2)} M€",
            )

            y4.metric(
                "Változás",
                f"{number(young_change / 1e6, 2)} M€",
                delta=(
                    f"{number(young_change_pct, 1)} %"
                    if np.isfinite(young_change_pct)
                    else None
                ),
            )

            with st.expander("Fiatalgazda-komponens részletei"):
                current_yfs_component = float(
                    data.loc[
                        young_mask,
                        "subs_yfs",
                    ].sum()
                )

                new_yfs_gross_component = float(
                    (
                        data.loc[
                            young_mask,
                            "area_yfs_cur_eligible",
                        ]
                        * yfs
                    ).sum()
                )

                yc1, yc2 = st.columns(2)

                yc1.metric(
                    "Jelenlegi fiatalgazda-támogatás",
                    f"{number(current_yfs_component / 1e6, 2)} M€",
                )

                yc2.metric(
                    "Új fiatalgazda-komponens – bruttó",
                    f"{number(new_yfs_gross_component / 1e6, 2)} M€",
                )

                st.caption(
                    "Az új fiatalgazda-komponens bruttó érték: "
                    "a degresszió és a capping az összesített bruttó "
                    "DABIS-ra kerül alkalmazásra, ezért a komponens "
                    "nettó része önmagában nem különíthető el egyértelműen."
                )

            with st.expander("DABIS-ráta meghatározása bruttó keretből"):
                target = st.number_input(
                    "Bruttó DABIS keret (M€)",
                    min_value=0.0,
                    value=943.17,
                    step=1.0,
                    format="%.2f",
                )

                if st.button("Ráta kiszámítása"):
                    rate = find_gross_flat_rate(
                        data,
                        target * 1e6,
                        yfs,
                        redist,
                    )

                    st.success(
                        f"A bruttó DABIS kerethez tartozó általános ráta: "
                        f"{number(rate, 2)} €/ha"
                    )

                    fitted_scenario = compute_capped_subsidies(
                        data,
                        rate,
                        yfs,
                        redist,
                        dabis_policy,
                    )

                    paid = fitted_scenario["subs_capped"].sum()
                    reduction = fitted_scenario["subs_total_reduction"].sum()

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Bruttó keret",
                        f"{number(target, 2)} M€",
                    )

                    c2.metric(
                        "Várható kifizetés",
                        f"{number(paid / 1e6, 2)} M€",
                    )

                    c3.metric(
                        "Elvonás",
                        f"{number(reduction / 1e6, 2)} M€",
                    )

            st.subheader("Az elvonás szerkezete")
            reduction_breakdown = summarize_reduction_breakdown(
                data,
                dabis,
                yfs,
                redist,
                dabis_policy,
            )

            deg_reduction = float(scenario["subs_degression_reduction"].sum())
            cap_reduction = float(scenario["subs_capping_reduction"].sum())
            cap_farms = int((scenario["subs_capping_reduction"] > 0).sum())

            rcol1, rcol2, rcol3, rcol4 = st.columns(4)
            rcol1.metric("Degresszióból", f"{number(deg_reduction / 1e6, 2)} M€")
            rcol2.metric("Cappingből", f"{number(cap_reduction / 1e6, 2)} M€")
            rcol3.metric("Összes elvonás", f"{number(total_reduction / 1e6, 2)} M€")
            rcol4.metric("Cappinggel érintett", f"{number(cap_farms)} üzem")

            chart_reduction = reduction_breakdown.set_index("Tétel")[["Elvonás (M€)"]]
            st.bar_chart(chart_reduction, y_label="Elvonás (M€)")

            reduction_display = reduction_breakdown.copy()
            reduction_display["Elvonás (€)"] = reduction_display["Elvonás (€)"].map(
                lambda value: f"{value:,.0f} €".replace(",", " ")
            )
            reduction_display["Elvonás (M€)"] = reduction_display["Elvonás (M€)"].map(
                lambda value: f"{value:.2f}"
            )
            reduction_display["Elvonási ráta (%)"] = reduction_display[
                "Elvonási ráta (%)"
            ].map(lambda value: "—" if pd.isna(value) else f"{value:.0f}%")
            reduction_display["Bruttó DABIS aránya (%)"] = reduction_display[
                "Bruttó DABIS aránya (%)"
            ].map(lambda value: f"{value:.2f}%")
            st.dataframe(reduction_display, hide_index=True, width="stretch")

            category = st.selectbox(
                "Gazdálkodói kör",
                ["összes gazda", "csak fiatal gazda", "nem fiatal gazda"],
            )
            grouped = calc_support_summary_by_area_class(
                data,
                dabis,
                redist,
                yfs,
                category,
                bins,
                labels,
                dabis_policy,
            )
            st.subheader("Hatás méretkategóriánként")

            categories = grouped.index.astype(str).tolist()
            current_meur = grouped["total_subs_cur"].to_numpy() / 1e6
            new_meur = grouped["total_subs_capped"].to_numpy() / 1e6
            pct_change = grouped["avg_change"].to_numpy()

            x = np.arange(len(categories))
            width = 0.36

            def add_labels(ax, bars, decimals=1, suffix=""):
                ymax = 0
                for bar in bars:
                    h = bar.get_height()
                    ymax = max(ymax, h)
                    ax.annotate(
                        f"{h:.{decimals}f}{suffix}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4 if h >= 0 else -4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if h >= 0 else "top",
                        fontsize=9,
                    )
                return ymax

            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(11, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [2.2, 1]},
            )

            bars1 = ax1.bar(
                x - width / 2,
                current_meur,
                width,
                label="Jelenlegi",
                color="#477ea6",
            )

            bars2 = ax1.bar(
                x + width / 2,
                new_meur,
                width,
                label="Új",
                color="#0d9871",
            )

            max1 = add_labels(ax1, bars1, decimals=1, suffix="")
            max2 = add_labels(ax1, bars2, decimals=1, suffix="")
            ymax = max(max1, max2)

            ax1.set_ylabel("Támogatás (M€)", fontsize=11, fontweight="bold")
            ax1.set_title(
                "Jelenlegi és új támogatás méretkategóriánként",
                fontsize=13,
                fontweight="bold",
            )
            ax1.legend(loc="upper right", frameon=True)
            ax1.grid(axis="y", linestyle="--", alpha=0.5)
            ax1.set_axisbelow(True)
            ax1.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)

            colors = ["#0d9871" if v >= 0 else "#b22222" for v in pct_change]

            bars3 = ax2.bar(
                x,
                pct_change,
                color=colors,
                width=0.55,
            )

            add_labels(ax2, bars3, decimals=1, suffix="%")

            ax2.axhline(0, color="black", linewidth=1)
            ax2.set_ylabel("Változás (%)", fontsize=11, fontweight="bold")
            ax2.set_title(
                "Relatív változás méretkategóriánként",
                fontsize=12,
                fontweight="bold",
            )
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.set_xlabel("Méretkategória", fontsize=11, fontweight="bold")
            ax2.grid(axis="y", linestyle="--", alpha=0.5)
            ax2.set_axisbelow(True)

            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

            detail = grouped.rename(
                columns={
                    "total_farmers": "Üzemszám",
                    "total_area": "Terület (ha)",
                    "total_subs_cur": "Jelenlegi (€)",
                    "total_subs_capped": "Új (€)",
                    "avg_change": "Fajlagos változás (%)",
                    "total_capped_loss": "Elvonás (€)",
                }
            )
            st.dataframe(detail, width="stretch")
            st.download_button(
                "Méretkategóriás táblázat letöltése",
                detail.to_csv(sep=";").encode("utf-8-sig"),
                file_name=f"kap_meretkategoria_{base_year}.csv",
                mime="text/csv",
            )

            with st.expander(f"Jogosultak és területek a {base_year}-es bázisban"):
                st.dataframe(
                    compute_dabis_support_summary(data),
                    hide_index=True,
                    width="stretch",
                )
            with st.expander("Elvonás bontása · fiatal és nem fiatal gazdálkodók"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Fiatal gazdálkodók")
                    st.dataframe(
                        summarize_reduction_breakdown(
                            data, dabis, yfs, redist, dabis_policy, "yf"
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                with col2:
                    st.write("Nem fiatal gazdálkodók")
                    st.dataframe(
                        summarize_reduction_breakdown(
                            data, dabis, 0, redist, dabis_policy, "not-yf"
                        ),
                        hide_index=True,
                        width="stretch",
                    )
