import os
import tempfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import matplotlib

from mff.utils import c_round

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mff.dabis_engine import gross_at_payment_cap
from mff.new_cap import (
    compute_capped_subsidies,
    compute_current_support,
    compute_dabis_support_summary,
    find_gross_flat_rate,
    read_extended_base_data,
)
from mff.plots import (
    PolicyParams,
    plot_diff_pct,
    plot_per_ha,
    plot_reduction,
    plot_support_comparison_by_area_class,
    plot_total,
)
from mff.scenario import ScenarioConfig, calculate_scenario
from mff.scenario_store import deserialize_scenarios, serialize_scenarios
from mff.summaries import (
    build_reduction_table,
    summarize_by_area_class,
    summarize_example_farms,
    summarize_reductions,
    summarize_young_farmers,
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
    /* ===== Main layout ===== */
    .stApp {
        background: #f5f8f8;
        color: #19313a;
    }

    .block-container {
        padding-top: 1.6rem;
        max-width: 1450px;
    }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background: #102934;
        color: #e3f2ee;
    }

    /* Sidebar headings, labels and help text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #e3f2ee;
    }

    [data-testid="stSidebar"] .stCaptionContainer,
    [data-testid="stSidebar"] .stCaptionContainer p {
        color: #9fb5bb;
    }

    /* ===== Input controls ===== */
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff;
        border-radius: 8px;
    }

    /* Explicit dark text in white input fields */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"],
    [data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] * {
        color: #19313a !important;
        -webkit-text-fill-color: #19313a;
    }

    [data-testid="stSidebar"] input::placeholder {
        color: #82979d !important;
        -webkit-text-fill-color: #82979d;
        opacity: 1;
    }

    /* Dropdown arrows */
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #526e75;
        fill: #526e75;
    }

    /* ===== Compact numeric inputs ===== */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        min-height: 2.15rem;
        padding: .25rem .55rem;
        font-variant-numeric: tabular-nums;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        display: none;
    }

    /* ===== Compact spacing ===== */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        margin-bottom: .1rem;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: .55rem;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border-radius: 10px;
    }

    /* ===== Metrics ===== */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfe9e8;
        border-radius: 12px;
        padding: 16px 19px;
    }

    /* ===== Typography ===== */
    .eyebrow {
        color: #168668;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .13em;
    }

    .title {
        font-size: clamp(2rem, 3vw, 3rem);
        line-height: 1.15;
        letter-spacing: -.04em;
        font-weight: 800;
        margin: .35rem 0;
    }

    .subtitle {
        color: #698388;
        font-size: .9rem;
        margin: 0 0 1.2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def number(value: float, decimals: int = 0) -> str:
    rounded = c_round(value, decimals)
    return f"{rounded:,.{decimals}f}".replace(",", " ").replace(".", ",")


@st.cache_data(show_spinner=False)
def tables_to_xlsx(tables: dict[str, pd.DataFrame]) -> bytes:
    """Export DataFrames to separate, formatted Excel worksheets."""
    output = BytesIO()
    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
        engine_kwargs={
            "options": {"strings_to_formulas": False, "strings_to_urls": False}
        },
    ) as writer:
        header_format = writer.book.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#102934",
                "text_wrap": True,
                "valign": "vcenter",
                "border": 0,
            }
        )
        integer_format = writer.book.add_format({"num_format": "#,##0"})
        decimal_format = writer.book.add_format({"num_format": "#,##0.00"})
        percentage_format = writer.book.add_format({"num_format": '0.00"%"'})

        for sheet_name, df in tables.items():
            if df.empty and len(df.columns) == 0:
                continue
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_row(0, 33)
            ws.set_default_row(19)
            ws.autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)

            for col_idx, col_name in enumerate(df.columns):
                ws.write(0, col_idx, str(col_name), header_format)
                series = df[col_name]
                width = min(
                    40,
                    max(
                        13,
                        len(str(col_name)) + 2,
                        max(
                            (len(str(v)) + 1 for v in series.head(100).dropna()),
                            default=0,
                        ),
                    ),
                )
                if pd.api.types.is_numeric_dtype(
                    series
                ) and not pd.api.types.is_bool_dtype(series):
                    fmt = (
                        percentage_format
                        if "(%)" in str(col_name)
                        else integer_format
                        if pd.api.types.is_integer_dtype(series)
                        else decimal_format
                    )
                else:
                    fmt = None
                ws.set_column(col_idx, col_idx, width, fmt)

    return output.getvalue()


@st.cache_data(show_spinner="Üzemsoros állomány betöltése…")
def load_local_data(year: int) -> pd.DataFrame:
    return read_extended_base_data(year)


@st.cache_data(show_spinner="Ábra készítése a plots.py függvényeivel…")
def notebook_plot(
    kind: str,
    config: ScenarioConfig,
    is_young: bool,
) -> bytes:
    """A plots.py függvényének mentett PNG-jét adja át a Streamlitnek."""
    policy = PolicyParams(
        base_payment_per_ha=config.dabis_per_ha,
        yfs_per_ha=config.yfs_per_ha if is_young else 0,
        redist_params=config.redistributive_per_ha,
        yfs_cap_ha=config.yfs_max_ha,
        dabis_policy=config.policy,
        is_young_farmer=is_young,
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


SCENARIOS_FILE = Path(__file__).resolve().parent / "data" / "scenarios.json"


def scenarios() -> dict[str, ScenarioConfig]:
    if "saved_scenarios" not in st.session_state:
        st.session_state["saved_scenarios"] = deserialize_scenarios(
            SCENARIOS_FILE.read_text(encoding="utf-8")
        )

    return st.session_state["saved_scenarios"]


def apply_scenario(config: ScenarioConfig) -> None:
    """Must run before widget creation, or inside a widget callback."""
    values = {
        "base_year": config.year,
        "dabis": config.dabis_per_ha,
        "yfs": config.yfs_per_ha,
        "yfs_max_ha": config.yfs_max_ha,
        "redist_1": config.redistributive_per_ha[0],
        "redist_2": config.redistributive_per_ha[1],
        "band_1": config.thresholds[0],
        "band_2": config.thresholds[1],
        "band_3": config.thresholds[2],
        "rate_20_50_pct": round(config.rates[0] * 100),
        "rate_50_75_pct": round(config.rates[1] * 100),
        "rate_75_plus_pct": round(config.rates[2] * 100),
        "exclude_young_farmer": config.exclude_young_farmer,
        "cap_enabled": config.payment_cap is not None,
        "cap_value": config.payment_cap
        if config.payment_cap is not None
        else 100_000.0,
    }
    st.session_state.update(values)


def on_scenario_change() -> None:
    apply_scenario(scenarios()[st.session_state["selected_scenario"]])


def import_uploaded_scenarios() -> None:
    uploaded = st.session_state.get("scenario_import_file")
    if uploaded is None:
        st.session_state["scenario_notice"] = ("warning", "Válassz egy JSON-fájlt.")
        return
    try:
        incoming = deserialize_scenarios(uploaded.getvalue().decode("utf-8"))
        conflicts = set(incoming) & set(scenarios())
        if conflicts:
            raise ValueError(
                "Már létező forgatókönyvnevek: " + ", ".join(sorted(conflicts))
            )
        scenarios().update(incoming)
        st.session_state["scenario_notice"] = (
            "success",
            f"{len(incoming)} forgatókönyv importálva.",
        )
    except (UnicodeError, ValueError, TypeError, KeyError) as exc:
        st.session_state["scenario_notice"] = ("error", f"Sikertelen import: {exc}")


if "selected_scenario" not in st.session_state:
    selected = next(iter(scenarios()))
    st.session_state["selected_scenario"] = selected
    apply_scenario(scenarios()[selected])

pending_name = st.session_state.pop("scenario_pending_select", None)
if pending_name is not None:
    st.session_state["selected_scenario"] = pending_name
    on_scenario_change()

if "pending_dabis" in st.session_state:
    st.session_state["dabis"] = st.session_state.pop("pending_dabis")

with st.sidebar:
    st.header("Forgatókönyv")
    st.selectbox(
        "Aktív forgatókönyv",
        options=list(scenarios()),
        key="selected_scenario",
        on_change=on_scenario_change,
    )
    st.caption("A szerkesztések mentésig csak ebben a munkamenetben érvényesek.")
    st.divider()
    st.subheader("Alapparaméterek")

    base_year = st.selectbox("Bázisév", options=[2024, 2025, 2026], key="base_year")

    st.caption("**Területalapú támogatások** · €/ha")
    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        dabis = st.number_input(
            "DABIS",
            min_value=0.0,
            max_value=1_000.0,
            step=5.0,
            format="%.1f",
            key="dabis",
        )
    with col_b:
        yfs = st.number_input(
            "Fiatalgazda (YFS)",
            min_value=0.0,
            max_value=500.0,
            step=5.0,
            format="%.1f",
            key="yfs",
        )

    yfs_max_ha = st.number_input(
        "YFS felső területi határa (ha)",
        min_value=1.0,
        max_value=10_000.0,
        step=10.0,
        format="%.0f",
        key="yfs_max_ha",
    )

    st.caption("**Redisztributív támogatás** · €/ha")
    redist_a, redist_b = st.columns(2, gap="small")
    with redist_a:
        redist_1 = st.number_input(
            "1. ráta",
            min_value=0.0,
            max_value=1_000.0,
            step=5.0,
            format="%.1f",
            key="redist_1",
            help="A redisztributív támogatás első fajlagos rátája.",
        )
    with redist_b:
        redist_2 = st.number_input(
            "2. ráta",
            min_value=0.0,
            max_value=1_000.0,
            step=5.0,
            format="%.1f",
            key="redist_2",
            help="A redisztributív támogatás második fajlagos rátája.",
        )
    st.caption("A két ráta területi sávjait a számítási motor rögzíti.")

    with st.expander("Degresszió és kifizetési plafon", expanded=False):
        st.caption("Bruttó támogatási sávhatárok (€), marginális elvonással.")
        t1, t2, t3 = st.columns(3, gap="small")
        with t1:
            band_1 = st.number_input(
                "Határ 1",
                min_value=1_000.0,
                max_value=500_000.0,
                step=5_000.0,
                format="%.0f",
                key="band_1",
            )
        with t2:
            band_2 = st.number_input(
                "Határ 2",
                min_value=1_000.0,
                max_value=500_000.0,
                step=5_000.0,
                format="%.0f",
                key="band_2",
            )
        with t3:
            band_3 = st.number_input(
                "Határ 3",
                min_value=1_000.0,
                max_value=1_000_000.0,
                step=5_000.0,
                format="%.0f",
                key="band_3",
            )

        st.caption("**Elvonási ráták** · %")
        r1, r2, r3 = st.columns(3, gap="small")
        with r1:
            rate_20_50_pct = st.number_input(
                "2. sáv",
                min_value=0,
                max_value=100,
                step=5,
                key="rate_20_50_pct",
            )
        with r2:
            rate_50_75_pct = st.number_input(
                "3. sáv",
                min_value=0,
                max_value=100,
                step=5,
                key="rate_50_75_pct",
            )
        with r3:
            rate_75_plus_pct = st.number_input(
                "4. sáv",
                min_value=0,
                max_value=100,
                step=5,
                key="rate_75_plus_pct",
            )

        exclude_young_farmer = st.toggle(
            "YFS mentessége az elvonás alól",
            key="exclude_young_farmer",
            help="A YFS nem része sem a degresszió, sem a capping alapjának.",
        )
        cap_enabled = st.toggle("Kifizetési plafon alkalmazása", key="cap_enabled")
        cap_value = st.number_input(
            "Kifizetési plafon (€ / üzem)",
            min_value=1_000.0,
            max_value=1_000_000.0,
            step=5_000.0,
            format="%.0f",
            disabled=not cap_enabled,
            key="cap_value",
        )

    if not 0 < band_1 < band_2 < band_3:
        st.error("A három degressziós sávhatárnak növekvő sorrendben kell lennie.")
        st.stop()

    config = ScenarioConfig(
        year=base_year,
        dabis_per_ha=dabis,
        yfs_per_ha=yfs,
        yfs_max_ha=yfs_max_ha,
        redistributive_per_ha=(redist_1, redist_2),
        thresholds=(band_1, band_2, band_3),
        rates=(rate_20_50_pct / 100, rate_50_75_pct / 100, rate_75_plus_pct / 100),
        payment_cap=cap_value if cap_enabled else None,
        exclude_young_farmer=exclude_young_farmer,
    )

    cap_trigger = gross_at_payment_cap(config.policy)
    if config.payment_cap is None:
        st.caption("Plafon: kikapcsolva")
    elif cap_trigger is None:
        st.caption("A beállított plafon e ráták mellett nem válik kötővé.")
    else:
        st.caption(
            f"Plafon érvényesülése: kb. {number(cap_trigger)} € bruttó összegtől"
        )

    st.divider()
    st.caption(
        f"A {base_year}-es BISS + CRISS + fiatal gazda támogatáshoz hasonlít. "
        "Az AÖP és más támogatási elemek itt nem szerepelnek."
    )

    st.divider()
    st.subheader("Forgatókönyvek mentése")
    name = st.text_input("Új forgatókönyv neve", key="new_scenario_name")
    if st.button("Mentés új néven"):
        new_name = name.strip()
        if not new_name or len(new_name) > 100:
            st.warning("Adj meg egy legfeljebb 100 karakteres nevet.")
        elif new_name in scenarios():
            st.warning("Ilyen néven már létezik forgatókönyv.")
        else:
            scenarios()[new_name] = config
            st.session_state["scenario_pending_select"] = new_name
            st.rerun()

    selected_name = st.session_state["selected_scenario"]
    if selected_name != next(iter(scenarios())):
        left, right = st.columns(2)
        if left.button("Felülírás", use_container_width=True):
            scenarios()[selected_name] = config
            st.success("Forgatókönyv mentve.")
        if right.button("Törlés", use_container_width=True):
            del scenarios()[selected_name]
            st.session_state["scenario_pending_select"] = next(iter(scenarios()))
            st.rerun()

    st.download_button(
        "Forgatókönyvek exportálása (JSON)",
        data=serialize_scenarios(scenarios()),
        file_name="scenarios.json",
        mime="application/json",
    )
    st.file_uploader(
        "Forgatókönyvek importálása (JSON)", type="json", key="scenario_import_file"
    )
    st.button("Importálás", on_click=import_uploaded_scenarios)
    notice = st.session_state.pop("scenario_notice", None)
    if notice is not None:
        getattr(st, notice[0])(notice[1])


required = {
    "area_biss_criss",
    "area_yfs_cur_eligible",
    "subs_biss",
    "subs_redist",
    "subs_yfs",
}

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

    farm = pd.DataFrame(
        {
            "area_biss_criss": [area],
            "area_yfs_cur_eligible": [min(area, config.yfs_max_ha) if is_young else 0],
        }
    )
    farm_new = compute_capped_subsidies(
        farm,
        config.dabis_per_ha,
        config.yfs_per_ha,
        config.redistributive_per_ha,
        config.policy,
    ).iloc[0]
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

    try:
        st.image(
            notebook_plot(chart_kind, config, is_young),
            width="stretch",
        )
    except ValueError as exc:
        st.warning(f"Az ábra ezekkel a paraméterekkel nem készíthető el: {exc}")

    with st.expander("Jellemző üzemméretek"):
        overview = summarize_example_farms(config, is_young)

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
        st.download_button(
            "Jellemző üzemméretek letöltése (Excel)",
            data=tables_to_xlsx({"Üzemméretek": overview}),
            file_name=f"kap_uzemmeretek_{base_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with data_tab:
    st.subheader(f"{base_year}-es üzemsoros állomány")

    try:
        data: pd.DataFrame | None = load_local_data(base_year)
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
            results = calculate_scenario(data, config)

            st.caption(f"{base_year}-es bázis · helyi állomány")

            st.markdown(f"#### {base_year}-es bázis")
            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Üzemszám",
                f"{number(results.eligible_farms)} db",
            )

            k2.metric(
                "Jogosult terület",
                f"{number(results.eligible_area)} ha",
            )

            k3.metric(
                f"{base_year}-es támogatás",
                f"{number(results.total_current / 1e6, 2)} M€",
            )

            # Forgatókönyv
            st.markdown("#### DABIS-forgatókönyv")
            k4, k5, k6 = st.columns(3)

            k4.metric(
                "Bruttó DABIS",
                f"{number(results.total_gross / 1e6, 2)} M€",
            )

            k5.metric(
                "Kifizetett DABIS",
                f"{number(results.total_paid / 1e6, 2)} M€",
            )

            k6.metric(
                "Összes elvonás",
                f"{number(results.total_reduction / 1e6, 2)} M€",
            )

            # ------------------------------------------------------------
            # Fiatal gazdálkodók: jelenlegi vs. új rendszer
            # ------------------------------------------------------------
            young = summarize_young_farmers(results, config)

            st.markdown("#### Fiatal gazdálkodók")
            st.caption(
                "Azok a gazdaságok tartoznak ide, amelyeknél a "
                "fiatalgazda-jogosult terület nagyobb mint 0 ha."
            )

            y1, y2, y3, y4 = st.columns(4)

            y1.metric(
                "Fiatal gazdák száma",
                f"{number(young.farms)} db",
            )

            y2.metric(
                "Jelenlegi támogatás",
                f"{number(young.current / 1e6, 2)} M€",
            )

            y3.metric(
                "Új DABIS",
                f"{number(young.new / 1e6, 2)} M€",
            )

            y4.metric(
                "Változás",
                f"{number(young.change / 1e6, 2)} M€",
                delta=(
                    f"{number(young.change_pct, 1)} %"
                    if np.isfinite(young.change_pct)
                    else None
                ),
            )

            with st.expander("Fiatalgazda-komponens részletei"):
                yc1, yc2 = st.columns(2)

                yc1.metric(
                    "Jelenlegi fiatalgazda-támogatás",
                    f"{number(young.current_yfs / 1e6, 2)} M€",
                )

                yc2.metric(
                    "Új fiatalgazda-komponens – bruttó",
                    f"{number(young.new_yfs_gross / 1e6, 2)} M€",
                )

                if config.exclude_young_farmer:
                    st.caption(
                        "A fiatalgazda-komponens nem része az elvonási alapnak, "
                        "és teljes egészében kifizetésre kerül."
                    )
                else:
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
                    rate_data = data.copy()
                    rate_data["area_yfs_cur_eligible"] = config.young_area(data)

                    rate = find_gross_flat_rate(
                        rate_data,
                        target * 1e6,
                        config.yfs_per_ha,
                        config.redistributive_per_ha,
                    )

                    st.success(
                        f"A bruttó DABIS kerethez tartozó általános ráta: "
                        f"{number(rate, 2)} €/ha"
                    )

                    fitted = calculate_scenario(
                        data,
                        replace(config, dabis_per_ha=rate),
                    )
                    paid = fitted.total_paid
                    reduction = fitted.total_reduction

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
            reductions = summarize_reductions(results, config)

            reduction_breakdown = reductions["all"]

            rcol1, rcol2, rcol3, rcol4 = st.columns(4)
            rcol1.metric(
                "Degresszióból", f"{number(results.degression_reduction / 1e6, 2)} M€"
            )
            rcol2.metric(
                "Cappingből", f"{number(results.capping_reduction / 1e6, 2)} M€"
            )
            rcol3.metric(
                "Összes elvonás", f"{number(results.total_reduction / 1e6, 2)} M€"
            )
            rcol4.metric("Cappinggel érintett", f"{number(results.capped_farms)} üzem")

            chart_reduction = reduction_breakdown.set_index("Tétel")[["Elvonás (M€)"]]
            st.bar_chart(chart_reduction, y_label="Elvonás (M€)")

            reduction_display = build_reduction_table(
                data,
                config,
                breakdown=reduction_breakdown,
                scenario=results.new,
            ).drop(columns=["Elvonás (€)"])

            st.caption(
                "A bruttó és nettó sávösszegek a támogatás marginális részeire vonatkoznak; "
                "a sávonkénti elvonás a degresszióból eredő levonás. "
                "A sávba sorolt gazdaságok mutatói ezzel szemben az adott bruttó "
                "támogatási kategória teljes támogatását, területét és összes "
                "(degressziós és plafon miatti) elvonását írják le. "
                "A teljes elvonás aránya csoportszintű hányados, nem a sávban "
                "alkalmazott marginális ráta. A sávval érintett gazdaságok több "
                "sávban szerepelhetnek. YFS-mentességnél a gazdaságokat teljes "
                "bruttó támogatásuk szerint soroljuk be, a marginális elvonási "
                "alap azonban nem tartalmazza a mentesített YFS-t. "
                "A plafon sora átfedő: csoportadatai nem összegezhetők a négy "
                "sáv adataival. A plafon sorának 'Elvonás (M€)' értéke kizárólag "
                "a plafon miatti többletelvonás, ez hozzáadható a sávok "
                "degressziós elvonásához. A 'Plafon feletti bruttó alap' "
                "nem azonos a plafon miatti tényleges elvonással."
            )

            # A megjelenítés formázott; az XLSX-export számértékeket kap.
            formatters = {
                "Elvonás (M€)": lambda x: number(x, 2),
                "Elvonási ráta (%)": lambda x: f"{number(x, 2)}%",
                "Bruttó DABIS aránya (%)": lambda x: f"{number(x, 2)}%",
                "Bruttó DABIS (M€)": lambda x: number(x, 2),
                "Nettó DABIS (M€)": lambda x: number(x, 2),
                "Sávba tartozó gazdaságok (db)": lambda x: number(x),
                "Sávval érintett gazdaságok (db)": lambda x: number(x),
                "Sávba tartozó gazdaságok területe (ha)": lambda x: number(x, 1),
                "Sávba tartozók bruttó támogatása (M€)": lambda x: number(x, 2),
                "Sávba tartozók teljes elvonása (M€)": lambda x: number(x, 2),
                "Sávba tartozók nettó támogatása (M€)": lambda x: number(x, 2),
                "Elvonással érintett gazdaságok (db)": lambda x: number(x),
                "Átlagos teljes elvonás (€/gazdaság)": lambda x: number(x, 0),
                "Teljes elvonás aránya (%)": lambda x: f"{number(x, 2)}%",
                "Plafon feletti bruttó alap (M€)": lambda x: number(x, 2),
                "Plafon miatti elvonás (M€)": lambda x: number(x, 2),
            }
            formatted = reduction_display.style.format(
                {
                    col: fmt
                    for col, fmt in formatters.items()
                    if col in reduction_display
                },
                na_rep="—",
            )
            st.dataframe(formatted, hide_index=True, width="stretch")
            st.download_button(
                "Elvonás szerkezete letöltése (Excel)",
                data=tables_to_xlsx({"Elvonás": reduction_display}),
                file_name=f"kap_elvonas_{base_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            category = st.selectbox(
                "Gazdálkodói kör",
                ["összes gazda", "csak fiatal gazda", "nem fiatal gazda"],
            )
            grouped = summarize_by_area_class(
                results,
                category,
            )
            st.subheader("Hatás méretkategóriánként")

            fig = plot_support_comparison_by_area_class(grouped)
            try:
                st.pyplot(fig)
            finally:
                plt.close(fig)

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
                "Méretkategóriás táblázat letöltése (Excel)",
                data=tables_to_xlsx({"Méretkategóriák": detail}),
                file_name=f"kap_meretkategoria_{base_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            base_summary = compute_dabis_support_summary(data)
            with st.expander(f"Jogosultak és területek a {base_year}-es bázisban"):
                st.dataframe(
                    base_summary,
                    hide_index=True,
                    width="stretch",
                )
                st.download_button(
                    "Jogosultak és területek letöltése (Excel)",
                    data=tables_to_xlsx({"Jogosultak és területek": base_summary}),
                    file_name=f"kap_jogosult_terulet_{base_year}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with st.expander("Elvonás bontása · fiatal és nem fiatal gazdálkodók"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("Fiatal gazdálkodók")
                    st.dataframe(
                        reductions["young"],
                        hide_index=True,
                        width="stretch",
                    )
                    st.download_button(
                        "Fiatal gazdálkodók elvonása (Excel)",
                        data=tables_to_xlsx({"Fiatalok": reductions["young"]}),
                        file_name=f"kap_elvonas_fiatal_{base_year}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                with col2:
                    st.write("Nem fiatal gazdálkodók")
                    st.dataframe(
                        reductions["non_young"],
                        hide_index=True,
                        width="stretch",
                    )
                    st.download_button(
                        "Nem fiatal gazdálkodók elvonása (Excel)",
                        data=tables_to_xlsx({"Nem fiatalok": reductions["non_young"]}),
                        file_name=f"kap_elvonas_nem_fiatal_{base_year}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            st.divider()
            st.download_button(
                "Összes táblázat letöltése (Excel)",
                data=tables_to_xlsx(
                    {
                        "Üzemméretek": overview,
                        "Elvonás": reduction_display,
                        "Méretkategóriák": detail,
                        "Jogosultak és területek": base_summary,
                        "Fiatalok": reductions["young"],
                        "Nem fiatalok": reductions["non_young"],
                    }
                ),
                file_name=f"kap_osszes_tabla_{base_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
