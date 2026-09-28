import os
import tempfile
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mff.new_cap import (
    calc_degressive_and_capping_steps,
    compute_capped_subsidies,
    compute_current_support,
    compute_dabis_support_summary,
    find_budget,
    find_flat_rate,
    generate_labels_from_bins,
    read_extended_base_data,
)
from mff.plots import (
    PolicyParams,
    build_rate_area_matrix,
    calc_support_summary_by_area_class,
    plot_diff_pct,
    plot_per_ha,
    plot_reduction,
    plot_total,
)

st.set_page_config(page_title="KAP Hatáselemző", page_icon="🌾", layout="wide")
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


@st.cache_data(show_spinner="2024-es üzemsoros állomány betöltése…")
def load_local_data(year: int) -> pd.DataFrame:
    # Pontosan ugyanaz az adatbetöltő, mint a notebook első futtatott cellájában.
    return read_extended_base_data(year)


@st.cache_data(show_spinner="Ábra készítése a plots.py függvényeivel…")
def notebook_plot(
    kind: str, dabis: float, yfs: float, redist: tuple[float, float], is_young: bool
) -> bytes:
    """A plots.py függvényének mentett PNG-jét adja át a Streamlitnek."""
    policy = PolicyParams(
        base_payment_per_ha=dabis,
        yfs_per_ha=yfs if is_young else 0,
        redist_params=redist,
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
    st.caption("A notebook induló értékei: 220 / 50 / (0, 0) €/ha")
    dabis = st.number_input(
        "DABIS (€/ha)",
        min_value=100.0,
        max_value=500.0,
        value=220.0,
        step=5.0,
        key="dabis",
    )
    yfs = st.number_input(
        "Fiatal gazda (€/ha)", min_value=0.0, max_value=200.0, value=50.0, step=5.0
    )
    r1 = st.number_input(
        "Redisztribúció · 0–10 ha (€/ha)",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
    )
    r2 = st.number_input(
        "Redisztribúció · 10–150 ha (€/ha)",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
    )
    st.divider()
    st.caption(
        "A 2024-es BISS + CRISS + fiatal gazda támogatáshoz hasonlít. Az AÖP és más támogatási elemek itt nem szerepelnek."
    )

redist = (r1, r2)
bins = [0, 10, 150, 300, 1200, float("inf")]
labels = generate_labels_from_bins(bins)

st.markdown(
    '<div class="eyebrow">2024-ES BÁZIS · ÚJ KAP FORGATÓKÖNYV</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="title">KAP Hatáselemző</div>', unsafe_allow_html=True)

farm_tab, data_tab, rates_tab = st.tabs(
    [
        "Üzemméret szerinti hatás",
        "Üzemsoros eredmények",
        "Keret és ráták",
    ]
)

with farm_tab:
    a, b = st.columns(2)
    with a:
        area = st.number_input(
            "Üzemméret (ha)", min_value=0.1, max_value=10000.0, value=100.0, step=1.0
        )
    with b:
        is_young = st.checkbox("Fiatal gazdálkodó", value=False)

    # Az új támogatást a notebook compute_capped_subsidies függvénye számolja.
    farm = pd.DataFrame(
        {
            "area_biss_criss": [area],
            "area_yfs_cur_eligible": [min(area, 300) if is_young else 0],
        }
    )
    farm_new = compute_capped_subsidies(farm, dabis, yfs, redist).iloc[0]
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
                notebook_plot(chart_kind, dabis, yfs, redist, is_young), width="stretch"
            )
        except ValueError as exc:
            st.warning(f"Az ábra ezekkel a paraméterekkel nem készíthető el: {exc}")
    st.caption(
        "Az ábrákat közvetlenül az mff.plots függvényei készítik; a számítási szabályok az mff.new_cap modulból származnak."
    )

    with st.expander("Jellemző üzemméretek"):
        areas = [5, 10, 50, 100, 150, 300, 500, 1200, 1500]
        examples = pd.DataFrame(
            {
                "area_biss_criss": areas,
                "area_yfs_cur_eligible": [
                    min(x, 300) if is_young else 0 for x in areas
                ],
            }
        )
        examples_new = compute_capped_subsidies(examples, dabis, yfs, redist)
        current = [compute_current_support(x, is_young) for x in areas]
        overview = pd.DataFrame(
            {
                "Üzemméret (ha)": areas,
                "Jelenlegi (€)": current,
                "Új (€)": examples_new["subs_capped"].to_numpy(),
            }
        )
        overview["Változás (€)"] = overview["Új (€)"] - overview["Jelenlegi (€)"]
        st.dataframe(overview, hide_index=True, width="stretch")

with data_tab:
    st.subheader("2024-es üzemsoros állomány")
    uploaded = st.file_uploader(
        "Másik data_extended_2024.parquet betöltése (opcionális)",
        type=["parquet"],
        help="Távoli Streamlit telepítésnél a fájl a szerverre kerül.",
    )
    try:
        data = (
            pd.read_parquet(BytesIO(uploaded.getvalue()))
            if uploaded
            else load_local_data(2024)
        )
    except (FileNotFoundError, OSError, ImportError, ValueError) as exc:
        data = None
        st.info(
            f"Nincs betöltött üzemsoros állomány ({exc}). Helyezd az input/data_extended_2024.parquet fájlt a projektbe, vagy töltsd fel itt."
        )

    if data is not None:
        required = {
            "area_biss_criss",
            "area_yfs_cur_eligible",
            "subs_biss",
            "subs_redist",
            "subs_yfs",
        }
        missing = required.difference(data.columns)
        if missing:
            st.error(
                f"A notebook API-jához szükséges oszlopok hiányoznak: {', '.join(sorted(missing))}"
            )
        else:
            st.caption(
                f"{number(len(data))} üzem · {number(data['area_biss_criss'].sum())} ha · {'feltöltött' if uploaded else 'helyi'} állomány"
            )
            total_current = float(
                data[["subs_biss", "subs_redist", "subs_yfs"]].sum().sum()
            )
            total_new = find_budget(data, dabis, yfs, redist)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Üzemszám", number(len(data)))
            k2.metric("2024-es összeg", f"{number(total_current / 1e6, 2)} M€")
            k3.metric("Új keret", f"{number(total_new / 1e6, 2)} M€")
            k4.metric("Változás", f"{number((total_new - total_current) / 1e6, 2)} M€")

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
            )
            st.subheader("Hatás méretkategóriánként")
            plot_data = (
                grouped[["total_subs_cur", "total_subs_capped"]].rename(
                    columns={"total_subs_cur": "Jelenlegi", "total_subs_capped": "Új"}
                )
                / 1e6
            )
            st.bar_chart(
                plot_data, color=["#477ea6", "#0d9871"], y_label="Támogatás (M€)"
            )
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
                file_name="kap_meretkategoria_2024.csv",
                mime="text/csv",
            )

            with st.expander("Jogosultak és területek a 2024-es bázisban"):
                st.dataframe(
                    compute_dabis_support_summary(data),
                    hide_index=True,
                    width="stretch",
                )
            with st.expander("Degressziós sávok · fiatal és nem fiatal gazdálkodók"):
                st.caption(
                    "A notebook calc_degressive_and_capping_steps függvénye szerinti bontás."
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Fiatal gazdálkodók")
                    st.dataframe(
                        calc_degressive_and_capping_steps(
                            data, dabis, redist, yfs, "yf"
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                with col2:
                    st.write("Nem fiatal gazdálkodók")
                    st.dataframe(
                        calc_degressive_and_capping_steps(
                            data, dabis, redist, 0, "not-yf"
                        ),
                        hide_index=True,
                        width="stretch",
                    )

with rates_tab:
    st.subheader("DABIS-ráta és szükséges keret")
    if data is None or required.difference(data.columns):
        st.info(
            "A keretszámításhoz tölts be 2024-es üzemsoros állományt az előző lapon."
        )
    else:
        st.metric(
            "A beállított ráta szükséges kerete",
            f"{number(find_budget(data, dabis, yfs, redist) / 1e6, 2)} M€",
        )
        target = st.number_input(
            "Célkeret (M€)", min_value=0.0, value=943.16708678, step=1.0, format="%.2f"
        )
        if st.button("DABIS-ráta illesztése", type="primary"):
            try:
                root = find_flat_rate(data, target * 1e6, yfs, redist).root
                st.session_state["fitted_rate"] = root
                st.session_state["fitted_target"] = target
                st.session_state["fitted_yfs_redist"] = (yfs, redist)
            except ValueError as exc:
                st.error(
                    f"A notebook 100–500 €/ha közötti gyökkeresése nem talált megoldást: {exc}"
                )
        if "fitted_rate" in st.session_state:
            if st.session_state.get("fitted_target") == target and st.session_state.get(
                "fitted_yfs_redist"
            ) == (yfs, redist):
                st.success(
                    f"Illesztett DABIS-ráta: {number(st.session_state.fitted_rate, 2)} €/ha"
                )
                if st.button("Illesztett ráta átvétele"):
                    st.session_state["pending_dabis"] = float(
                        st.session_state.fitted_rate
                    )
                    st.rerun()
            else:
                st.info(
                    "A keret vagy valamelyik kiegészítés változott; számítsd újra az illesztést."
                )

        st.divider()
        st.subheader("Ráta és méretkategória")
        st.caption(
            "A notebook build_rate_area_matrix függvénye itt 0 €/ha új fiatal gazda támogatással számol."
        )
        if st.button("Rátasöprés számítása (100–300 €/ha)"):
            rates = np.linspace(100, 300, 30)
            matrix = build_rate_area_matrix(data, redist, bins, labels, rates)
            pivot = matrix.pivot(
                index="area_class", columns="subs_per_ha", values="perc_diff"
            )
            st.session_state["sweep_table"] = pivot.reindex(labels)
        if "sweep_table" in st.session_state:
            st.dataframe(
                st.session_state.sweep_table.style.background_gradient(
                    cmap="RdYlGn", axis=None
                ),
                width="stretch",
            )

st.caption(
    "Forrás: a csatolt new_cap_vs_current_cap notebook és az mff.new_cap / mff.plots modulok. A 2028–2034-es összegek forgatókönyvek, nem elfogadott szabályok."
)
