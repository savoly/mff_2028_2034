from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="MFF 2028–2034 | CAP / NRPP dashboard",
    page_icon="🌾",
    layout="wide",
)

st.title("MFF 2028–2034 – CAP / NRPP allokációs dashboard")
st.caption(
    "Valós 2024-es EK-adatbázison: DABIS, YF, SFS, CIS, "
    "ring-fenced / non-ring-fenced keretek, zöld tracking és társfinanszírozás."
)


# ============================================================
# CONSTANTS / INPUT STRUCTURE
# ============================================================

DATA_DIR = Path("input")

REQUIRED_COLUMNS = [
    "regszam",
    "area_biss_criss",
    "area_yfs",
    "area_aop",
    "area_vp_akg_2021",
    "area_tk_cukorrepa",
    "area_tk_szemes_feherjenoveny",
    "area_tk_szalas_feherjenoveny",
    "area_tk_extenziv_gyumolcs",
    "area_tk_intenziv_gyumolcs",
    "area_tk_ipari_olajnoveny",
    "area_tk_ipari_zoldsegnoveny",
    "area_tk_zoldsegnoveny",
    "area_tk_rizs",
    "count_tk_hizottbika",
    "count_tk_anyatehen",
    "count_tk_tejhasznu_tehen",
    "count_tk_anyajuh",
    "area_yfs_cur_eligible",
    "subs_biss",
    "subs_redist",
    "subs_yfs",
]

CIS_COMPONENTS = {
    "Cukorrépa": ("area_tk_cukorrepa", "ha"),
    "Szemes fehérjenövény": ("area_tk_szemes_feherjenoveny", "ha"),
    "Szálas fehérjenövény": ("area_tk_szalas_feherjenoveny", "ha"),
    "Extenzív gyümölcs": ("area_tk_extenziv_gyumolcs", "ha"),
    "Intenzív gyümölcs": ("area_tk_intenziv_gyumolcs", "ha"),
    "Ipari olajnövény": ("area_tk_ipari_olajnoveny", "ha"),
    "Ipari zöldségnövény": ("area_tk_ipari_zoldsegnoveny", "ha"),
    "Zöldségnövény": ("area_tk_zoldsegnoveny", "ha"),
    "Rizs": ("area_tk_rizs", "ha"),
    "Hízott bika": ("count_tk_hizottbika", "állat"),
    "Anyatehén": ("count_tk_anyatehen", "állat"),
    "Tejhasznú tehén": ("count_tk_tejhasznu_tehen", "állat"),
    "Anyajuh": ("count_tk_anyajuh", "állat"),
}

# Tracking defaults used as editable modelling assumptions.
TRACKING_DEFAULTS = {
    "DABIS": 0.40,
    "YF top-up": 0.40,
    "SFS": 0.40,
    "AEC": 1.00,
    "ANC": 0.40,
    "CIS": 0.40,
    "Green investment": 1.00,
    "Risk management": 1.00,
    "Young farmer startup": 0.00,
}


# ============================================================
# HELPERS
# ============================================================


def eur_m(value: float) -> str:
    return f"{value:,.1f} M€".replace(",", " ").replace(".", ",")


def eur_bn(value: float) -> str:
    return f"{value:,.2f} mrd €".replace(",", " ").replace(".", ",")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%".replace(".", ",")


@st.cache_data(show_spinner=False)
def read_extended_data(base_year: int) -> pd.DataFrame:
    """
    Input produced by generate_extended_base_data():
        input/data_extended_{base_year}.parquet
    """
    path = DATA_DIR / f"data_extended_{base_year}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Nem találom az inputot: {path}. "
            "Futtasd előbb a generate_extended_base_data() függvényt."
        )

    data = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            "Hiányzó oszlopok a data_extended fájlból: " + ", ".join(missing)
        )

    data = data.copy()

    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "regszam"]
    data[numeric_cols] = data[numeric_cols].fillna(0)

    return data


def dabis_after_degression(x: np.ndarray) -> np.ndarray:
    """
    Modellezett degresszió:
      20–50k €: 25%
      50–75k €: 50%
      75k € felett: 75%
    """
    x = np.asarray(x, dtype=float)

    r1 = 0.25 * np.clip(
        np.minimum(x, 50_000) - 20_000,
        0,
        30_000,
    )
    r2 = 0.50 * np.clip(
        np.minimum(x, 75_000) - 50_000,
        0,
        25_000,
    )
    r3 = 0.75 * np.clip(
        x - 75_000,
        0,
        None,
    )

    return x - r1 - r2 - r3


def national_from_eu(
    eu_amount_m: float,
    national_share: float,
) -> float:
    """
    national_share = nemzeti rész / teljes közkiadás.
    """
    if not 0 <= national_share < 1:
        raise ValueError("A nemzeti aránynak [0, 1) intervallumban kell lennie.")

    return eu_amount_m * national_share / (1.0 - national_share)


# ============================================================
# LOAD DATA
# ============================================================

st.sidebar.header("Adat")

base_year = st.sidebar.selectbox(
    "Bázisév",
    options=[2024],
    index=0,
)

try:
    data = read_extended_data(base_year)
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

n_farms = len(data)
area_biss = data["area_biss_criss"].to_numpy(dtype=float)
total_area_ha = area_biss.sum()

baseline_subs_m = data[["subs_biss", "subs_redist", "subs_yfs"]].sum().sum() / 1_000_000


# ============================================================
# SIDEBAR – CAP DESIGN
# ============================================================

st.sidebar.divider()
st.sidebar.header("CAP design")

dabis_rate = st.sidebar.slider(
    "DABIS fajlagos (€/ha)",
    min_value=130,
    max_value=240,
    value=220,
    step=5,
)

apply_degression = st.sidebar.checkbox(
    "Degresszió alkalmazása",
    value=True,
)

young_topup = st.sidebar.slider(
    "Fiatal gazda top-up (€/ha)",
    min_value=0,
    max_value=150,
    value=50,
    step=5,
)

sfs_max_area = st.sidebar.slider(
    "SFS modellezett felső területhatár (ha)",
    min_value=1.0,
    max_value=50.0,
    value=10.0,
    step=1.0,
)

sfs_flat = st.sidebar.slider(
    "SFS átalány (€/gazdaság/év)",
    min_value=0,
    max_value=3_000,
    value=1_500,
    step=100,
)

st.sidebar.subheader("Egyéb éves EU-borítékok")

aec_eu_m = st.sidebar.number_input(
    "AEC / környezet-klíma (M€/év)",
    min_value=0.0,
    value=180.0,
    step=10.0,
)

anc_eu_m = st.sidebar.number_input(
    "ANC (M€/év)",
    min_value=0.0,
    value=120.0,
    step=10.0,
)

investment_eu_m = st.sidebar.number_input(
    "Beruházás (M€/év)",
    min_value=0.0,
    value=160.0,
    step=10.0,
)

risk_eu_m = st.sidebar.number_input(
    "Kockázatkezelés (M€/év)",
    min_value=0.0,
    value=70.0,
    step=10.0,
)

young_startup_eu_m = st.sidebar.number_input(
    "Fiatal gazda induló (M€/év)",
    min_value=0.0,
    value=0.0,
    step=5.0,
)


# ============================================================
# SIDEBAR – CIS
# ============================================================

st.sidebar.divider()
st.sidebar.header("CIS")

cis_alpha = st.sidebar.select_slider(
    "CIS plafon",
    options=[0.20, 0.25],
    value=0.20,
    format_func=lambda x: f"{x:.0%}",
)

with st.sidebar.expander("CIS egységösszegek", expanded=False):
    cis_rates = {}

    for label, (_, unit) in CIS_COMPONENTS.items():
        cis_rates[label] = st.number_input(
            f"{label} (€/ {unit})",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key=f"cis_rate_{label}",
        )


# ============================================================
# SIDEBAR – NRPP / RING-FENCING
# ============================================================

st.sidebar.divider()
st.sidebar.header("NRPP / ring-fencing")

nrpp_total_bn = st.sidebar.number_input(
    "Magyar NRPP összesen (mrd €)",
    min_value=1.0,
    value=37.70,
    step=0.10,
)

migration_bn = st.sidebar.number_input(
    "Migration / Security earmark (mrd €)",
    min_value=0.0,
    value=0.50,
    step=0.05,
)

scf_bn = st.sidebar.number_input(
    "Social Climate Fund (mrd €)",
    min_value=0.0,
    value=2.20,
    step=0.05,
)

cap_min_bn = st.sidebar.number_input(
    "CAP ring-fenced minimum (mrd €)",
    min_value=0.0,
    value=9.24,
    step=0.05,
)

ldr_min_bn = st.sidebar.number_input(
    "Kevésbé fejlett régiók minimuma (mrd €)",
    min_value=0.0,
    value=20.71269,
    step=0.05,
)

ldr_planned_bn = st.sidebar.number_input(
    "Kevésbé fejlett régiók tervezett kerete (mrd €)",
    min_value=0.0,
    value=float(ldr_min_bn),
    step=0.05,
)

other_flexible_use_bn = st.sidebar.number_input(
    "Egyéb felhasznált szabad keret (mrd €)",
    min_value=0.0,
    value=0.0,
    step=0.05,
)


# ============================================================
# SIDEBAR – GREEN TRACKING
# ============================================================

st.sidebar.divider()
st.sidebar.header("Klíma / környezet tracking")

nrpp_green_target = st.sidebar.slider(
    "NRPP cél",
    min_value=0.30,
    max_value=0.60,
    value=0.43,
    step=0.01,
    format="%.2f",
)

cis_tracking = st.sidebar.slider(
    "CIS átlagos tracking",
    min_value=0.0,
    max_value=1.0,
    value=TRACKING_DEFAULTS["CIS"],
    step=0.05,
    format="%.2f",
)

anc_tracking = st.sidebar.slider(
    "ANC átlagos tracking",
    min_value=0.0,
    max_value=1.0,
    value=TRACKING_DEFAULTS["ANC"],
    step=0.05,
    format="%.2f",
)

green_investment_share = st.sidebar.slider(
    "Beruházásból green investment",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
    format="%.2f",
)

other_nrpp_green_bn = st.sidebar.number_input(
    "Nem CAP NRPP zöld tracking (mrd €/7 év)",
    min_value=0.0,
    value=0.0,
    step=0.1,
)


# ============================================================
# SIDEBAR – COFINANCING
# ============================================================

st.sidebar.divider()
st.sidebar.header("Magyar társfinanszírozás")

cofin_default = 0.30

aec_nat_share = st.sidebar.slider(
    "AEC magyar rész",
    0.0,
    0.80,
    cofin_default,
    0.05,
    format="%.2f",
)
anc_nat_share = st.sidebar.slider(
    "ANC magyar rész",
    0.0,
    0.80,
    cofin_default,
    0.05,
    format="%.2f",
)
investment_nat_share = st.sidebar.slider(
    "Beruházás magyar rész",
    0.0,
    0.80,
    cofin_default,
    0.05,
    format="%.2f",
)
risk_nat_share = st.sidebar.slider(
    "Kockázatkezelés magyar rész",
    0.0,
    0.80,
    cofin_default,
    0.05,
    format="%.2f",
)
young_startup_nat_share = st.sidebar.slider(
    "Fiatal gazda induló magyar rész",
    0.0,
    0.80,
    cofin_default,
    0.05,
    format="%.2f",
)


# ============================================================
# CAP CALCULATION ON REAL DATA
# ============================================================

gross_dabis_eur = area_biss * dabis_rate

if apply_degression:
    paid_dabis_eur = dabis_after_degression(gross_dabis_eur)
else:
    paid_dabis_eur = gross_dabis_eur

dabis_gross_m = gross_dabis_eur.sum() / 1_000_000
dabis_paid_m = paid_dabis_eur.sum() / 1_000_000
dabis_reduction_m = (gross_dabis_eur.sum() - paid_dabis_eur.sum()) / 1_000_000

yfs_area_ha = data["area_yfs_cur_eligible"].sum()
young_budget_m = yfs_area_ha * young_topup / 1_000_000

sfs_mask = (data["area_biss_criss"] > 0) & (data["area_biss_criss"] <= sfs_max_area)
sfs_farms = int(sfs_mask.sum())
sfs_budget_m = sfs_farms * sfs_flat / 1_000_000

cis_rows = []

for label, (column, unit) in CIS_COMPONENTS.items():
    quantity = float(data[column].sum())
    rate = cis_rates[label]
    amount_m = quantity * rate / 1_000_000

    cis_rows.append(
        {
            "CIS komponens": label,
            "Jogosult mennyiség": quantity,
            "Egység": unit,
            "Egységösszeg (€)": rate,
            "Boríték (M€)": amount_m,
        }
    )

cis_table = pd.DataFrame(cis_rows)
cis_budget_m = cis_table["Boríték (M€)"].sum()

cis_reference_m = dabis_paid_m + aec_eu_m + sfs_budget_m
cis_max_m = cis_alpha * cis_reference_m
cis_headroom_m = cis_max_m - cis_budget_m

cap_annual_eu_m = (
    dabis_paid_m
    + young_budget_m
    + sfs_budget_m
    + cis_budget_m
    + aec_eu_m
    + anc_eu_m
    + investment_eu_m
    + risk_eu_m
    + young_startup_eu_m
)

cap_7y_eu_bn = cap_annual_eu_m * 7 / 1_000


# ============================================================
# RING-FENCED / NON-RING-FENCED
# ============================================================

general_allocation_bn = nrpp_total_bn - migration_bn - scf_bn

initial_non_ring_bn = general_allocation_bn - cap_min_bn - ldr_min_bn

cap_gap_bn = max(
    cap_min_bn - cap_7y_eu_bn,
    0.0,
)
cap_extra_bn = max(
    cap_7y_eu_bn - cap_min_bn,
    0.0,
)

ldr_gap_bn = max(
    ldr_min_bn - ldr_planned_bn,
    0.0,
)
ldr_extra_bn = max(
    ldr_planned_bn - ldr_min_bn,
    0.0,
)

non_ring_remaining_bn = (
    initial_non_ring_bn - cap_extra_bn - ldr_extra_bn - other_flexible_use_bn
)

ring_valid = (
    cap_gap_bn <= 1e-9
    and ldr_gap_bn <= 1e-9
    and initial_non_ring_bn >= -1e-9
    and non_ring_remaining_bn >= -1e-9
)


# ============================================================
# COFINANCING
# ============================================================

cofin_specs = [
    ("AEC", aec_eu_m, aec_nat_share),
    ("ANC", anc_eu_m, anc_nat_share),
    ("Beruházás", investment_eu_m, investment_nat_share),
    ("Kockázatkezelés", risk_eu_m, risk_nat_share),
    ("Fiatal gazda induló", young_startup_eu_m, young_startup_nat_share),
]

cofin_rows = []

for name, eu_m, hu_share in cofin_specs:
    hu_m = national_from_eu(eu_m, hu_share)
    total_m = eu_m + hu_m

    cofin_rows.append(
        {
            "Intézkedés": name,
            "EU-rész (M€/év)": eu_m,
            "Magyar arány": hu_share,
            "Magyar rész (M€/év)": hu_m,
            "Teljes közkiadás (M€/év)": total_m,
        }
    )

cofin_table = pd.DataFrame(cofin_rows)

hu_cofin_annual_m = cofin_table["Magyar rész (M€/év)"].sum()
hu_cofin_7y_bn = hu_cofin_annual_m * 7 / 1_000


# ============================================================
# GREEN TRACKING
# ============================================================

green_rows = [
    ("DABIS", dabis_paid_m, TRACKING_DEFAULTS["DABIS"]),
    ("YF top-up", young_budget_m, TRACKING_DEFAULTS["YF top-up"]),
    ("SFS", sfs_budget_m, TRACKING_DEFAULTS["SFS"]),
    ("AEC", aec_eu_m, TRACKING_DEFAULTS["AEC"]),
    ("ANC", anc_eu_m, anc_tracking),
    ("CIS", cis_budget_m, cis_tracking),
    ("Beruházás", investment_eu_m, green_investment_share),
    ("Kockázatkezelés", risk_eu_m, TRACKING_DEFAULTS["Risk management"]),
    (
        "Fiatal gazda induló",
        young_startup_eu_m,
        TRACKING_DEFAULTS["Young farmer startup"],
    ),
]

green_table = pd.DataFrame(
    green_rows,
    columns=[
        "Intézkedés",
        "EU kiadás (M€/év)",
        "Tracking",
    ],
)
green_table["Tracking hozzájárulás (M€/év)"] = (
    green_table["EU kiadás (M€/év)"] * green_table["Tracking"]
)
green_table["Tracking hozzájárulás (mrd €/7 év)"] = (
    green_table["Tracking hozzájárulás (M€/év)"] * 7 / 1_000
)

cap_green_7y_bn = green_table["Tracking hozzájárulás (mrd €/7 év)"].sum()

nrpp_green_required_bn = nrpp_total_bn * nrpp_green_target
nrpp_green_total_bn = cap_green_7y_bn + other_nrpp_green_bn
nrpp_green_gap_bn = nrpp_green_required_bn - nrpp_green_total_bn


# ============================================================
# DATA OVERVIEW
# ============================================================

st.subheader("2024-es input adat")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Támogatásigénylők",
    f"{n_farms:,}".replace(",", " "),
)
m2.metric(
    "BISS/CRISS terület",
    f"{total_area_ha / 1_000_000:.2f} M ha",
)
m3.metric(
    "YFS jogosult terület",
    f"{yfs_area_ha / 1_000:.1f} ezer ha",
)
m4.metric(
    "2024 BISS+CRISS+YFS referencia",
    eur_m(baseline_subs_m),
)

with st.expander("Input struktúra / diagnosztika"):
    diagnostics = pd.DataFrame(
        {
            "Mutató": [
                "area_biss_criss",
                "area_yfs_cur_eligible",
                "area_aop",
                "area_vp_akg_2021",
            ],
            "Összeg (ha)": [
                data["area_biss_criss"].sum(),
                data["area_yfs_cur_eligible"].sum(),
                data["area_aop"].sum(),
                data["area_vp_akg_2021"].sum(),
            ],
        }
    )

    st.dataframe(
        diagnostics.style.format({"Összeg (ha)": "{:,.0f}"}),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "`area_aop` és `area_vp_akg_2021` itt referencia/proxy. "
        "A dashboard nem feltételezi automatikusan, hogy ezek egy az egyben "
        "a 2028–2034-es AEC jogosultsági területei."
    )


# ============================================================
# CAP MODEL
# ============================================================

st.divider()
st.subheader("CAP modell – valós üzemi adatokon")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("DABIS", eur_m(dabis_paid_m))
k2.metric("Degressziós elvonás", eur_m(dabis_reduction_m))
k3.metric("YF top-up", eur_m(young_budget_m))
k4.metric("SFS", eur_m(sfs_budget_m))
k5.metric("CIS", eur_m(cis_budget_m))

st.caption(
    f"SFS-ben {sfs_farms:,} gazdaság van a modellezett "
    f"≤ {sfs_max_area:.0f} ha küszöb mellett.".replace(",", " ")
)

cap_left, cap_right = st.columns([1.1, 1])

with cap_left:
    allocation_table = pd.DataFrame(
        {
            "Intézkedés": [
                "DABIS",
                "YF top-up",
                "SFS",
                "CIS",
                "AEC",
                "ANC",
                "Beruházás",
                "Kockázatkezelés",
                "Fiatal gazda induló",
            ],
            "M€/év": [
                dabis_paid_m,
                young_budget_m,
                sfs_budget_m,
                cis_budget_m,
                aec_eu_m,
                anc_eu_m,
                investment_eu_m,
                risk_eu_m,
                young_startup_eu_m,
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_alloc = allocation_table.sort_values("M€/év")
    ax.barh(
        plot_alloc["Intézkedés"],
        plot_alloc["M€/év"],
    )
    ax.set_xlabel("EU-forrás (M€/év)")
    ax.set_title("Modellezett CAP-allokáció")
    ax.grid(axis="x", alpha=0.2)

    st.pyplot(fig, use_container_width=True)

with cap_right:
    st.metric(
        "CAP EU-forrás / év",
        eur_m(cap_annual_eu_m),
    )
    st.metric(
        "CAP EU-forrás / 2028–2034",
        eur_bn(cap_7y_eu_bn),
    )
    st.metric(
        "CAP ring-fence minimum",
        eur_bn(cap_min_bn),
        delta=eur_bn(cap_7y_eu_bn - cap_min_bn),
    )

    st.markdown("**CIS plafon**")
    st.latex(
        r"""
        CIS_{\max}
        =
        \alpha(DABIS + AEC + SFS)
        """
    )

    if cis_headroom_m >= 0:
        st.success(f"CIS a plafon alatt: még {eur_m(cis_headroom_m)} mozgástér.")
    else:
        st.error(f"CIS-plafon túllépése: {eur_m(abs(cis_headroom_m))}.")

with st.expander("CIS részletek", expanded=False):
    st.dataframe(
        cis_table.style.format(
            {
                "Jogosult mennyiség": "{:,.1f}",
                "Egységösszeg (€)": "{:,.2f}",
                "Boríték (M€)": "{:,.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# RING-FENCED / NON-RING-FENCED
# ============================================================

st.divider()
st.subheader("NRPP – ring-fenced / non-ring-fenced logika")

r1, r2, r3, r4, r5 = st.columns(5)

r1.metric(
    "NRPP összesen",
    eur_bn(nrpp_total_bn),
)
r2.metric(
    "General allocation",
    eur_bn(general_allocation_bn),
)
r3.metric(
    "Kezdeti non-ring-fenced",
    eur_bn(initial_non_ring_bn),
)
r4.metric(
    "CAP extra a minimum felett",
    eur_bn(cap_extra_bn),
)
r5.metric(
    "Szabad maradvány",
    eur_bn(non_ring_remaining_bn),
)

ring_table = pd.DataFrame(
    {
        "Blokk": [
            "CAP",
            "Kevésbé fejlett régiók",
            "Migration / Security",
            "Social Climate Fund",
            "Non-ring-fenced kiinduló",
        ],
        "Minimum / earmark (mrd €)": [
            cap_min_bn,
            ldr_min_bn,
            migration_bn,
            scf_bn,
            initial_non_ring_bn,
        ],
        "Modellezett / tervezett (mrd €)": [
            cap_7y_eu_bn,
            ldr_planned_bn,
            migration_bn,
            scf_bn,
            non_ring_remaining_bn,
        ],
    }
)

st.dataframe(
    ring_table.style.format(
        {
            "Minimum / earmark (mrd €)": "{:,.2f}",
            "Modellezett / tervezett (mrd €)": "{:,.2f}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.latex(
    r"""
    B_{general}
    =
    B_{NRPP}
    -
    B_{migration}
    -
    B_{SCF}
    """
)

st.latex(
    r"""
    B_{NRF,0}
    =
    B_{general}
    -
    B_{CAP,min}
    -
    B_{LDR,min}
    """
)

st.latex(
    r"""
    B_{NRF,remaining}
    =
    B_{NRF,0}
    -
    (CAP-CAP_{min})_+
    -
    (LDR-LDR_{min})_+
    -
    Other_{flex}
    """
)

if cap_gap_bn > 0:
    st.error(
        f"A modellezett CAP {eur_bn(cap_gap_bn)} összeggel "
        "a ring-fenced minimum alatt van."
    )

if ldr_gap_bn > 0:
    st.error(
        f"A tervezett LDR-keret {eur_bn(ldr_gap_bn)} összeggel a minimum alatt van."
    )

if non_ring_remaining_bn < 0:
    st.error(
        f"A non-ring-fenced keretet {eur_bn(abs(non_ring_remaining_bn))} "
        "összeggel túlosztottad."
    )
elif ring_valid:
    st.success("A ring-fenced minimumok teljesülnek, és a szabad keret nem negatív.")

st.caption(
    "A non-ring-fenced maradvány nem azonos az NRPP-rendelet "
    "külön 25%-os mid-term 'flexibility amount' mechanizmusával."
)


# ============================================================
# COFINANCING
# ============================================================

st.divider()
st.subheader("Magyar társfinanszírozás")

cf1, cf2, cf3 = st.columns(3)

cf1.metric(
    "Magyar rész / év",
    eur_m(hu_cofin_annual_m),
)
cf2.metric(
    "Magyar rész / 7 év",
    eur_bn(hu_cofin_7y_bn),
)
cf3.metric(
    "CAP EU + modellezett HU közkiadás / 7 év",
    eur_bn(cap_7y_eu_bn + hu_cofin_7y_bn),
)

cofin_display = cofin_table.copy()
cofin_display["Magyar arány"] *= 100

st.dataframe(
    cofin_display.style.format(
        {
            "EU-rész (M€/év)": "{:,.1f}",
            "Magyar arány": "{:.0f}%",
            "Magyar rész (M€/év)": "{:,.1f}",
            "Teljes közkiadás (M€/év)": "{:,.1f}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.latex(
    r"""
    HU_j
    =
    EU_j
    \frac{s_j}{1-s_j}
    """
)


# ============================================================
# GREEN TRACKING
# ============================================================

st.divider()
st.subheader("NRPP klíma- és környezeti tracking")

g1, g2, g3, g4 = st.columns(4)

g1.metric(
    "NRPP target",
    pct(nrpp_green_target),
)
g2.metric(
    "Szükséges",
    eur_bn(nrpp_green_required_bn),
)
g3.metric(
    "CAP-ból trackelt",
    eur_bn(cap_green_7y_bn),
)
g4.metric(
    "Hiány / többlet",
    eur_bn(-nrpp_green_gap_bn),
)

green_display = green_table.copy()
green_display["Tracking"] *= 100

st.dataframe(
    green_display.style.format(
        {
            "EU kiadás (M€/év)": "{:,.1f}",
            "Tracking": "{:.0f}%",
            "Tracking hozzájárulás (M€/év)": "{:,.1f}",
            "Tracking hozzájárulás (mrd €/7 év)": "{:,.2f}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

if nrpp_green_gap_bn > 0:
    st.warning(
        f"A 43%-os célhoz még {eur_bn(nrpp_green_gap_bn)} "
        "tracking-hozzájárulás szükséges a CAP-on kívüli NRPP-elemekből "
        "és/vagy magasabb zöld intenzitásból."
    )
else:
    st.success("A beállított CAP + nem CAP tracking teljesíti a célértéket.")


# ============================================================
# DABIS DISTRIBUTION
# ============================================================

st.divider()
st.subheader("DABIS – üzemi degresszió")

plot_df = pd.DataFrame(
    {
        "gross": gross_dabis_eur,
        "paid": paid_dabis_eur,
    }
).sort_values("gross")

step = max(len(plot_df) // 2_000, 1)
plot_df = plot_df.iloc[::step]

fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(
    plot_df["gross"],
    plot_df["gross"],
    label="Degresszió nélkül",
    linewidth=1.4,
)
ax.plot(
    plot_df["gross"],
    plot_df["paid"],
    label="Degresszió után",
    linewidth=2.0,
)

for threshold in [20_000, 50_000, 75_000]:
    ax.axvline(
        threshold,
        linestyle="--",
        alpha=0.45,
    )

ax.set_xlabel("Bruttó DABIS / üzem (€)")
ax.set_ylabel("Tényleges DABIS / üzem (€)")
ax.grid(alpha=0.2)
ax.legend()

st.pyplot(fig, use_container_width=True)


# ============================================================
# EXPORT
# ============================================================

st.divider()
st.subheader("Export")

st.download_button(
    "CAP allokáció CSV",
    allocation_table.to_csv(index=False).encode("utf-8-sig"),
    "cap_allocation.csv",
    "text/csv",
)

st.download_button(
    "CIS részletek CSV",
    cis_table.to_csv(index=False).encode("utf-8-sig"),
    "cis_components.csv",
    "text/csv",
)

st.download_button(
    "Ring / non-ring CSV",
    ring_table.to_csv(index=False).encode("utf-8-sig"),
    "nrpp_ring_fencing.csv",
    "text/csv",
)

st.download_button(
    "Magyar társfinanszírozás CSV",
    cofin_table.to_csv(index=False).encode("utf-8-sig"),
    "hungarian_cofinancing.csv",
    "text/csv",
)

st.download_button(
    "Green tracking CSV",
    green_table.to_csv(index=False).encode("utf-8-sig"),
    "nrpp_green_tracking.csv",
    "text/csv",
)
