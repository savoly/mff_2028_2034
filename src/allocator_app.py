from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE / STYLE
# ============================================================

st.set_page_config(
    page_title="MFF 2028–2034 | CAP / NRPP dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        min-width: 340px;
        max-width: 340px;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    [data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.055);
        border: 1px solid rgba(127, 127, 127, 0.16);
        border-radius: 14px;
        padding: 14px 16px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border-color: rgba(127, 127, 127, 0.18);
    }

    .hero {
        padding: 1.2rem 1.35rem;
        border: 1px solid rgba(127,127,127,.18);
        border-radius: 16px;
        background: rgba(127,127,127,.045);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0 0 .25rem 0;
        font-size: 2rem;
        line-height: 1.15;
    }

    .hero p {
        margin: 0;
        opacity: .75;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .08em;
        font-size: .72rem;
        font-weight: 700;
        opacity: .62;
        margin-bottom: .35rem;
    }

    .section-note {
        opacity: .72;
        font-size: .9rem;
        margin-top: -.3rem;
        margin-bottom: 1rem;
    }

    .status-good {
        padding: .75rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(60,160,90,.30);
        background: rgba(60,160,90,.08);
    }

    .status-warn {
        padding: .75rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(210,160,40,.30);
        background: rgba(210,160,40,.08);
    }

    .status-bad {
        padding: .75rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(200,70,70,.30);
        background: rgba(200,70,70,.08);
    }

    .small-muted {
        opacity: .65;
        font-size: .82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">MFF 2028–2034 · Magyarország</div>
        <h1>CAP / NRPP allokációs dashboard</h1>
        <p>
            Valós 2024-es EK-adatokon: DABIS, YF, SFS, CIS,
            ring-fenced / non-ring-fenced keretek, társfinanszírozás
            és klíma-/környezeti tracking.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
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

# 2024-es CIS referencia a read_extended_base_data(2024) állományhoz.
# A `rate` értékek a megadott tk_* fajlagosok; ezek töltődnek be
# alapértelmezett egységösszegként a felületen.
CIS_REFERENCE = {
    "Cukorrépa": {
        "rate": 490.56,
        "budget": 7_915_693.20,
        "min": 424.52,
        "max": 881.70,
    },
    "Szemes fehérjenövény": {
        "rate": 130.02,
        "budget": 15_962_772.76,
        "min": 163.01,
        "max": 302.73,
    },
    "Szálas fehérjenövény": {
        "rate": 66.34,
        "budget": 10_985_262.30,
        "min": 44.92,
        "max": 77.00,
    },
    "Extenzív gyümölcs": {
        "rate": 199.75,
        "budget": 6_063_331.33,
        "min": 143.54,
        "max": 191.39,
    },
    "Intenzív gyümölcs": {
        "rate": 378.81,
        "budget": 6_653_919.68,
        "min": 280.11,
        "max": 429.50,
    },
    "Ipari olajnövény": {
        "rate": 121.58,
        "budget": 740_499.63,
        "min": 76.71,
        "max": 142.47,
    },
    "Ipari zöldségnövény": {
        "rate": 384.99,
        "budget": 15_619_889.00,
        "min": 260.44,
        "max": 390.66,
    },
    "Zöldségnövény": {
        "rate": 135.63,
        "budget": 6_455_314.80,
        "min": 180.92,
        "max": 289.48,
    },
    "Rizs": {
        "rate": 840.68,
        "budget": 1_979_171.48,
        "min": 582.75,
        "max": 874.13,
    },
    "Hízott bika": {
        "rate": 95.96,
        "budget": 5_061_180.70,
        "min": 40.76,
        "max": 65.22,
    },
    "Anyatehén": {
        "rate": 168.12,
        "budget": 34_631_327.25,
        "min": 111.96,
        "max": 174.94,
    },
    "Tejhasznú tehén": {
        "rate": 168.12,
        "budget": 68_274_257.17,
        "min": 237.74,
        "max": 396.24,
    },
    "Anyajuh": {
        "rate": 35.25,
        "budget": 21_767_730.60,
        "min": 24.16,
        "max": 35.53,
    },
}

CIS_CURRENT_EUR = 202_110_350.0
CIS_CURRENT_M = CIS_CURRENT_EUR / 1_000_000


# 2028–2034 baseline defaultok a feltöltött
# MFF_2028-2034_Agrárköltségvetési_modell_2026-03-18_ver_01.xlsx alapján.
# DABIS és CIS külön, a 2024-es üzemi / fajlagos logikából számolódik.
# Az AEC egyetlen app-mezőben összevonja az AÖP 2028-as EU-részét
# és a jelenlegi környezeti/klíma kötelezettségvállalások éves EU-átlagát.
MODEL_DEFAULTS_M = {
    "aec": 326.200216684,
    "anc": 0.0,
    "mandatory_req": 20.850387300,
    "investment": 52.498636246,
    "risk": 1.310862310,
    "young_startup": 18.194521428,
    "farm_relief": 0.0,
    "leader": 16.461056032,
    # Tudás/innováció = jelenlegi tudásmegosztás + EIP együttműködés.
    "knowledge": 6.795749884,
    # Területi/helyi együttműködés = együttműködések LEADER és EIP nélkül.
    "territorial_coop": 3.017966218,
    # 2028–2034, szcenárió 1. éves átlag.
    "school_scheme": 5.965786000,
    "sectoral_support": 46.44781983947153,
    "crisis_payments": 0.0,
}

# A márciusi baseline társfinanszírozási feltételei.
# Az AEC összevont értéke miatt itt súlyozott arányt használunk
# (AÖP: 35%; AEK-II környezeti rész: 80%).
COFIN_DEFAULTS = {
    "aec": 0.72,
    "anc": 0.80,
    "investment": 0.80,
    "risk": 0.80,
    "young_startup": 0.80,
}

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


def signed_bn(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.2f} mrd €".replace(".", ",")


def status_box(text: str, level: str = "good") -> None:
    css_class = {
        "good": "status-good",
        "warn": "status-warn",
        "bad": "status-bad",
    }[level]
    st.markdown(
        f'<div class="{css_class}">{text}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_extended_data(base_year: int) -> pd.DataFrame:
    path = DATA_DIR / f"data_extended_{base_year}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Nem találom az inputot: {path}. "
            "Futtasd előbb a generate_extended_base_data() függvényt."
        )

    data = pd.read_parquet(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Hiányzó oszlopok: " + ", ".join(missing))

    data = data.copy()
    numeric_cols = [column for column in REQUIRED_COLUMNS if column != "regszam"]
    data[numeric_cols] = data[numeric_cols].fillna(0)

    return data


def dabis_after_degression(x: np.ndarray) -> np.ndarray:
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
    if not 0 <= national_share < 1:
        raise ValueError("A nemzeti aránynak [0, 1) intervallumban kell lennie.")

    return eu_amount_m * national_share / (1.0 - national_share)


# ============================================================
# SIDEBAR – DATA
# ============================================================

st.sidebar.markdown("## Beállítások")
st.sidebar.caption("A ritkábban módosított paraméterek lenyitható blokkokban vannak.")

with st.sidebar.expander("📦 Adatforrás", expanded=True):
    base_year = st.selectbox(
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

with st.sidebar.expander("🌾 CAP design", expanded=True):
    dabis_rate = st.slider(
        "DABIS fajlagos (€/ha)",
        min_value=130,
        max_value=240,
        value=220,
        step=5,
    )

    apply_degression = st.checkbox(
        "Degresszió alkalmazása",
        value=True,
    )

    young_topup = st.slider(
        "Fiatal gazda top-up (€/ha)",
        min_value=0,
        max_value=150,
        value=50,
        step=5,
    )

    sfs_max_area = st.slider(
        "SFS felső területhatár (ha)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    )

    sfs_flat = st.slider(
        "SFS átalány (€/gazdaság/év)",
        min_value=0,
        max_value=3_000,
        value=0,
        step=100,
    )

with st.sidebar.expander("💶 Egyéb CAP borítékok", expanded=False):
    st.caption(
        "Default: 2026-03-18-as agrárköltségvetési modell baseline; "
        "az AEC az AÖP + környezeti/klíma kötelezettségvállalások összevont EU-kerete."
    )
    aec_eu_m = st.number_input(
        "AEC / környezet-klíma (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["aec"]),
        step=1.0,
        format="%.2f",
    )
    anc_eu_m = st.number_input(
        "ANC (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["anc"]),
        step=1.0,
        format="%.2f",
    )
    investment_eu_m = st.number_input(
        "Beruházás (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["investment"]),
        step=1.0,
        format="%.2f",
    )
    risk_eu_m = st.number_input(
        "Kockázatkezelés (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["risk"]),
        step=1.0,
        format="%.2f",
    )
    young_startup_eu_m = st.number_input(
        "Fiatal / új gazda, vidéki vállalkozás induló (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["young_startup"]),
        step=1.0,
        format="%.2f",
    )

with st.sidebar.expander("📚 További CAP-beavatkozások", expanded=False):
    st.caption(
        "A COM(2025) 565 Magyarországon alkalmazható Article 35(1) beavatkozásai "
        "+ Article 38 válságkifizetés."
    )

    mandatory_req_eu_m = st.number_input(
        "35(1)(e) Kötelező előírásokból eredő hátrányok (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["mandatory_req"]),
        step=1.0,
        format="%.2f",
    )
    farm_relief_eu_m = st.number_input(
        "35(1)(k) Gazdasághelyettesítési szolgáltatások (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["farm_relief"]),
        step=1.0,
        format="%.2f",
    )
    leader_eu_m = st.number_input(
        "35(1)(l) LEADER (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["leader"]),
        step=1.0,
        format="%.2f",
    )
    knowledge_eu_m = st.number_input(
        "35(1)(m) Tudásmegosztás és innováció (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["knowledge"]),
        step=1.0,
        format="%.2f",
    )
    territorial_coop_eu_m = st.number_input(
        "35(1)(n) Területi és helyi együttműködés (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["territorial_coop"]),
        step=1.0,
        format="%.2f",
    )

    school_scheme_eu_m = st.number_input(
        "35(1)(q) EU iskolaprogram (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["school_scheme"]),
        step=1.0,
        format="%.2f",
    )
    sectoral_support_eu_m = st.number_input(
        "35(1)(r) Egyes ágazatok támogatása (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["sectoral_support"]),
        step=1.0,
        format="%.2f",
    )
    crisis_payments_eu_m = st.number_input(
        "Art. 38 Válságkifizetés gazdálkodóknak (M€/év)",
        min_value=0.0,
        value=float(MODEL_DEFAULTS_M["crisis_payments"]),
        step=1.0,
        format="%.2f",
        help="Természeti katasztrófa, kedvezőtlen éghajlati vagy katasztrofális esemény esetén.",
    )


# ============================================================
# SIDEBAR – CIS
# ============================================================

with st.sidebar.expander("🔗 CIS", expanded=False):
    cis_alpha = st.select_slider(
        "CIS plafon",
        options=[0.20, 0.25],
        value=0.20,
        format_func=lambda value: f"{value:.0%}",
    )

    st.caption(
        f"2024-es CIS referencia: {eur_m(CIS_CURRENT_M)}. "
        "Az egységösszegek alapértékei a megadott tk-fajlagosok."
    )

    cis_rates = {}

    for label, (_, unit) in CIS_COMPONENTS.items():
        ref = CIS_REFERENCE[label]
        cis_rates[label] = st.number_input(
            f"{label} (€/{unit})",
            min_value=0.0,
            value=float(ref["rate"]),
            step=1.0,
            format="%.2f",
            key=f"cis_rate_{label}",
            help=(
                f"2024 referenciafajlagos: {ref['rate']:.2f} €/{unit}; "
                f"referencia-boríték: {ref['budget'] / 1_000_000:.2f} M€; "
                f"megadott min–max: {ref['min']:.2f}–{ref['max']:.2f} €/{unit}."
            ),
        )


# ============================================================
# SIDEBAR – NRPP / RING-FENCING
# ============================================================

with st.sidebar.expander("🧩 NRPP / ring-fencing", expanded=True):
    nrpp_total_bn = st.number_input(
        "NRPP összesen (mrd €)",
        min_value=1.0,
        value=37.70,
        step=0.10,
    )
    migration_bn = st.number_input(
        "Migration / Security (mrd €)",
        min_value=0.0,
        value=0.50,
        step=0.05,
    )
    scf_bn = st.number_input(
        "Social Climate Fund (mrd €)",
        min_value=0.0,
        value=2.20,
        step=0.05,
    )
    cap_min_bn = st.number_input(
        "CAP minimum (mrd €)",
        min_value=0.0,
        value=9.24,
        step=0.05,
    )
    ldr_min_bn = st.number_input(
        "Kevésbé fejlett régiók minimuma (mrd €)",
        min_value=0.0,
        value=20.71269,
        step=0.05,
    )
    ldr_planned_bn = st.number_input(
        "Kevésbé fejlett régiók tervezett kerete (mrd €)",
        min_value=0.0,
        value=float(ldr_min_bn),
        step=0.05,
    )
    other_flexible_use_bn = st.number_input(
        "Egyéb felhasznált szabad keret (mrd €)",
        min_value=0.0,
        value=0.0,
        step=0.05,
    )


# ============================================================
# SIDEBAR – TRACKING / COFIN
# ============================================================

with st.sidebar.expander("🌱 Zöld tracking", expanded=False):
    nrpp_green_target = st.slider(
        "NRPP cél",
        min_value=0.30,
        max_value=0.60,
        value=0.43,
        step=0.01,
        format="%.2f",
    )
    cis_tracking = st.slider(
        "CIS átlagos tracking",
        min_value=0.0,
        max_value=1.0,
        value=TRACKING_DEFAULTS["CIS"],
        step=0.05,
        format="%.2f",
    )
    anc_tracking = st.slider(
        "ANC átlagos tracking",
        min_value=0.0,
        max_value=1.0,
        value=TRACKING_DEFAULTS["ANC"],
        step=0.05,
        format="%.2f",
    )
    green_investment_share = st.slider(
        "Beruházásból green investment",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        format="%.2f",
    )
    other_nrpp_green_bn = st.number_input(
        "Nem CAP NRPP zöld tracking (mrd €/7 év)",
        min_value=0.0,
        value=0.0,
        step=0.1,
    )

with st.sidebar.expander("🇭🇺 Magyar társfinanszírozás", expanded=False):
    st.caption(
        "Baseline default: AÖP 35%, a többi modellezett VP-tétel 80%; "
        "az összevont AEC-hez súlyozott ~72%-os magyar rész tartozik."
    )

    aec_nat_share = st.slider(
        "AEC magyar rész",
        0.0,
        0.80,
        COFIN_DEFAULTS["aec"],
        0.01,
        format="%.2f",
    )
    anc_nat_share = st.slider(
        "ANC magyar rész",
        0.0,
        0.80,
        COFIN_DEFAULTS["anc"],
        0.05,
        format="%.2f",
    )
    investment_nat_share = st.slider(
        "Beruházás magyar rész",
        0.0,
        0.80,
        COFIN_DEFAULTS["investment"],
        0.05,
        format="%.2f",
    )
    risk_nat_share = st.slider(
        "Kockázatkezelés magyar rész",
        0.0,
        0.80,
        COFIN_DEFAULTS["risk"],
        0.05,
        format="%.2f",
    )
    young_startup_nat_share = st.slider(
        "Fiatal gazda induló magyar rész",
        0.0,
        0.80,
        COFIN_DEFAULTS["young_startup"],
        0.05,
        format="%.2f",
    )


# ============================================================
# CALCULATIONS – CAP
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

    ref = CIS_REFERENCE[label]

    cis_rows.append(
        {
            "CIS komponens": label,
            "Jogosult mennyiség": quantity,
            "Egység": unit,
            "Egységösszeg (€)": rate,
            "2024 referencia (€)": ref["rate"],
            "Referencia-boríték (M€)": ref["budget"] / 1_000_000,
            "Boríték (M€)": amount_m,
        }
    )

cis_table = pd.DataFrame(cis_rows)
cis_budget_m = cis_table["Boríték (M€)"].sum()
cis_reference_delta_m = cis_budget_m - CIS_CURRENT_M

cis_reference_m = dabis_paid_m + young_budget_m + aec_eu_m + sfs_budget_m
cis_max_m = cis_alpha * cis_reference_m
cis_headroom_m = cis_max_m - cis_budget_m

# Article 35(1)(a)–(k) és (r): ezek számítanak a CAP income-support minimumhoz.
# A jelenlegi YF top-up a DABIS egyik modellezett alkomponenseként szerepel.
cap_income_support_annual_eu_m = (
    dabis_paid_m
    + young_budget_m
    + cis_budget_m
    + anc_eu_m
    + mandatory_req_eu_m
    + aec_eu_m
    + sfs_budget_m
    + risk_eu_m
    + investment_eu_m
    + young_startup_eu_m
    + farm_relief_eu_m
    + sectoral_support_eu_m
)

# Article 35(1)(l)–(q) + Article 38: CAP, de nem része az Annex XVIII
# szerinti ring-fenced CAP income-support minimumnak.
cap_non_ring_annual_eu_m = (
    leader_eu_m
    + knowledge_eu_m
    + territorial_coop_eu_m
    + school_scheme_eu_m
    + crisis_payments_eu_m
)

cap_annual_eu_m = cap_income_support_annual_eu_m + cap_non_ring_annual_eu_m

cap_income_support_7y_bn = cap_income_support_annual_eu_m * 7 / 1_000
cap_non_ring_7y_bn = cap_non_ring_annual_eu_m * 7 / 1_000
cap_7y_eu_bn = cap_annual_eu_m * 7 / 1_000


# ============================================================
# CALCULATIONS – RING-FENCING
# ============================================================

general_allocation_bn = nrpp_total_bn - migration_bn - scf_bn

initial_non_ring_bn = general_allocation_bn - cap_min_bn - ldr_min_bn

cap_gap_bn = max(cap_min_bn - cap_income_support_7y_bn, 0.0)
cap_extra_bn = max(cap_income_support_7y_bn - cap_min_bn, 0.0)

ldr_gap_bn = max(ldr_min_bn - ldr_planned_bn, 0.0)
ldr_extra_bn = max(ldr_planned_bn - ldr_min_bn, 0.0)

non_ring_remaining_bn = (
    initial_non_ring_bn
    - cap_extra_bn
    - cap_non_ring_7y_bn
    - ldr_extra_bn
    - other_flexible_use_bn
)

ring_valid = (
    cap_gap_bn <= 1e-9
    and ldr_gap_bn <= 1e-9
    and initial_non_ring_bn >= -1e-9
    and non_ring_remaining_bn >= -1e-9
)


# ============================================================
# CALCULATIONS – COFINANCING
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
# CALCULATIONS – GREEN TRACKING
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
# SHARED TABLES
# ============================================================

allocation_table = pd.DataFrame(
    [
        {
            "Cikk": "35(1)(a)",
            "Rövid név": "DABIS",
            "Intézkedés": "Degresszív területalapú jövedelemtámogatás",
            "M€/év": dabis_paid_m + young_budget_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(b)",
            "Rövid név": "CIS",
            "Intézkedés": "Termeléshez kötött jövedelemtámogatás",
            "M€/év": cis_budget_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(d)",
            "Rövid név": "ANC",
            "Intézkedés": "Természeti és egyéb területspecifikus hátrányok",
            "M€/év": anc_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(e)",
            "Rövid név": "Kötelező hátrány",
            "Intézkedés": "Kötelező előírásokból eredő hátrányok",
            "M€/év": mandatory_req_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(f)",
            "Rövid név": "AEC",
            "Intézkedés": "Agrár-környezetvédelmi és éghajlat-politikai intézkedések",
            "M€/év": aec_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(g)",
            "Rövid név": "SFS",
            "Intézkedés": "Kisgazdaságok támogatása",
            "M€/év": sfs_budget_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(h)",
            "Rövid név": "Kockázat",
            "Intézkedés": "Kockázatkezelési eszközök",
            "M€/év": risk_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(i)",
            "Rövid név": "Beruházás",
            "Intézkedés": "Beruházások gazdálkodóknak és erdőgazdálkodóknak",
            "M€/év": investment_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(j)",
            "Rövid név": "Induló",
            "Intézkedés": "Fiatal/új gazdák, vidéki vállalkozások és kisgazdaságok indulása/fejlesztése",
            "M€/év": young_startup_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(k)",
            "Rövid név": "Helyettesítés",
            "Intézkedés": "Gazdasághelyettesítési szolgáltatások",
            "M€/év": farm_relief_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "35(1)(l)",
            "Rövid név": "LEADER",
            "Intézkedés": "LEADER",
            "M€/év": leader_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "nem",
        },
        {
            "Cikk": "35(1)(m)",
            "Rövid név": "Tudás/innováció",
            "Intézkedés": "Tudásmegosztás és innováció a mezőgazdaságban, erdészetben és vidéken",
            "M€/év": knowledge_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "nem",
        },
        {
            "Cikk": "35(1)(n)",
            "Rövid név": "Területi együttm.",
            "Intézkedés": "Területi és helyi együttműködési kezdeményezések",
            "M€/év": territorial_coop_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "nem",
        },
        {
            "Cikk": "35(1)(q)",
            "Rövid név": "Iskolaprogram",
            "Intézkedés": "EU iskolaprogram",
            "M€/év": school_scheme_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "nem",
        },
        {
            "Cikk": "35(1)(r)",
            "Rövid név": "Ágazati",
            "Intézkedés": "Egyes ágazatok támogatása",
            "M€/év": sectoral_support_eu_m,
            "HU státusz": "✓ alkalmazható",
            "CAP minimumba": "igen",
        },
        {
            "Cikk": "Art. 38",
            "Rövid név": "Válságkifizetés",
            "Intézkedés": "Válságkifizetés gazdálkodóknak",
            "M€/év": crisis_payments_eu_m,
            "HU státusz": "✓ feltételes",
            "CAP minimumba": "nem",
        },
    ]
)

ring_table = pd.DataFrame(
    {
        "Blokk": [
            "CAP income support minimum",
            "CAP egyéb / non-ring",
            "Kevésbé fejlett régiók",
            "Migration / Security",
            "Social Climate Fund",
            "Non-ring-fenced kiinduló",
        ],
        "Minimum / earmark (mrd €)": [
            cap_min_bn,
            0.0,
            ldr_min_bn,
            migration_bn,
            scf_bn,
            initial_non_ring_bn,
        ],
        "Modellezett / tervezett (mrd €)": [
            cap_income_support_7y_bn,
            cap_non_ring_7y_bn,
            ldr_planned_bn,
            migration_bn,
            scf_bn,
            non_ring_remaining_bn,
        ],
    }
)


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

top1, top2, top3, top4, top5 = st.columns(5)

top1.metric(
    "CAP összesen 2028–2034",
    eur_bn(cap_7y_eu_bn),
    delta=f"income support: {signed_bn(cap_income_support_7y_bn - cap_min_bn)}",
)

top2.metric(
    "Szabad NRPP maradvány",
    eur_bn(non_ring_remaining_bn),
)

top3.metric(
    "Magyar társfinanszírozás",
    eur_bn(hu_cofin_7y_bn),
)

top4.metric(
    "Zöld tracking",
    pct(nrpp_green_total_bn / nrpp_total_bn),
    delta=pct((nrpp_green_total_bn - nrpp_green_required_bn) / nrpp_total_bn),
)

top5.metric(
    "CIS kihasználtság",
    pct(cis_budget_m / cis_max_m) if cis_max_m > 0 else "—",
)

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if cap_gap_bn > 0:
        status_box(
            f"<b>CAP minimum:</b> hiányzik {eur_bn(cap_gap_bn)}.",
            "bad",
        )
    else:
        status_box(
            f"<b>CAP minimum:</b> teljesül, többlet {eur_bn(cap_extra_bn)}.",
            "good",
        )

with status_col2:
    if non_ring_remaining_bn < 0:
        status_box(
            f"<b>Szabad keret:</b> túlosztás {eur_bn(abs(non_ring_remaining_bn))}.",
            "bad",
        )
    elif non_ring_remaining_bn < 0.5:
        status_box(
            f"<b>Szabad keret:</b> csak {eur_bn(non_ring_remaining_bn)} maradt.",
            "warn",
        )
    else:
        status_box(
            f"<b>Szabad keret:</b> {eur_bn(non_ring_remaining_bn)} maradt.",
            "good",
        )

with status_col3:
    if nrpp_green_gap_bn > 0:
        status_box(
            f"<b>Zöld cél:</b> hiányzik {eur_bn(nrpp_green_gap_bn)} tracking.",
            "warn",
        )
    else:
        status_box(
            f"<b>Zöld cél:</b> teljesül, többlet {eur_bn(abs(nrpp_green_gap_bn))}.",
            "good",
        )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_cap, tab_ring, tab_cofin, tab_green, tab_data = st.tabs(
    [
        "📊 Vezetői összkép",
        "🌾 CAP modell",
        "🧩 Ring-fence",
        "🇭🇺 Társfinanszírozás",
        "🌱 Zöld tracking",
        "🗂 Adat & CIS",
    ]
)


# ============================================================
# TAB – OVERVIEW
# ============================================================

with tab_overview:
    st.subheader("Vezetői összkép")
    st.markdown(
        '<div class="section-note">'
        "A legfontosabb keretek és a modellezett CAP allokáció egy nézetben."
        "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1])

    with left:
        plot_alloc = allocation_table.sort_values("M€/év")

        fig, ax = plt.subplots(figsize=(10, 8.2))
        ax.barh(
            plot_alloc["Rövid név"],
            plot_alloc["M€/év"],
        )
        ax.set_xlabel("EU-forrás (M€/év)")
        ax.set_title("Modellezett CAP-allokáció")
        ax.grid(axis="x", alpha=0.2)

        st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown("#### Keretlogika")

        overview_table = pd.DataFrame(
            {
                "Mutató": [
                    "NRPP összesen",
                    "General allocation",
                    "CAP income support minimum",
                    "Modellezett CAP income support",
                    "Modellezett CAP összesen",
                    "LDR minimum",
                    "Kezdeti non-ring-fenced",
                    "Szabad maradvány",
                ],
                "mrd €": [
                    nrpp_total_bn,
                    general_allocation_bn,
                    cap_min_bn,
                    cap_income_support_7y_bn,
                    cap_7y_eu_bn,
                    ldr_min_bn,
                    initial_non_ring_bn,
                    non_ring_remaining_bn,
                ],
            }
        )

        st.dataframe(
            overview_table.style.format({"mrd €": "{:,.2f}"}),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("#### 2024-es adatbázis")
        a1, a2 = st.columns(2)
        a1.metric(
            "Támogatásigénylők",
            f"{n_farms:,}".replace(",", " "),
        )
        a2.metric(
            "BISS/CRISS terület",
            f"{total_area_ha / 1_000_000:.2f} M ha",
        )

        a3, a4 = st.columns(2)
        a3.metric(
            "YFS jogosult terület",
            f"{yfs_area_ha / 1_000:.1f} ezer ha",
        )
        a4.metric(
            "2024 referencia",
            eur_m(baseline_subs_m),
        )


# ============================================================
# TAB – CAP
# ============================================================

with tab_cap:
    st.subheader("CAP modell – valós üzemi adatokon")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("DABIS", eur_m(dabis_paid_m))
    c2.metric("Elvonás", eur_m(dabis_reduction_m))
    c3.metric("YF top-up", eur_m(young_budget_m))
    c4.metric("SFS", eur_m(sfs_budget_m))
    c5.metric(
        "CIS",
        eur_m(cis_budget_m),
        delta=f"2024 ref.: {eur_m(CIS_CURRENT_M)}",
        delta_color="off",
    )

    st.caption(
        f"SFS: {sfs_farms:,} gazdaság a ≤ {sfs_max_area:.0f} ha "
        "modellezett küszöb mellett.".replace(",", " ")
    )

    cap_left, cap_right = st.columns([1.2, 1])

    with cap_left:
        st.markdown("#### Éves CAP-allokáció – HU-ban alkalmazható beavatkozások")
        st.dataframe(
            allocation_table.style.format({"M€/év": "{:,.1f}"}),
            hide_index=True,
            use_container_width=True,
        )

    with cap_right:
        st.markdown("#### CAP minimum")
        st.metric(
            "CAP income support / 2028–2034",
            eur_bn(cap_income_support_7y_bn),
            delta=signed_bn(cap_income_support_7y_bn - cap_min_bn),
        )
        st.caption(f"Teljes modellezett CAP: {eur_bn(cap_7y_eu_bn)}")

        st.metric(
            "CIS plafon",
            eur_m(cis_max_m),
            delta=eur_m(cis_headroom_m),
        )

        if cis_headroom_m >= 0:
            status_box(
                f"A CIS a plafon alatt van. "
                f"Maradék mozgástér: <b>{eur_m(cis_headroom_m)}</b>.",
                "good",
            )
        else:
            status_box(
                f"A CIS túllépi a plafont "
                f"<b>{eur_m(abs(cis_headroom_m))}</b> összeggel.",
                "bad",
            )

    st.markdown("#### DABIS degresszió")

    plot_df = pd.DataFrame(
        {
            "gross": gross_dabis_eur,
            "paid": paid_dabis_eur,
        }
    ).sort_values("gross")

    step = max(len(plot_df) // 2_000, 1)
    plot_df = plot_df.iloc[::step]

    fig, ax = plt.subplots(figsize=(12, 4.8))
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
# TAB – RING FENCE
# ============================================================

with tab_ring:
    st.subheader("Ring-fenced / non-ring-fenced keret")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric("General allocation", eur_bn(general_allocation_bn))
    r2.metric("Kezdeti szabad keret", eur_bn(initial_non_ring_bn))
    r3.metric("CAP extra", eur_bn(cap_extra_bn))
    r4.metric("Maradó szabad keret", eur_bn(non_ring_remaining_bn))

    ring_left, ring_right = st.columns([1.2, 1])

    with ring_left:
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

    with ring_right:
        st.markdown("#### Értelmezés")

        if ring_valid:
            status_box(
                "A ring-fenced minimumok teljesülnek, "
                "és a non-ring-fenced keret nem negatív.",
                "good",
            )
        else:
            if cap_gap_bn > 0:
                status_box(
                    f"CAP minimumhiány: <b>{eur_bn(cap_gap_bn)}</b>.",
                    "bad",
                )
            if ldr_gap_bn > 0:
                status_box(
                    f"LDR minimumhiány: <b>{eur_bn(ldr_gap_bn)}</b>.",
                    "bad",
                )
            if non_ring_remaining_bn < 0:
                status_box(
                    f"Non-ring-fenced túlosztás: "
                    f"<b>{eur_bn(abs(non_ring_remaining_bn))}</b>.",
                    "bad",
                )

        st.markdown(
            """
            <div class="small-muted">
            A CAP income-support és az LDR itt minimumként viselkedik.
            A CAP-minimum feletti income-support, valamint a nem ring-fenced
            CAP-beavatkozások a szabad/non-ring-fenced mozgásteret fogyasztják.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Képletek"):
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
            (CAP_{income}-CAP_{min})_+
            -
            CAP_{nonring}
            -
            (LDR-LDR_{min})_+
            -
            Other_{flex}
            """
        )


# ============================================================
# TAB – COFINANCING
# ============================================================

with tab_cofin:
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
        "CAP EU + HU / 7 év",
        eur_bn(cap_7y_eu_bn + hu_cofin_7y_bn),
    )

    cf_left, cf_right = st.columns([1.25, 1])

    with cf_left:
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

    with cf_right:
        plot_cf = cofin_table.sort_values("Teljes közkiadás (M€/év)")

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.barh(
            plot_cf["Intézkedés"],
            plot_cf["EU-rész (M€/év)"],
            label="EU-rész",
        )
        ax.barh(
            plot_cf["Intézkedés"],
            plot_cf["Magyar rész (M€/év)"],
            left=plot_cf["EU-rész (M€/év)"],
            label="Magyar rész",
        )
        ax.set_xlabel("M€/év")
        ax.grid(axis="x", alpha=0.2)
        ax.legend()

        st.pyplot(fig, use_container_width=True)

    with st.expander("Társfinanszírozási képlet"):
        st.latex(
            r"""
            HU_j
            =
            EU_j
            \frac{s_j}{1-s_j}
            """
        )
        st.caption(
            "30%-os magyar rész esetén 100 M€ EU-forráshoz "
            "42,86 M€ magyar költségvetési hozzájárulás tartozik."
        )


# ============================================================
# TAB – GREEN TRACKING
# ============================================================

with tab_green:
    st.subheader("NRPP klíma- és környezeti tracking")

    current_green_share = (
        nrpp_green_total_bn / nrpp_total_bn if nrpp_total_bn > 0 else 0.0
    )

    g1, g2, g3, g4 = st.columns(4)

    g1.metric("Cél", pct(nrpp_green_target))
    g2.metric("Jelenlegi tracking", pct(current_green_share))
    g3.metric("CAP hozzájárulás", eur_bn(cap_green_7y_bn))
    g4.metric("Hiány / többlet", signed_bn(-nrpp_green_gap_bn))

    if nrpp_green_gap_bn > 0:
        status_box(
            f"A célhoz még <b>{eur_bn(nrpp_green_gap_bn)}</b> "
            "tracking-hozzájárulás szükséges.",
            "warn",
        )
    else:
        status_box(
            f"A cél teljesül. Többlet: <b>{eur_bn(abs(nrpp_green_gap_bn))}</b>.",
            "good",
        )

    st.progress(
        min(
            max(
                current_green_share / nrpp_green_target
                if nrpp_green_target > 0
                else 0.0,
                0.0,
            ),
            1.0,
        )
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


# ============================================================
# TAB – DATA / CIS / EXPORT
# ============================================================

with tab_data:
    st.subheader("Adat, CIS és export")

    data_left, data_right = st.columns([1, 1.35])

    with data_left:
        st.markdown("#### Input diagnosztika")

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
            "`area_aop` és `area_vp_akg_2021` referencia/proxy; "
            "nem automatikus 2028–2034 AEC-jogosultság."
        )

    with data_right:
        st.markdown("#### CIS komponensek")

        st.dataframe(
            cis_table.style.format(
                {
                    "Jogosult mennyiség": "{:,.1f}",
                    "Egységösszeg (€)": "{:,.2f}",
                    "2024 referencia (€)": "{:,.2f}",
                    "Referencia-boríték (M€)": "{:,.2f}",
                    "Boríték (M€)": "{:,.2f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Export")

    e1, e2, e3, e4, e5 = st.columns(5)

    with e1:
        st.download_button(
            "CAP allokáció",
            allocation_table.to_csv(index=False).encode("utf-8-sig"),
            "cap_allocation.csv",
            "text/csv",
            use_container_width=True,
        )

    with e2:
        st.download_button(
            "CIS részletek",
            cis_table.to_csv(index=False).encode("utf-8-sig"),
            "cis_components.csv",
            "text/csv",
            use_container_width=True,
        )

    with e3:
        st.download_button(
            "Ring / non-ring",
            ring_table.to_csv(index=False).encode("utf-8-sig"),
            "nrpp_ring_fencing.csv",
            "text/csv",
            use_container_width=True,
        )

    with e4:
        st.download_button(
            "Társfinanszírozás",
            cofin_table.to_csv(index=False).encode("utf-8-sig"),
            "hungarian_cofinancing.csv",
            "text/csv",
            use_container_width=True,
        )

    with e5:
        st.download_button(
            "Green tracking",
            green_table.to_csv(index=False).encode("utf-8-sig"),
            "nrpp_green_tracking.csv",
            "text/csv",
            use_container_width=True,
        )
