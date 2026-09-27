from data_tools.db import Manager

from mff.new_cap import generate_extended_base_data, query_mepar_data

engine = Manager("mvh-admin", "mvh").engine

data = generate_extended_base_data(2024, engine)
mepar = query_mepar_data(engine)

data = data.merge(mepar, how="left", on="regszam")
data["megye"] = data["megye"].fillna("Ismeretlen")

data = data.sort_values(by="area_biss_criss")

ECO_SCHEMES = 202_125_000
AKG_2021 = 376_417_351.59

coupled_payments = {
    "tk_cukorrepa": {"budget": 7_915_693.20, "min": 424.52, "max": 881.70},
    "tk_szemes_feherjenoveny": {"budget": 15_962_772.76, "min": 163.01, "max": 302.73},
    "tk_szalas_feherjenoveny": {"budget": 10_985_262.30, "min": 44.92, "max": 77.00},
    "tk_extenziv_gyumolcs": {"budget": 6_063_331.33, "min": 143.54, "max": 191.39},
    "tk_intenziv_gyumolcs": {"budget": 6_653_919.68, "min": 280.11, "max": 429.50},
    "tk_ipari_olajnoveny": {"budget": 740_499.63, "min": 76.71, "max": 142.47},
    "tk_ipari_zoldsegnoveny": {"budget": 15_619_889.00, "min": 260.44, "max": 390.66},
    "tk_zoldsegnoveny": {"budget": 6_455_314.80, "min": 180.92, "max": 289.48},
    "tk_rizs": {"budget": 1_979_171.48, "min": 582.75, "max": 874.13},
    "tk_hizottbika": {"budget": 5_061_180.70, "min": 40.76, "max": 65.22},
    "tk_anyatehen": {"budget": 34_631_327.25, "min": 111.96, "max": 174.94},
    "tk_tejhasznu_tehen": {"budget": 68_274_257.17, "min": 237.74, "max": 396.24},
    "tk_anyajuh": {"budget": 21_767_730.60, "min": 24.16, "max": 35.53},
}

aop_fajlagos = 47.92
vp_akg_2021_fajlagos = 313.57

tk_cukorrepa_fajlagos = 490.56
tk_szemes_feherjenoveny_fajlagos = 130.02
tk_szalas_feherjenoveny_fajlagos = 66.34
tk_extenziv_gyumolcs_fajlagos = 199.75
tk_intentziv_gyumolcs_fajlagos = 378.81
tk_ipari_olajnoveny_fajlagos = 121.58
tk_ipari_zoldsegnoveny_fajlagos = 384.99
tk_zoldsegnoveny_fajlagos = 135.63
tk_rizs_fajlagos = 840.68
tk_hizottbika_fajlagos = 95.96
tk_anyatehen_fajlagos = 168.12
tk_tejhasznu_tehen_fajlagos = 168.12
tk_anyajuh_fajlagos = 35.25

data["subs_aop"] = aop_fajlagos * data["area_aop"]
data["subs_vp_akg_2021"] = vp_akg_2021_fajlagos * data["area_vp_akg_2021"]
data["subs_tk_cukorrepa"] = tk_cukorrepa_fajlagos * data["area_tk_cukorrepa"]
data["subs_tk_szemes_feherjenoveny"] = (
    tk_szemes_feherjenoveny_fajlagos * data["area_tk_szemes_feherjenoveny"]
)
data["subs_tk_szalas_feherjenoveny"] = (
    tk_szalas_feherjenoveny_fajlagos * data["area_tk_szalas_feherjenoveny"]
)
data["subs_tk_extenziv_gyumolcs"] = (
    tk_extenziv_gyumolcs_fajlagos * data["area_tk_extenziv_gyumolcs"]
)
data["subs_tk_intenziv_gyumolcs"] = (
    tk_intentziv_gyumolcs_fajlagos * data["area_tk_intenziv_gyumolcs"]
)
data["subs_tk_ipari_olajnoveny"] = (
    tk_ipari_olajnoveny_fajlagos * data["area_tk_ipari_olajnoveny"]
)
data["subs_tk_ipari_zoldsegnoveny"] = (
    tk_ipari_zoldsegnoveny_fajlagos * data["area_tk_ipari_zoldsegnoveny"]
)
data["subs_tk_zoldsegnoveny"] = (
    tk_zoldsegnoveny_fajlagos * data["area_tk_zoldsegnoveny"]
)
data["subs_tk_rizs"] = tk_rizs_fajlagos * data["area_tk_rizs"]
data["subs_tk_hizottbika"] = tk_hizottbika_fajlagos * data["count_tk_hizottbika"]
data["subs_tk_anyatehen"] = tk_anyatehen_fajlagos * data["count_tk_anyatehen"]
data["subs_tk_tejhasznu_tehen"] = (
    tk_tejhasznu_tehen_fajlagos * data["count_tk_tejhasznu_tehen"]
)
data["subs_tk_anyajuh"] = tk_anyajuh_fajlagos * data["count_tk_anyajuh"]

data["subs_total"] = (
    data["subs_biss"].fillna(0)
    + data["subs_redist"].fillna(0)
    + data["subs_yfs"].fillna(0)
    + data["subs_aop"].fillna(0)
    + data["subs_vp_akg_2021"].fillna(0)
    + data["subs_tk_cukorrepa"].fillna(0)
    + data["subs_tk_szemes_feherjenoveny"].fillna(0)
    + data["subs_tk_extenziv_gyumolcs"].fillna(0)
    + data["subs_tk_intenziv_gyumolcs"].fillna(0)
    + data["subs_tk_ipari_olajnoveny"].fillna(0)
    + data["subs_tk_ipari_zoldsegnoveny"].fillna(0)
    + data["subs_tk_zoldsegnoveny"].fillna(0)
    + data["subs_tk_rizs"].fillna(0)
    + data["subs_tk_hizottbika"].fillna(0)
    + data["subs_tk_anyatehen"].fillna(0)
    + data["subs_tk_tejhasznu_tehen"].fillna(0)
    + data["subs_tk_anyajuh"].fillna(0)
)

rename_rules = {"subs_redist": "subs_criss"}

target_cols = [
    "regszam",
    "area_biss_criss",
    "area_yfs",
    "subs_biss",
    "subs_criss",
    "subs_yfs",
    "subs_aop",
    "subs_vp_akg_2021",
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
    "subs_total",
    "megye",
]

data = data.rename(columns=rename_rules)

data = data[target_cols]
data = data[~data["regszam"].isin([1002124212, 1022483254, 1022730206])]

data.to_csv("input/data_2024.csv", index=False)
