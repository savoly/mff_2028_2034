from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import RootResults, root_scalar, minimize

from sqlalchemy.engine import Engine
from sqlalchemy import text

from utils import c_round

Number = float | int


def generate_data_for_gams(base_year: int, engine: Engine) -> None:
    sql = text("""
        SELECT
            ROW_NUMBER() OVER (ORDER BY SUM(terulet_elf)) AS uzem_kod,
            SUM(terulet_elf) AS area_biss_criss,
            LEAST(SUM(CASE WHEN fiatal_mgi_termelo = 1 THEN terulet_elf ELSE 0 END), 300) AS area_yfs,
            CASE WHEN SUM(terulet_elf) <= 1200 THEN LEAST(SUM(terulet_elf), 10) ELSE 0 END AS area_criss_step1,
            CASE WHEN SUM(terulet_elf) <= 1200 THEN LEAST(GREATEST(SUM(terulet_elf) - 10, 0), 140) ELSE 0 END AS area_criss_step2
        FROM ek_adatok.tera
        WHERE ev = :base_year
          AND teruletalapu_alaptamogatas = 1
          AND tam_nem_ig = 0
          AND terulet_elf > 0
        GROUP BY regszam
        ORDER BY area_biss_criss
    """)
    data = pd.read_sql_query(sql, con=engine, params={"base_year": base_year})
    data.to_csv("base_2024.csv", index=False)


def compute_dabis_support_summary(data: pd.DataFrame) -> pd.DataFrame:
    data_criss = data.loc[data["area_biss_criss"] <= 1200, "area_biss_criss"].copy()

    return pd.DataFrame(
        {
            "Támogatási elemek": [
                "Általános támogatás",
                "Differenciált támogatás (0–10 ha)",
                "Differenciált támogatás (10–150 ha)",
                "Fiatal gazda támogatás",
            ],
            "Jogosultak száma, db": [
                data.shape[0],
                data_criss.shape[0],
                data_criss[data_criss > 10].shape[0],
                data[data["area_yfs_cur_eligible"] > 0].shape[0],
            ],
            "Jogosult terület, ha": [
                data["area_biss_criss"].sum(),
                np.minimum(data_criss, 10).sum(),
                np.maximum(0, np.minimum(150, data_criss) - 10).sum(),
                data["area_yfs_cur_eligible"].sum(),
            ],
        }
    )


def calculate_flat_rate(
    data: pd.DataFrame,
    budget: float,
    subs_per_ha: float,
    yfs_per_ha: float,
    redist_per_ha: tuple[float, float],
) -> float:
    return (
        budget
        - compute_capped_subsidies(data, subs_per_ha, yfs_per_ha, redist_per_ha)[
            "subs_capped"
        ].sum()
    )


def find_flat_rate(
    data: pd.DataFrame, budget: float, yfs: float, redist_per_ha: tuple[float, float]
) -> RootResults:
    return root_scalar(
        lambda x: calculate_flat_rate(data, budget, x, yfs, redist_per_ha),
        bracket=(100.0, 500.0),
        method="brentq",
    )


def find_budget(
    data: pd.DataFrame, flat_rate: float, yfs: float, redist_per_ha: tuple[float, float]
) -> RootResults:
    return root_scalar(
        lambda x: calculate_flat_rate(data, x, flat_rate, yfs, redist_per_ha),
        bracket=(1e6, 1.5 * 1e9),
        method="brentq",
    )


def find_cur_new_equal_root(
    eur_per_ha: float, redist_per_ha: tuple[float, float], yfs_per_ha: float
) -> float:
    def f(x):
        return apply_reductions(
            eur_per_ha * x + yfs_per_ha * min(300, x) + cal_redist(x, redist_per_ha)
        ) - compute_current_support(x, yfs_per_ha)

    sol = root_scalar(f, bracket=(1e-3, 4e9), method="brentq")
    return sol.root


def apply_reductions(amount: float) -> float:
    capped = 0.0

    if amount <= 20_000:
        capped = amount
    elif amount <= 50_000:
        capped = 20_000 + 0.75 * (amount - 20_000)
    elif amount <= 75_000:
        capped = 20_000 + 0.75 * 30_000 + 0.5 * (amount - 50_000)
    else:
        capped = 20_000 + 0.75 * 30_000 + 0.5 * 25_000 + 0.25 * (amount - 75_000)
        capped = min(capped, 100_000)

    return capped


def apply_reductions_vec(series: pd.Series) -> pd.Series:
    x = series.to_numpy()
    capped = np.piecewise(
        x,
        [
            x <= 20_000,
            (x > 20_000) & (x <= 50_000),
            (x > 50_000) & (x <= 75_000),
            x > 75_000,
        ],
        [
            lambda x: x,
            lambda x: 20_000 + 0.75 * (x - 20_000),
            lambda x: 20_000 + 0.75 * 30_000 + 0.5 * (x - 50_000),
            lambda x: np.minimum(
                20_000 + 0.75 * 30_000 + 0.5 * 25_000 + 0.25 * (x - 75_000), 100_000
            ),
        ],
    )
    return pd.Series(capped, index=series.index)


def cal_redist_vec(
    biss_ter: pd.Series | np.ndarray, redist_per_ha=(86.98, 43.49)
) -> np.ndarray:
    x = np.asarray(biss_ter, dtype=float)

    part1 = redist_per_ha[0] * np.minimum(10, x)
    part2 = redist_per_ha[1] * np.maximum(0, np.minimum(150, x) - 10)

    result = np.where((x <= 1200) & (x != 0), part1 + part2, 0.0)
    return result


def cal_redist(biss_ter: float, redist_per_ha=[86.98, 43.49]) -> float:
    if biss_ter <= 1200 and biss_ter != 0:
        part1 = redist_per_ha[0] * min(10, biss_ter)
        part2 = redist_per_ha[1] * max(0, min(150, biss_ter) - 10)
        return part1 + part2
    else:
        return 0


def generate_extended_base_data(base_year: int, engine: Engine) -> pd.DataFrame:
    sql = text("""
        WITH alap_tabla AS (
            SELECT
                regszam,
                terulet_elf,
                fiatal_mgi_termelo,
                aop,
                vp_akg_2021,
                tk_cukorrepa,
                tk_szemes_feherjenoveny,
                tk_szalas_feherjenoveny,
                tk_extenziv_gyumolcs,
                tk_intenziv_gyumolcs,
                tk_ipari_olajnoveny,
                tk_ipari_zoldseg,
                tk_zoldsegnoveny,
                tk_rizs,
                teruletalapu_alaptamogatas
            FROM ek_adatok.tera
            WHERE
                ev =:base_year
                AND teruletalapu_alaptamogatas = 1
                AND tam_nem_ig = 0
                AND terulet_elf > 0

            UNION ALL

            SELECT
                regszam,
                terulet_elf,
                NULL AS fiatal_mgi_termelo,
                NULL AS aop,
                NULL AS vp_akg_2021,
                tk_cukorrepa,
                tk_szemes_feherjenoveny,
                tk_szalas_feherjenoveny,
                tk_extenziv_gyumolcs,
                tk_intenziv_gyumolcs,
                tk_ipari_olajnoveny,
                tk_ipari_zoldseg,
                tk_zoldsegnoveny,
                tk_rizs,
                NULL as teruletalapu_alaptamogatas
            FROM ek_adatok.masodvetes
            WHERE ev =:base_year AND terulet_elf > 0
        ),

        tbl_noveny AS (
            SELECT
                regszam,
                SUM(CASE WHEN teruletalapu_alaptamogatas = 1 THEN terulet_elf ELSE 0 END) AS area_biss_criss,
                SUM(CASE WHEN fiatal_mgi_termelo = 1 THEN terulet_elf ELSE 0 END) AS area_yfs,
                SUM(CASE WHEN aop = 1 THEN terulet_elf ELSE 0 END) AS area_aop,
                SUM(CASE WHEN vp_akg_2021 = 1 THEN terulet_elf ELSE 0 END) AS area_vp_akg_2021,
                SUM(CASE WHEN tk_cukorrepa = 1 THEN terulet_elf ELSE 0 END) AS area_tk_cukorrepa,
                SUM(CASE WHEN tk_szemes_feherjenoveny = 1 THEN terulet_elf ELSE 0 END) AS area_tk_szemes_feherjenoveny,
                SUM(CASE WHEN tk_szalas_feherjenoveny = 1 THEN terulet_elf ELSE 0 END) AS area_tk_szalas_feherjenoveny,
                SUM(CASE WHEN tk_extenziv_gyumolcs = 1 THEN terulet_elf ELSE 0 END) AS area_tk_extenziv_gyumolcs,
                SUM(CASE WHEN tk_intenziv_gyumolcs = 1 THEN terulet_elf ELSE 0 END) AS area_tk_intenziv_gyumolcs,
                SUM(CASE WHEN tk_ipari_olajnoveny = 1 THEN terulet_elf ELSE 0 END) AS area_tk_ipari_olajnoveny,
                SUM(CASE WHEN tk_ipari_zoldseg = 1 THEN terulet_elf ELSE 0 END) AS area_tk_ipari_zoldsegnoveny,
                SUM(CASE WHEN tk_zoldsegnoveny = 1 THEN terulet_elf ELSE 0 END) AS area_tk_zoldsegnoveny,
                SUM(CASE WHEN tk_rizs = 1 THEN terulet_elf ELSE 0 END) AS area_tk_rizs
            FROM alap_tabla
            GROUP BY regszam
        ),

        tbl_allat AS (
            SELECT
                regszam,
                SUM(CASE WHEN allat_tip = 'hízottbika' THEN letszam_ig ELSE 0 END) AS count_tk_hizottbika,
                SUM(CASE WHEN allat_tip = 'anyajuh' THEN letszam_ig ELSE 0 END) AS count_tk_anyajuh,
                SUM(CASE WHEN allat_tip = 'tejhasznú tehén' THEN letszam_ig ELSE 0 END) AS count_tk_tejhasznu_tehen,
                SUM(CASE WHEN allat_tip = 'anyatehén' THEN letszam_ig ELSE 0 END) AS count_tk_anyatehen
            FROM mvh_tk_kozv.tam_allat
            WHERE
                ev =:base_year
                AND allat_tip IN (
                    'hízottbika', 'anyajuh', 'tejhasznú tehén', 'anyatehén'
                )
            GROUP BY regszam
        )

        SELECT
            COALESCE(tbl_noveny.regszam, tbl_allat.regszam) AS regszam,
            COALESCE(tbl_noveny.area_biss_criss, 0) AS area_biss_criss,
            COALESCE(tbl_noveny.area_yfs, 0) AS area_yfs,
            COALESCE(tbl_noveny.area_aop, 0) AS area_aop,
            COALESCE(tbl_noveny.area_vp_akg_2021, 0) AS area_vp_akg_2021,
            COALESCE(tbl_noveny.area_tk_cukorrepa, 0) AS area_tk_cukorrepa,
            COALESCE(tbl_noveny.area_tk_szemes_feherjenoveny, 0) AS area_tk_szemes_feherjenoveny,
            COALESCE(tbl_noveny.area_tk_szalas_feherjenoveny, 0) AS area_tk_szalas_feherjenoveny,
            COALESCE(tbl_noveny.area_tk_extenziv_gyumolcs, 0) AS area_tk_extenziv_gyumolcs,
            COALESCE(tbl_noveny.area_tk_intenziv_gyumolcs, 0) AS area_tk_intenziv_gyumolcs,
            COALESCE(tbl_noveny.area_tk_ipari_olajnoveny, 0) AS area_tk_ipari_olajnoveny,
            COALESCE(tbl_noveny.area_tk_ipari_zoldsegnoveny, 0) AS area_tk_ipari_zoldsegnoveny,
            COALESCE(tbl_noveny.area_tk_zoldsegnoveny, 0) AS area_tk_zoldsegnoveny,
            COALESCE(tbl_noveny.area_tk_rizs, 0) AS area_tk_rizs,
            COALESCE(tbl_allat.count_tk_hizottbika, 0) AS count_tk_hizottbika,
            COALESCE(tbl_allat.count_tk_anyatehen, 0) AS count_tk_anyatehen,
            COALESCE(tbl_allat.count_tk_tejhasznu_tehen, 0) AS count_tk_tejhasznu_tehen,
            COALESCE(tbl_allat.count_tk_anyajuh, 0) AS count_tk_anyajuh
        FROM tbl_noveny
        FULL JOIN tbl_allat ON tbl_noveny.regszam = tbl_allat.regszam;
    """)

    data = pd.read_sql(sql, con=engine, params={"base_year": base_year})

    if base_year == 2024:
        yfs_data = pd.read_excel(
            "input/2024EK Feldolgozás bizonylatok 2023-2027 Területalapú kérelem soros analitika DP03-BISS.xlsx",
            engine="calamine",
        )
        yfs_eligible = yfs_data[yfs_data["Jóváhagyott támogatás (Ft)"] > 0][
            "Ügyfél azonosító"
        ].tolist()
        data.loc[~data["regszam"].isin(yfs_eligible), "area_yfs"] = 0
    data["area_yfs_cur_eligible"] = np.minimum(data["area_yfs"], 300)
    data["subs_biss"] = data["area_biss_criss"] * 148.1
    data["subs_redist"] = cal_redist_vec(data["area_biss_criss"])
    data["subs_yfs"] = data["area_yfs_cur_eligible"] * 90

    data.to_parquet(f"input/data_extended_{base_year}.parquet")
    return data


def generate_base_data(base_year: int, engine: Engine) -> pd.DataFrame:
    sql = text("""
        SELECT regszam,
               SUM(terulet_elf) AS area_biss_criss,
               SUM(CASE WHEN fiatal_mgi_termelo = 1 THEN terulet_elf ELSE 0 END) AS area_yfs
        FROM ek_adatok.tera
        WHERE ev = :base_year AND teruletalapu_alaptamogatas = 1 AND tam_nem_ig = 0 AND terulet_elf > 0
        GROUP BY regszam
    """)
    data = pd.read_sql(sql, con=engine, params={"base_year": base_year})

    if base_year == 2024:
        yfs_data = pd.read_excel(
            "input/2024EK Feldolgozás bizonylatok 2023-2027 Területalapú kérelem soros analitika DP03-BISS.xlsx",
            engine="calamine",
        )
        yfs_eligible = yfs_data[yfs_data["Jóváhagyott támogatás (Ft)"] > 0][
            "Ügyfél azonosító"
        ].tolist()
        data.loc[~data["regszam"].isin(yfs_eligible), "area_yfs"] = 0
    data["area_yfs_cur_eligible"] = np.minimum(data["area_yfs"], 300)
    data["subs_biss"] = data["area_biss_criss"] * 148.1
    data["subs_redist"] = cal_redist_vec(data["area_biss_criss"])
    data["subs_yfs"] = data["area_yfs_cur_eligible"] * 90

    data.to_parquet(f"input/data_{base_year}.parquet")
    return data


def read_base_data(base_year: int) -> pd.DataFrame:
    return pd.read_parquet(f"input/data_{base_year}.parquet")


def read_extended_base_data(base_year: int) -> pd.DataFrame:
    return pd.read_parquet(f"input/data_extended_{base_year}.parquet")


def compute_capped_subsidies(
    df: pd.DataFrame,
    subs_per_ha: float,
    yfs_per_ha: float,
    redist_per_ha: tuple[float, float],
) -> pd.DataFrame:
    df = df.copy()

    df["subs_eur"] = (
        subs_per_ha * df["area_biss_criss"]
        + yfs_per_ha * df["area_yfs_cur_eligible"]
        + cal_redist_vec(df["area_biss_criss"], redist_per_ha)
    )
    df["subs_capped"] = apply_reductions_vec(df["subs_eur"])
    return df


def analyze_by_area_categories(
    df: pd.DataFrame, bins: list[Number], labels: list[str]
) -> pd.DataFrame:
    """
    Group the DataFrame into area size categories and compute summary statistics.
    """
    df = df.copy()
    df["area_class"] = pd.cut(
        df["area_biss_criss"], bins=bins, labels=labels, right=False
    )

    grouped = df.groupby("area_class", observed=False).agg(
        total_area=("area_biss_criss", "sum"),
        total_subs_cur=("subs_cur", "sum"),
        total_subs_capped=("subs_capped", "sum"),
        total_subs_uncapped=("subs_eur", "sum"),
        total_farmers=("regszam", "size"),
    )
    grouped = grouped.assign(
        avg_subs_per_ha_capped=grouped["total_subs_capped"] / grouped["total_area"],
        avg_subs_per_ha_cur=grouped["total_subs_cur"] / grouped["total_area"],
    )
    grouped = grouped.assign(
        avg_change=100
        * (grouped["avg_subs_per_ha_capped"] / grouped["avg_subs_per_ha_cur"] - 1),
        total_capped_loss=(
            grouped["total_subs_uncapped"] - grouped["total_subs_capped"]
        ).clip(lower=0),
    )

    return grouped


def _fmt(x: float, rounding: int) -> str:
    return str(int(x)) if float(x).is_integer() else str(int(c_round(x, rounding)))


def generate_labels_from_bins(bins: list[float], rounding: int = 0) -> list[str]:
    labels = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]

        if np.isinf(upper):
            label = f"{_fmt(lower, rounding)}+ ha"
        else:
            label = f"{_fmt(lower, rounding)}–{_fmt(upper, rounding)} ha"

        labels.append(label)

    return labels


def summarize_farms_by_area_categories(
    data: pd.DataFrame, bins: list[float]
) -> pd.DataFrame:
    labels = generate_labels_from_bins(bins)

    df = data.copy()
    df["area_class"] = pd.cut(
        df["area_biss_criss"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    total_farms: int = int(len(df))
    total_area: float = float(df["area_biss_criss"].sum())

    grouped: pd.DataFrame = (
        df.groupby("area_class", observed=False)
        .agg(
            number_of_farms=("area_biss_criss", "size"),
            total_area=("area_biss_criss", "sum"),
        )
        .astype({"number_of_farms": "int64"})
    )

    grouped["share_of_farms_pct"] = 100 * grouped["number_of_farms"] / total_farms
    grouped["share_of_area_pct"] = 100 * grouped["total_area"] / total_area

    grouped["cumulative_number_of_farms"] = (
        grouped["number_of_farms"].cumsum().astype(int)
    )
    grouped["cumulative_area"] = grouped["total_area"].cumsum()
    grouped["cumulative_farms_pct"] = grouped["share_of_farms_pct"].cumsum()
    grouped["cumulative_area_pct"] = grouped["share_of_area_pct"].cumsum()

    pct_cols = [
        "share_of_farms_pct",
        "share_of_area_pct",
        "cumulative_farms_pct",
        "cumulative_area_pct",
    ]

    for col in pct_cols:
        grouped[col] = grouped[col].map(lambda x: c_round(x, 2))

    data = grouped.reset_index()
    cols = [
        "area_class",
        "number_of_farms",
        "share_of_farms_pct",
        "cumulative_number_of_farms",
        "cumulative_farms_pct",
        "total_area",
        "share_of_area_pct",
        "cumulative_area",
        "cumulative_area_pct",
    ]
    return data[cols]


def compute_current_support(hectares: float, yfs: float) -> float:
    yfs = 93.10 if yfs != 0 else 0
    return 148.1 * hectares + cal_redist(hectares) + yfs * min(hectares, 300)


def calc_steps(
    x: float,
    val: float,
    eur_per_ha: float,
    redist_per_ha: tuple[float, float],
    yfs_per_ha: float,
) -> float:
    return val - (
        eur_per_ha * x + yfs_per_ha * min(x, 300) + cal_redist(x, redist_per_ha)
    )


def calc_thresholds(
    thresholds: Iterable[Number],
    eur_per_ha: float,
    redist_per_ha: tuple[float, float],
    yfs_per_ha: float,
) -> list[float]:
    result = []
    for val in thresholds:
        res = root_scalar(
            lambda x: calc_steps(x, val, eur_per_ha, redist_per_ha, yfs_per_ha),
            bracket=(1e1, 1e5),
            method="brentq",
        ).root
        result.append(c_round(res, 2))

    return result


def create_dabis_summary(df: pd.DataFrame, bins, labels) -> pd.DataFrame:
    df["Üzemméret kategória"] = pd.cut(
        df["area_biss_criss"], bins=bins, labels=labels, right=False
    )

    grouped = df.groupby("Üzemméret kategória", as_index=False, observed=False).agg(
        **{
            "Üzemszám, db": ("regszam", "size"),
            "Terület, ha": ("area_biss_criss", "sum"),
            "Támogatás degresszió nélkül, EUR": (
                "subs_dabis_before_capping",
                "sum",
            ),
            "Támogatás degresszióval és cappinggel, EUR": (
                "subs_dabis_after_capping",
                "sum",
            ),
        }
    )

    grouped["Támogatás aránya degresszió után, %"] = np.where(
        grouped["Támogatás degresszió nélkül, EUR"] > 0,
        100
        * grouped["Támogatás degresszióval és cappinggel, EUR"]
        / grouped["Támogatás degresszió nélkül, EUR"],
        np.nan,
    )

    grouped.insert(
        0,
        "Degresszív és capping lépcsők",
        [
            "0–20 000 EUR",
            "20 000–50 000 EUR",
            "50 000–75 000 EUR",
            "75 000–100 000 EUR",
            "100 000 EUR felett",
        ],
    )

    return grouped


def calc_degressive_and_capping_steps(
    data: pd.DataFrame,
    dabis_per_ha: float,
    redistributive_per_ha: tuple[float, float],
    yfs_per_ha: float,
    type_of_calc: str = "all",
) -> pd.DataFrame:
    thresholds = [20_000, 50_000, 75_000, 255_000]
    thresholds_ha = calc_thresholds(
        thresholds, dabis_per_ha, redistributive_per_ha, yfs_per_ha
    )

    bins = [0] + thresholds_ha + [np.inf]
    labels = generate_labels_from_bins(bins)

    df = data.copy()

    df["subs_dabis_before_capping"] = (
        dabis_per_ha * df["area_biss_criss"]
        + yfs_per_ha * df["area_yfs_cur_eligible"]
        + cal_redist_vec(df["area_biss_criss"], redistributive_per_ha)
    )
    df["subs_dabis_after_capping"] = apply_reductions_vec(
        df["subs_dabis_before_capping"]
    )

    mask = df["area_yfs_cur_eligible"] > 0
    if type_of_calc == "yf":
        df = df[mask]
    elif type_of_calc == "not-yf":
        df = df[~mask]
    else:
        df = df

    return create_dabis_summary(df, bins, labels)


def calc_ratio_subs(x, eur_per_ha, yfs_per_ha, redist_per_ha):
    new_subs = apply_reductions(
        eur_per_ha * x + yfs_per_ha * min(300, x) + cal_redist(x, redist_per_ha)
    )
    cur_subs = compute_current_support(x, yfs_per_ha)
    return (new_subs - cur_subs) / (cur_subs + 1e-12)


def _neg_ratio_logz(z, eur_per_ha, yfs_per_ha, redist_per_ha):
    x = float(np.exp(z[0]))  # ensures x > 0
    return -calc_ratio_subs(x, eur_per_ha, yfs_per_ha, redist_per_ha)


def maximize_ratio(x0, eur_per_ha, yfs_per_ha, redist_per_ha):
    res = minimize(
        _neg_ratio_logz,
        [np.log(max(x0, 1e-8))],
        args=(eur_per_ha, yfs_per_ha, redist_per_ha),
        method="Nelder-Mead",
    )
    return float(np.exp(res.x[0]))
