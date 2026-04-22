import pandas as pd

from data_tools.db import Manager
from mff.new_cap import generate_extended_base_data

engine = Manager("mvh-admin", "mvh").engine

data = generate_extended_base_data(2024, engine)

engine = Manager("mvh-admin", "mvh").engine

QUERY = """
SELECT DISTINCT ON (t.regszam)
          t.regszam,
          h.megye_nev AS megye
FROM ek_adatok.tera t
LEFT JOIN ek_adatok.mepar m
    ON t.ev = m.ev
   AND t.mepar_blokkaz = m.mepar_blokkaz
LEFT JOIN szotar.hnt_2024 h
    ON m.mepar_blokk_telepules = h.helyseg_nev
WHERE
    t.ev = 2024
    AND t.teruletalapu_alaptamogatas = 1
    AND t.tam_nem_ig = 0
    AND t.terulet_elf > 0
GROUP BY
    t.regszam,
    h.megye_nev
ORDER BY
    t.regszam,
    SUM(t.terulet_elf) DESC,
    h.megye_nev
"""

mepar = pd.read_sql(sql=QUERY, con=engine)

data = data.merge(mepar, how="left", on="regszam")

mapping = {
    "Bács-Kiskun": "Bacs-Kiskun",
    "Szabolcs-Szatmár-Bereg": "Szabolcs-Szatmar-Bereg",
    "Hajdú-Bihar": "Hajdu-Bihar",
    "Békés": "Bekes",
    "Csongrád-Csanád": "Csongrad-Csanad",
    "Jász-Nagykun-Szolnok": "Jasz-Nagykun-Szolnok",
    "Pest": "Pest",
    "Borsod-Abaúj-Zemplén": "Borsod-Abauj-Zemplen",
    "Győr-Moson-Sopron": "Gyor-Moson-Sopron",
    "Somogy": "Somogy",
    "Fejér": "Fejer",
    "Heves": "Heves",
    "Tolna": "Tolna",
    "Veszprém": "Veszprem",
    "Baranya": "Baranya",
    "Zala": "Zala",
    "Vas": "Vas",
    "Nógrád": "Nograd",
    "Komárom-Esztergom": "Komarom-Esztergom",
    "főváros": "Budapest",
}

data["megye"] = data["megye"].fillna("Ismeretlen")

df = data.drop("regszam", axis=1).sort_values(by="area_biss_criss")

df.to_csv("data_2024.csv", index=False)

rename_rules = {
    "subs_biss": "biss",
    "subs_redist": "criss",
    "subs_yfs": "yf",
    "subs_aop": "aop",
    "subs_vp_akg_2021": "akg",
}

target_cols = [
    "area_biss_criss",
    "area_yfs",
    "biss",
    "criss",
    "yf",
    "aop",
    "akg",
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

df_aligned = df.rename(columns=rename_rules)[target_cols]
df_aligned.to_csv("data_2024_aligned.csv", index=False)
