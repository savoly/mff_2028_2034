import numpy as np
import pandas as pd
import polars as pl
from data_tools.db import Manager
from sqlalchemy import bindparam, text

from mff.new_cap import cal_redist_vec, read_extended_base_data

engine = Manager("mvh-admin", "mvh").engine


def query_area(regszam_lst):
    query = text("""
        SELECT regszam,
               sum(terulet_elf) as ter
        FROM ek_adatok.tera
        WHERE ev = 2024
          AND terulet_elf > 0
          AND tam_nem_ig = 0
          AND teruletalapu_alaptamogatas = 1
          AND regszam IN :regszam_lst
        GROUP BY ev, regszam
    """).bindparams(
        bindparam("regszam_lst", expanding=True),
    )

    return pl.read_database(
        query,
        connection=engine,
        execute_options={
            "parameters": {
                "regszam_lst": regszam_lst,
            }
        },
    ).to_pandas()


data_merged = pd.read_excel("farms_to_merge.xlsx", engine="calamine")
regszam_lst = data_merged["regszam"].astype("Int64").dropna().to_list()
data_area = query_area(regszam_lst)

data_merged = (
    data_merged[["azonosito", "regszam", "nev", "cegnev", "cim"]]
    .dropna()
    .reset_index(drop=True)
)
data_merged = data_merged.merge(data_area, how="left", on="regszam")
data_merged = data_merged.dropna().reset_index(drop=True)
data_merged.to_excel("input/data_merged_queried.xlsx", index=False)

data_merged = pd.read_excel("input/data_merged_queried.xlsx", engine="calamine")
data_merged["ossz_ter"] = data_merged.groupby("azonosito")["ter"].transform("sum")
data_merged.to_excel("data_merged_queried.xlsx", index=False)


data_merged = pd.read_excel("input/data_merged_queried.xlsx")

cols = [
    "area_biss_criss",
    "area_yfs",
    "area_yfs_cur_eligible",
    "subs_biss",
    "subs_redist",
    "subs_yfs",
]

data = read_extended_base_data(2024)

df = data.merge(
    data_merged[["regszam", "azonosito"]],
    on="regszam",
    how="left",
)

to_merge = df[df["azonosito"].notna()]
rest = df[df["azonosito"].isna()]

merged = to_merge.groupby("azonosito", as_index=False)[cols].sum()
merged["regszam"] = to_merge.groupby("azonosito")["regszam"].first().values

out = pd.concat(
    [
        rest[data.columns],  # untouched
        merged[data.columns],  # collapsed
    ],
    ignore_index=True,
)

out["area_yfs_cur_eligible"] = np.minimum(out["area_yfs"], 300)
out["subs_biss"] = out["area_biss_criss"] * 148.1
out["subs_redist"] = cal_redist_vec(out["area_biss_criss"])
out["subs_yfs"] = out["area_yfs_cur_eligible"] * 90
out.to_parquet("input/database_merged.parquet")

data = pd.read_parquet("input/database_merged.parquet")
