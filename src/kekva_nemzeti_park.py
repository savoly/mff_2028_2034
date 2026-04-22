from data_tools.db import Manager

import polars as pl
from sqlalchemy import text, bindparam

engine = Manager("mvh-admin", "mvh").engine

data = pl.read_excel(
    "input/kekva_nemzeti_park.xlsx",
    sheet_name="KEKVA",
    columns=["nev", "regszam", "tipus"],
)

regszams = data.select(pl.col("regszam").drop_nulls()).to_series().to_list()


def query_data(regszam_lst, ev_lst):
    query = text("""
        SELECT ev,
               regszam,
               sum(terulet_elf) as ter
        FROM ek_adatok.tera
        WHERE ev IN :ev_lst
          AND terulet_elf > 0
          AND tam_nem_ig = 0
          AND teruletalapu_alaptamogatas = 1
          AND regszam IN :regszam_lst
        GROUP BY ev, regszam
    """).bindparams(
        bindparam("regszam_lst", expanding=True),
        bindparam("ev_lst", expanding=True),
    )

    return pl.read_database(
        query,
        connection=engine,
        execute_options={
            "parameters": {
                "ev_lst": ev_lst,
                "regszam_lst": regszam_lst,
            }
        },
    )


df = query_data(regszams, [2023, 2024]).pivot("ev", index="regszam", values="ter")

df = data.join(df, on="regszam", how="left")


def query_data_kifiz(regszam_lst, ev_lst):
    query = text("""
       SELECT ev, regszam, nev, cim, jogcim, tam FROM mvh_kif.kifiz
        WHERE ev IN :ev_lst AND 
              regszam IN :regszam_lst AND
              tam > 0
    """).bindparams(
        bindparam("regszam_lst", expanding=True),
        bindparam("ev_lst", expanding=True),
    )

    return pl.read_database(
        query,
        connection=engine,
        execute_options={
            "parameters": {
                "ev_lst": ev_lst,
                "regszam_lst": regszam_lst,
            }
        },
    )


df = query_data_kifiz(regszams, [2024]).pivot(
    index="nev", on="jogcim", values="tam", aggregate_function="sum"
)

cols = df.columns
df = df.select([cols[0]] + sorted(cols[1:]))

df.write_excel("output/kekva.xlsx")
