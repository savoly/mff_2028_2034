import pandas as pd
from sqlalchemy import text


def create_lakhely_szekhely_tbl(
    year: int,
    engine,
    fallback_year: int = 2024,
) -> pd.DataFrame:
    query = text("""
        WITH eligible AS (
            SELECT
                regszam,
                SUM(terulet_elf) AS ter
            FROM ek_adatok.tera
            WHERE ev = :year
            GROUP BY regszam
        ),

        ugyfel AS (
            SELECT
                u.regszam,
                u.lakhely_szekhely,
                u.nev,
                u.vallalkozasi_forma,
                u.ostermelo,
                u.telephelyek_fioktelepek_egyebek_szama
            FROM altalanos.ugyfela u
            WHERE u.ev = :year

            UNION ALL

            SELECT
                u.regszam,
                u.lakhely_szekhely,
                u.nev,
                u.vallalkozasi_forma,
                u.ostermelo,
                u.telephelyek_fioktelepek_egyebek_szama
            FROM altalanos.ugyfela u
            WHERE u.ev = :fallback_year
              AND NOT EXISTS (
                  SELECT 1
                  FROM altalanos.ugyfela current
                  WHERE current.ev = :year
                    AND current.regszam = u.regszam
              )
        ),

        members AS (
            SELECT DISTINCT
                u.lakhely_szekhely,
                u.regszam,
                e.ter
            FROM ugyfel u
            INNER JOIN eligible e USING (regszam)
            WHERE u.lakhely_szekhely IS NOT NULL
        ),

        address_stats AS (
            SELECT
                lakhely_szekhely,
                COUNT(*) AS regszam_db,
                SUM(ter) AS ter_osszes
            FROM members
            GROUP BY lakhely_szekhely
        )

        SELECT
            u.lakhely_szekhely,
            s.regszam_db,
            u.regszam,
            u.nev,
            u.vallalkozasi_forma,
            u.ostermelo,
            u.telephelyek_fioktelepek_egyebek_szama,
            e.ter,
            s.ter_osszes
        FROM ugyfel u
        INNER JOIN eligible e USING (regszam)
        INNER JOIN address_stats s USING (lakhely_szekhely)
        ORDER BY
            s.ter_osszes DESC,
            s.regszam_db DESC,
            u.lakhely_szekhely DESC,
            u.regszam;
    """)

    data = pd.read_sql(
        query,
        con=engine,
        params={
            "year": year,
            "fallback_year": fallback_year,
        },
    )

    data.to_excel(f"input/farms_under_one_control_{year}.xlsx")
    data.to_parquet(f"input/farms_under_one_control_{year}.parquet")

    return data
