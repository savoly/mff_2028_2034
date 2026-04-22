WITH tbl_nov AS (
    SELECT
        regszam,
        SUM(terulet_elf) AS area_biss_criss,
        SUM(CASE WHEN tk_cukorrepa = 1 THEN terulet_elf ELSE 0 END) AS tk_cukorrepa,
        SUM(CASE WHEN tk_ipari_zoldseg = 1 THEN terulet_elf ELSE 0 END) AS tk_ipari_zoldseg,
        SUM(CASE WHEN tk_zoldsegnoveny = 1 THEN terulet_elf ELSE 0 END) AS tk_zoldsegnoveny,
        SUM(CASE WHEN tk_extenziv_gyumolcs = 1 THEN terulet_elf ELSE 0 END) AS tk_extenziv_gyumolcs,
        SUM(CASE WHEN tk_intenziv_gyumolcs = 1 THEN terulet_elf ELSE 0 END) AS tk_intenziv_gyumolcs,
        SUM(CASE WHEN tk_szemes_feherjenoveny = 1 THEN terulet_elf ELSE 0 END) AS tk_szemes_feherjenoveny,
        SUM(CASE WHEN tk_szalas_feherjenoveny = 1 THEN terulet_elf ELSE 0 END) AS tk_szalas_feherjenoveny,
        SUM(CASE WHEN tk_ipari_olajnoveny = 1 THEN terulet_elf ELSE 0 END) AS tk_ipari_olajnoveny,
        SUM(CASE WHEN tk_rizs = 1 THEN terulet_elf ELSE 0 END) AS tk_rizs,
        SUM(CASE WHEN fiatal_mgi_termelo = 1 THEN terulet_elf ELSE 0 END) AS area_yfs
    FROM ek_adatok.tera
    WHERE
        ev = 2024
        AND teruletalapu_alaptamogatas = 1
        AND tam_nem_ig = 0
        AND terulet_elf > 0
    GROUP BY regszam
),

tbl_allat AS (
    SELECT
        regszam,
        SUM(CASE WHEN allat_tip = 'anyajuh' THEN letszam_ig ELSE 0 END) AS tk_anyajuh,
        SUM(CASE WHEN allat_tip = 'hízottbika' THEN letszam_ig ELSE 0 END) AS tk_hizottbika,
        SUM(CASE WHEN allat_tip = 'tejhasznú tehén' THEN letszam_ig ELSE 0 END) AS tk_tejhasznu_tehen,
        SUM(CASE WHEN allat_tip = 'anyatehén' THEN letszam_ig ELSE 0 END) AS tk_anyatehen
    FROM mvh_tk_kozv.tam_allat
    WHERE ev = 2024
    GROUP BY regszam
)

SELECT
    tbl_nov.regszam,
    tbl_nov.area_biss_criss,
    tbl_nov.area_yfs,
    tbl_nov.tk_cukorrepa,
    tbl_nov.tk_ipari_zoldseg,
    tbl_nov.tk_zoldsegnoveny,
    tbl_nov.tk_extenziv_gyumolcs,
    tbl_nov.tk_intenziv_gyumolcs,
    tbl_nov.tk_szemes_feherjenoveny,
    tbl_nov.tk_szalas_feherjenoveny,
    tbl_nov.tk_ipari_olajnoveny,
    tbl_nov.tk_rizs,
    tbl_allat.tk_anyajuh,
    tbl_allat.tk_hizottbika,
    tbl_allat.tk_tejhasznu_tehen,
    tbl_allat.tk_anyatehen
FROM tbl_nov FULL OUTER JOIN
    tbl_allat ON tbl_nov.regszam = tbl_allat.regszam;
