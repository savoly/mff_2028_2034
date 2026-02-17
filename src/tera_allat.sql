with tbl_nov as (
    select
        regszam,
        sum(terulet_elf) as area_biss_criss,
        sum(case when tk_cukorrepa = 1 then terulet_elf else 0 end)
            as tk_cukorrepa,
        sum(case when tk_ipari_zoldseg = 1 then terulet_elf else 0 end)
            as tk_ipari_zoldseg,
        sum(case when tk_zoldsegnoveny = 1 then terulet_elf else 0 end)
            as tk_zoldsegnoveny,
        sum(case when tk_extenziv_gyumolcs = 1 then terulet_elf else 0 end)
            as tk_extenziv_gyumolcs,
        sum(case when tk_intenziv_gyumolcs = 1 then terulet_elf else 0 end)
            as tk_intenziv_gyumolcs,
        sum(case when tk_szemes_feherjenoveny = 1 then terulet_elf else 0 end)
            as tk_szemes_feherjenoveny,
        sum(case when tk_szalas_feherjenoveny = 1 then terulet_elf else 0 end)
            as tk_szalas_feherjenoveny,
        sum(case when tk_ipari_olajnoveny = 1 then terulet_elf else 0 end)
            as tk_ipari_olajnoveny,
        sum(case when tk_rizs = 1 then terulet_elf else 0 end) as tk_rizs,
        sum(case when fiatal_mgi_termelo = 1 then terulet_elf else 0 end)
            as area_yfs
    from ek_adatok.tera
    where
        ev = 2024
        and teruletalapu_alaptamogatas = 1
        and tam_nem_ig = 0
        and terulet_elf > 0
    group by regszam
),

tbl_allat as (
    select
        regszam,
        sum(case when allat_tip = 'anyajuh' then letszam_ig else 0 end)
            as tk_anyajuh,
        sum(case when allat_tip = 'hízottbika' then letszam_ig else 0 end)
            as tk_hizottbika,
        sum(
            case when allat_tip = 'tejhasznú tehén' then letszam_ig else 0 end
        ) as tk_tejhasznu_tehen,
        sum(case when allat_tip = 'anyatehén' then letszam_ig else 0 end)
            as tk_anyatehen
    from mvh_tk_kozv.tam_allat
    where ev = 2024
    group by regszam

)

select
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
from tbl_nov full outer join
    tbl_allat on tbl_nov.regszam = tbl_allat.regszam;
