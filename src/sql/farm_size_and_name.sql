with tbl_ugyfela as (
    select
        regszam,
        nev,
        vallalkozasi_forma,
        lakhely_szekhely
    from altalanos.ugyfela
    where ev = 2024
),

tbl_uzem_ter as (
    select
        regszam,
        sum(terulet_elf) as ter
    from ek_adatok.tera
    where
        ev = 2024
        and terulet_elf > 0
        and tam_nem_ig = 0
        and teruletalapu_alaptamogatas = 1
    group by regszam
)

select
    t1.ter,
    t1.regszam,
    t2.nev,
    t2.vallalkozasi_forma,
    t2.lakhely_szekhely
from tbl_uzem_ter as t1
left join
    tbl_ugyfela as t2
    on t1.regszam = t2.regszam
order by t2.ter desc;
