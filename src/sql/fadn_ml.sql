SELECT
    w.ev,
    w.akod,
    ts.legal,
    ts.tip_m10ste,
    ts.regio3,

    t2a.m2040_05 as egyeni_munkaora,
    t2b2.m2080_04 as tarsas_munkaora,
    t1a.m1117_08 AS osszes_mgi_ter_ha,
    t7b1.m7407_03 AS yfs_ha,
    t1a.m1100_08 as szantoterulet_ha,
    t1a.m1105_08 as gyep_ha,
    t1a.m1108_08 as konyhakert_ha,
    t1a.m1112_08 as szolo_ossz_ha,
    t1a.m1113_08 as szolo_termo_ha,
    t1a.m1114_08 as gyumolcsos_ossz,
    t1a.m1115_08 as gyumolcsos_termo_ha,
    t1a.m1120_08 as erdo_ha,
    t5b.m5229_08 as hizo_bika_atlag_allomany_db,
    t5b.m5266_08 as anyajuh_atlag_allomany_db,
    t5b.m5218_08 as tejelo_tehen_atlag_allomany_db,
    t5b.m5259_08 AS sertes_atlagletszam_db,
    t5b.m5309_08 as baromfi_atlag_allomany


FROM public.weights AS w

LEFT JOIN public.tipo_ste AS ts
    ON w.akod = ts.akod
   AND w.ev = ts.ev

LEFT JOIN public.t1_a AS t1a
    ON w.akod = t1a.akod
   AND w.ev = t1a.ev

LEFT JOIN public.t2_a AS t2a
    ON w.akod = t2a.akod
   AND w.ev = t2a.ev

LEFT JOIN public.t2_b2 AS t2b2
    ON w.akod = t2b2.akod
   AND w.ev = t2b2.ev

LEFT JOIN public.t4_a AS t4a
    ON w.akod = t4a.akod
   AND w.ev = t4a.ev

LEFT JOIN public.t5_b AS t5b
    ON w.akod = t5b.akod
   AND w.ev = t5b.ev

LEFT JOIN public.t7_b1 AS t7b1
    ON w.akod = t7b1.akod
   AND w.ev = t7b1.ev

WHERE w.ev IN (2023, 2024, 2025)
  AND ts.ste_ev = 2020

ORDER BY
    w.ev,
    w.akod;
