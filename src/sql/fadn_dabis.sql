SELECT
    w.ev,
    w.akod,
    ts.legal,
    ts.tip_m10ste,
    t1a.m1117_08 AS mezogazdasagi_terulet_ha,
    t7b1.m7400_03 AS biss_ha,
    t7b1.m7401_03 AS criss_ha,
    t7b1.m7407_03 AS yfs_ha,
    t5b.m5239_08 as szarvasmarha_atlagletszam_db,
    t5b.m5259_08 as sertes_atlagletszam_db,
    t5b.m5269_08 as juh_atlagletszam_db,
    t5b.m5309_08 as baromfi_atlagletszam_db,
    ts.meret_14ste AS meret_14,
    COALESCE(w.weights_ste_tipm10, w.weights_ste) AS weights_ste,
    1000 * t7c.m7219_03 AS tamogatasok_osszesen_ft,
    1000 * t4a.m4679_03 AS adozott_eredmeny_ft,
    1000 * (t4a.m4599_03 + t4a.m4559_03) AS ebitda_ft,
    1000 * t4a.m4280_03 as brutto_termelesi_ertek_ft,
    1000
    * (
        t4a.m4280_03
        - t4a.m4279_03
        + t5c.m5619_06
        - t5c.m5619_04
        + t5c.m5849_06
        - t5c.m5849_04
        + t5c.m6060_06
        - t5c.m6060_04
        + t5b.m5349_06
        - t5b.m5349_04
        + t5c.m6069_06
        - t5c.m6069_04
        + t5c.m6075_06
        - t5c.m6075_04
        - t4a.m4115_03
        - t4a.m4281_03
        - t4a.m4285_03
        - t4a.m4287_03
        - t4a.m4288_03
        - t4a.m4291_03
        - t4a.m4295_03
        - t4a.m4301_03
        - t4a.m4305_03
        - t4a.m4311_03
        - t4a.m4321_03
        - t4a.m4325_03
        - t4a.m4331_03
        - t4a.m4335_03
        - t4a.m4348_03
        - t4a.m4351_03
        - t4a.m4355_03
        - t4a.m4361_03
        - t4a.m4365_03
        - t4a.m4371_03
        - t4a.m4375_03
        - t4a.m4388_03
        - t4a.m4401_03
        - t4a.m4402_03
        - t4a.m4403_03
        - t4a.m4405_03
        - t4a.m4413_03
        - t4a.m4414_03
        - t4a.m4415_03
        - t4a.m4421_03
        - t4a.m4425_03
        - t4a.m4435_03
        - t4a.m4441_03
        - t4a.m4455_03
        + t4a.m4456_03
        - t4a.m4461_03
        - t4a.m4485_03
        - t4a.m4585_03
        + t4a.m4586_03
        - t4a.m4628_03
        - t4a.m4451_03
        - t4a.m4575_03
        - t4a.m4578_03
        - t4a.m4315_03
        + t4a.m4577_03
        - t4a.m4286_03
        - t4a.m4390_03
        - t4a.m4391_03
        - t4a.m4392_03
        - t4a.m4393_03
        - t4a.m4394_03
        - t4a.m4395_03
        - t4a.m4396_03
        - t4a.m4397_03
        - t4a.m4398_03
        - t4a.m4404_03
    ) AS brutto_hozzadott_ertek_ft,
    1000
    * (
        t4a.m4280_03
        - t4a.m4279_03
        + t5c.m5619_06
        - t5c.m5619_04
        + t5c.m5849_06
        - t5c.m5849_04
        + t5c.m6060_06
        - t5c.m6060_04
        + t5b.m5349_06
        - t5b.m5349_04
        + t5c.m6069_06
        - t5c.m6069_04
        + t5c.m6075_06
        - t5c.m6075_04
        - t4a.m4115_03
        - t4a.m4281_03
        - t4a.m4285_03
        - t4a.m4287_03
        - t4a.m4288_03
        - t4a.m4291_03
        - t4a.m4295_03
        - t4a.m4301_03
        - t4a.m4305_03
        - t4a.m4311_03
        - t4a.m4315_03
        - t4a.m4321_03
        - t4a.m4325_03
        - t4a.m4331_03
        - t4a.m4335_03
        - t4a.m4348_03
        - t4a.m4351_03
        - t4a.m4355_03
        - t4a.m4361_03
        - t4a.m4365_03
        - t4a.m4371_03
        - t4a.m4375_03
        - t4a.m4388_03
        - t4a.m4401_03
        - t4a.m4402_03
        - t4a.m4403_03
        - t4a.m4405_03
        - t4a.m4413_03
        - t4a.m4414_03
        - t4a.m4415_03
        - t4a.m4421_03
        - t4a.m4425_03
        - t4a.m4435_03
        - t4a.m4441_03
        - t4a.m4455_03
        + t4a.m4456_03
        - t4a.m4461_03
        - t4a.m4485_03
        - t4a.m4585_03
        + t4a.m4586_03
        - t4a.m4628_03
        - t4a.m4451_03
        - t4a.m4575_03
        - t4a.m4578_03
        - t5a.m5179_10
        - t5a.m5179_11
        + t4a.m4577_03
        - t4a.m4286_03
        - t4a.m4390_03
        - t4a.m4391_03
        - t4a.m4392_03
        - t4a.m4393_03
        - t4a.m4394_03
        - t4a.m4395_03
        - t4a.m4396_03
        - t4a.m4397_03
        - t4a.m4398_03
        - t4a.m4404_03
    ) AS netto_hozzadott_ertek_ft,
    1000 * t7b1.m7400_04 AS biss_ft,
    1000 * t7b1.m7401_04 AS criss_ft,
    1000 * t7b1.m7407_04 AS yfs_ft
FROM public.weights AS w
LEFT JOIN public.tipo_ste AS ts
    ON w.akod = ts.akod AND w.ev = ts.ev
LEFT JOIN public.t1_a AS t1a
    ON w.akod = t1a.akod AND w.ev = t1a.ev
LEFT JOIN public.t4_a AS t4a
    ON w.akod = t4a.akod AND w.ev = t4a.ev
LEFT JOIN public.t5_a AS t5a
    ON w.akod = t5a.akod AND w.ev = t5a.ev
LEFT JOIN public.t5_b AS t5b
    ON w.akod = t5b.akod AND w.ev = t5b.ev
LEFT JOIN public.t5_c AS t5c
    ON w.akod = t5c.akod AND w.ev = t5c.ev
LEFT JOIN public.t7_c AS t7c
    ON w.akod = t7c.akod AND w.ev = t7c.ev
LEFT JOIN public.t7_b1 AS t7b1
    ON w.akod = t7b1.akod AND w.ev = t7b1.ev
WHERE
    (w.ev = 2023 AND ts.ste_ev = 2020)
    OR (w.ev = 2024 AND ts.ste_ev = 2020)
ORDER BY w.ev, w.akod;
