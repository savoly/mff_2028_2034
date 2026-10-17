import json
from dataclasses import asdict
from math import isfinite

from mff.scenario import ScenarioConfig

FORMAT_VERSION = 1


def serialize_scenarios(scenarios: dict[str, ScenarioConfig]) -> str:
    return json.dumps(
        {
            "version": FORMAT_VERSION,
            "scenarios": {name: asdict(config) for name, config in scenarios.items()},
        },
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def deserialize_scenarios(content: str) -> dict[str, ScenarioConfig]:
    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
        raise ValueError("Ismeretlen forgatókönyv-fájlverzió.")
    entries = payload.get("scenarios")
    if not isinstance(entries, dict) or len(entries) > 100:
        raise ValueError("A forgatókönyvek formátuma vagy száma hibás.")

    fields = {
        "year",
        "dabis_per_ha",
        "yfs_per_ha",
        "yfs_max_ha",
        "redistributive_per_ha",
        "thresholds",
        "rates",
        "payment_cap",
        "exclude_young_farmer",
    }

    result = {}
    for name, raw in entries.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise ValueError("Érvénytelen forgatókönyvnév.")
        if not isinstance(raw, dict) or set(raw) != fields:
            raise ValueError(f"Hiányzó vagy ismeretlen mező: {name}")
        obj = raw.copy()
        for key, size in (
            ("redistributive_per_ha", 2),
            ("thresholds", 3),
            ("rates", 3),
        ):
            val = obj[key]
            if not isinstance(val, (tuple, list)) or len(val) != size:
                raise ValueError(f"Hibás {key}: {name}")
            obj[key] = tuple(val)
        for key in ("dabis_per_ha", "yfs_per_ha", "yfs_max_ha"):
            if (
                not isinstance(obj[key], (int, float))
                or isinstance(obj[key], bool)
                or not isfinite(obj[key])
            ):
                raise ValueError(f"Hibás {key}: {name}")
        for key in ("redistributive_per_ha", "thresholds", "rates"):
            if any(
                not isinstance(v, (int, float))
                or isinstance(v, bool)
                or not isfinite(v)
                for v in obj[key]
            ):
                raise ValueError(f"Hibás {key}: {name}")
        cap = obj["payment_cap"]
        if cap is not None and (
            not isinstance(cap, (int, float))
            or isinstance(cap, bool)
            or not isfinite(cap)
        ):
            raise ValueError(f"Hibás payment_cap: {name}")
        if (
            type(obj["exclude_young_farmer"]) is not bool
            or type(obj["year"]) is not int
        ):
            raise ValueError(f"Hibás év vagy kapcsoló: {name}")
        config = ScenarioConfig(**obj)
        # Sidebar UI constraints; imported state must be displayable by its widgets.
        a, b, c = config.thresholds
        if not (
            config.year in (2024, 2025, 2026)
            and 90 <= config.dabis_per_ha <= 240
            and 0 <= config.yfs_per_ha <= 200
            and 1 <= config.yfs_max_ha <= 10_000
            and 1000 <= a <= 500000
            and 1000 <= b <= 500000
            and 1000 <= c <= 1000000
            and all(0 <= x <= 1 and round(x * 100) == x * 100 for x in config.rates)
            and (cap is None or 1000 <= cap <= 1000000)
            and all(0 <= x <= 500 for x in config.redistributive_per_ha)
        ):
            raise ValueError(f"Nem támogatott sidebar-paraméter: {name}")
        result[name] = config
    return result
