from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

YEARS = list(range(2006, 2017))
AGE12 = [
    "15_19",
    "20_24",
    "25_29",
    "30_34",
    "35_39",
    "40_44",
    "45_49",
    "50_54",
    "55_59",
    "60_64",
    "65_69",
    "70_plus",
]
INDUSTRIES = [i for i in range(1, 19) if i != 6]
ARCHETYPES = {
    0: "native_low_skill",
    1: "native_medium_skill",
    2: "native_high_skill",
    3: "immigrant_recent_low",
    4: "immigrant_recent_high",
    5: "immigrant_mid_low",
    6: "immigrant_mid_high",
    7: "immigrant_established_low",
    8: "immigrant_established_high",
}
SECTORS = {0: "goods_producing", 1: "services"}
GOODS = {1, 2, 3, 4, 5}
CMA = 535


def industry_sector(ind: int) -> int:
    return 0 if ind in GOODS else 1


def make_parameters(out: Path, rng: np.random.Generator) -> None:
    p = out / "Data" / "Simulation" / "Parameters"
    p.mkdir(parents=True, exist_ok=True)
    ur = 0.065 + 0.02 * np.exp(-0.5 * (np.arange(len(YEARS)) - 3) ** 2)
    pd.DataFrame(
        {
            "year": YEARS,
            "unemployment_rate": np.round(ur, 4),
            "employment_rate": np.round(0.62 - 0.3 * (ur - 0.065), 4),
            "participation_rate": np.round(0.67 + rng.normal(0, 0.002, len(YEARS)), 4),
        }
    ).to_csv(p / "labor_force_summary_2006_2016.csv", index=False, lineterminator="\n")
    base_jfr = np.linspace(0.2, 0.05, len(AGE12))
    rows = [
        {
            "year": y,
            "age_group": ag,
            "monthly_job_finding_rate": round(float(max(0.01, b + rng.normal(0, 0.01))), 4),
        }
        for y in YEARS
        for ag, b in zip(AGE12, base_jfr)
    ]
    pd.DataFrame(rows).to_csv(
        p / "job_finding_rates_2006_2016.csv", index=False, lineterminator="\n"
    )
    rows = []
    for y in YEARS:
        for ind in INDUSTRIES:
            sep = float(np.clip(0.18 + rng.normal(0, 0.03), 0.08, 0.35))
            rows.append(
                {
                    "year": y,
                    "naics_18": ind,
                    "separation_rate": round(sep, 4),
                    "layoff_rate": round(sep * float(rng.uniform(0.3, 0.6)), 4),
                }
            )
    pd.DataFrame(rows).to_csv(
        p / "separation_rates_2006_2016.csv", index=False, lineterminator="\n"
    )
    ind_wage = {ind: float(rng.uniform(16, 34)) for ind in INDUSTRIES}
    rows = []
    for k, y in enumerate(YEARS):
        for ind in INDUSTRIES:
            rows.append(
                {
                    "year": y,
                    "naics_18": ind,
                    "avg_hourly_wage": round(ind_wage[ind] * 1.02**k, 2),
                    "wage_growth_rate": np.nan if y == 2006 else 0.02,
                }
            )
    pd.DataFrame(rows).to_csv(p / "wage_parameters_2006_2016.csv", index=False, lineterminator="\n")
    ages = np.arange(0, 111)
    qx = np.clip(0.0002 * np.exp(0.085 * ages), 0, 0.7)
    pd.DataFrame({"age": ages, "qx": np.round(qx, 6)}).to_csv(
        p / "mortality_rates.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(
        {
            "year": YEARS,
            "immigrants": rng.integers(95000, 130000, len(YEARS)),
            "emigrants": rng.integers(18000, 30000, len(YEARS)),
        }
    ).to_csv(p / "immigration_emigration_ontario.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        {"year": YEARS, "gtha_population_share": np.round(np.linspace(0.475, 0.5, len(YEARS)), 4)}
    ).to_csv(p / "gtha_share.csv", index=False, lineterminator="\n")
    ages = np.arange(15, 101)
    ret = np.where(
        ages < 55,
        0.001,
        np.where(ages < 65, 0.02 + 0.006 * (ages - 55), np.clip(0.2 + 0.01 * (ages - 65), 0, 0.6)),
    )
    pd.DataFrame({"age": ages, "retirement_rate": np.round(ret, 4)}).to_csv(
        p / "retirement_rates.csv", index=False, lineterminator="\n"
    )
    jf_mult = np.ones(len(YEARS))
    sep_mult = np.ones(len(YEARS))
    for k, y in enumerate(YEARS):
        if y in (2008, 2009):
            jf_mult[k], sep_mult[k] = (0.85, 1.25)
    pd.DataFrame(
        {"year": YEARS, "job_finding_multiplier": jf_mult, "separation_multiplier": sep_mult}
    ).to_csv(p / "recession_multipliers.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        {
            "naics_18": INDUSTRIES,
            "prob_lt20": np.round(rng.uniform(0.55, 0.8, len(INDUSTRIES)), 3),
            "prob_20_99": np.round(rng.uniform(0.1, 0.3, len(INDUSTRIES)), 3),
        }
    ).to_csv(p / "firm_size_probabilities.csv", index=False, lineterminator="\n")

    def _nilf_shares(ag: str) -> dict:
        i = AGE12.index(ag)
        if i <= 1:
            w = np.array([0.02, 0.85, 0.04, 0.09])
        elif i <= 7:
            w = np.array([0.08, 0.12, 0.3, 0.5])
        elif i <= 9:
            w = np.array([0.55, 0.02, 0.28, 0.15])
        else:
            w = np.array([0.9, 0.005, 0.075, 0.02])
        w = np.clip(w + rng.normal(0, 0.01, 4), 0.001, None)
        w = w / w.sum()
        return {
            "pct_retirement": round(100 * w[0], 2),
            "pct_school": round(100 * w[1], 2),
            "pct_illness": round(100 * w[2], 2),
            "pct_family": round(100 * w[3], 2),
        }

    rows_sex, rows_plain = ([], [])
    for y in YEARS:
        for ag in AGE12:
            sh = _nilf_shares(ag)
            rows_plain.append({"year": y, "age_group": ag, **sh})
            for sex in ("Male", "Female"):
                rows_sex.append(
                    {
                        "year": y,
                        "age_group": ag,
                        "sex": sex,
                        "nilf_total": int(rng.integers(20000, 90000)),
                        **sh,
                    }
                )
    pd.DataFrame(rows_plain).to_csv(
        p / "nilf_composition_2006_2016.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(rows_sex).to_csv(
        p / "nilf_composition_by_sex_2006_2016.csv", index=False, lineterminator="\n"
    )
    rows = []
    for y in YEARS:
        for ag in AGE12:
            i = AGE12.index(ag)
            rows.append(
                {
                    "year": y,
                    "age_group": ag,
                    "discouragement_rate": round(
                        float(np.clip(0.03 + 0.004 * i + rng.normal(0, 0.003), 0.005, 0.15)), 4
                    ),
                    "retirement_share": round(float(0.02 if i < 8 else 0.15 + 0.2 * (i - 8)), 4),
                }
            )
    pd.DataFrame(rows).to_csv(
        p / "labor_force_transitions_2006_2016.csv", index=False, lineterminator="\n"
    )


EXTENSION_YEARS = list(range(2017, 2026))


def make_extension(out: Path, rng: np.random.Generator) -> None:
    p = out / "Data" / "Simulation" / "Parameters"
    base_jfr = np.linspace(0.2, 0.05, len(AGE12))
    rows = [
        {
            "year": y,
            "age_group": ag,
            "monthly_job_finding_rate": round(float(max(0.01, b + rng.normal(0, 0.01))), 4),
        }
        for y in EXTENSION_YEARS
        for ag, b in zip(AGE12, base_jfr)
    ]
    pd.DataFrame(rows).to_csv(
        p / "job_finding_rates_2017_2025.csv", index=False, lineterminator="\n"
    )
    rows = []
    for y in EXTENSION_YEARS:
        for ind in INDUSTRIES:
            sep = float(np.clip(0.18 + rng.normal(0, 0.03), 0.08, 0.35))
            rows.append(
                {
                    "year": y,
                    "naics_18": ind,
                    "separation_rate": round(sep, 4),
                    "layoff_rate": round(sep * float(rng.uniform(0.3, 0.6)), 4),
                }
            )
    pd.DataFrame(rows).to_csv(
        p / "separation_rates_2017_2025.csv", index=False, lineterminator="\n"
    )
    rows_sex, rows_plain, rows_flow = ([], [], [])
    for y in EXTENSION_YEARS:
        for i, ag in enumerate(AGE12):
            if i <= 1:
                w = np.array([0.02, 0.85, 0.04, 0.09])
            elif i <= 7:
                w = np.array([0.08, 0.12, 0.3, 0.5])
            elif i <= 9:
                w = np.array([0.55, 0.02, 0.28, 0.15])
            else:
                w = np.array([0.9, 0.005, 0.075, 0.02])
            w = np.clip(w + rng.normal(0, 0.01, 4), 0.001, None)
            w = w / w.sum()
            sh = {
                "pct_retirement": round(100 * w[0], 2),
                "pct_school": round(100 * w[1], 2),
                "pct_illness": round(100 * w[2], 2),
                "pct_family": round(100 * w[3], 2),
            }
            rows_plain.append({"year": y, "age_group": ag, **sh})
            for sex in ("Male", "Female"):
                rows_sex.append(
                    {
                        "year": y,
                        "age_group": ag,
                        "sex": sex,
                        "nilf_total": int(rng.integers(20000, 90000)),
                        **sh,
                    }
                )
            rows_flow.append(
                {
                    "year": y,
                    "age_group": ag,
                    "discouragement_rate": round(
                        float(np.clip(0.03 + 0.004 * i + rng.normal(0, 0.003), 0.005, 0.15)), 4
                    ),
                    "retirement_share": round(float(0.02 if i < 8 else 0.15 + 0.2 * (i - 8)), 4),
                }
            )
    pd.DataFrame(rows_plain).to_csv(
        p / "nilf_composition_2017_2025.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(rows_sex).to_csv(
        p / "nilf_composition_by_sex_2017_2025.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(rows_flow).to_csv(
        p / "labor_force_transitions_2017_2025.csv", index=False, lineterminator="\n"
    )


def make_da_centroids(out: Path, rng: np.random.Generator, n_das: int) -> np.ndarray:
    p = out / "Data" / "Simulation" / "Parameters"
    p.mkdir(parents=True, exist_ok=True)
    da_ids = np.arange(35200001, 35200001 + n_das)
    lat = 43.7 + rng.uniform(-0.25, 0.25, n_das)
    lon = -79.4 + rng.uniform(-0.45, 0.45, n_das)
    pd.DataFrame(
        {"da_id": da_ids, "lat": np.round(lat, 6), "lon": np.round(lon, 6), "cma": CMA}
    ).to_csv(p / "da_centroids_gtha.csv", index=False, lineterminator="\n")
    return da_ids


def make_population(out: Path, rng: np.random.Generator, n: int, da_ids: np.ndarray) -> None:
    d = out / "Data_Agents" / "Microsim_Output"
    d.mkdir(parents=True, exist_ok=True)
    age = 15 + np.arange(n) % 71
    sex = rng.choice(["Male", "Female"], n)
    education = rng.integers(1, 7, n)
    skill = np.clip(1 + (education - 1) // 2 + rng.integers(-1, 2, n), 1, 4)
    p_nilf = np.clip(0.12 + 0.9 * np.maximum(0, age - 55) / 45.0, 0.1, 0.95)
    u = rng.random(n)
    status = np.where(u < p_nilf, "NILF", np.where(u < p_nilf + 0.05, "Unemployed", "Employed"))
    industry = rng.choice(INDUSTRIES, n).astype(float)
    industry[status == "NILF"] = np.nan
    wage = np.round(np.exp(rng.normal(np.log(22) + 0.08 * (skill - 2.5), 0.3)), 2)
    wage[status != "Employed"] = 0.0
    immigrant = rng.random(n) < 0.45
    ysa = np.where(immigrant, rng.integers(0, 40, n), 0)
    arch = np.where(
        ~immigrant,
        np.clip(skill - 1, 0, 2),
        3 + 2 * np.minimum(ysa // 10, 2) + (skill >= 3).astype(int),
    )
    df = pd.DataFrame(
        {
            "worker_id": [f"W{100000 + i}" for i in range(n)],
            "age": age,
            "sex": sex,
            "education": education,
            "skill_level": skill,
            "employment_status": status,
            "naics_18": industry,
            "hourly_wage": wage,
            "tenure_months": np.where(status == "Employed", rng.integers(1, 240, n), 0),
            "unemployment_duration_months": np.where(
                status == "Unemployed", rng.integers(1, 24, n), 0
            ),
            "hours_per_week": np.where(status == "Employed", rng.choice([35, 37.5, 40], n), 0),
            "immigrant_status": immigrant,
            "years_since_arrival": ysa,
            "archetype_id": arch.astype(int),
            "archetype_name": [ARCHETYPES[int(a)] for a in arch],
            "da_id": rng.choice(da_ids, n),
            "cma": CMA,
        }
    )
    df.to_csv(d / "population_2006.csv", index=False, lineterminator="\n")


def make_firms(out: Path, rng: np.random.Generator, n: int, da_ids: np.ndarray) -> None:
    d = out / "Data_Agents" / "firm_population"
    d.mkdir(parents=True, exist_ok=True)
    inds = [i for i in INDUSTRIES for _ in range(18 if i == 5 else 3)]
    inds += list(rng.choice(INDUSTRIES, max(0, n - len(inds))))
    inds = np.array(inds[: max(n, len(inds))])
    size = np.maximum(1, rng.lognormal(np.log(12), 0.9, len(inds))).astype(int)
    df = pd.DataFrame(
        {
            "firm_id": [f"F{5000 + i}" for i in range(len(inds))],
            "naics_18": inds,
            "estimated_employees": size,
            "target_size": size,
            "archetype_id": rng.integers(0, 12, len(inds)),
            "archetype_name": [f"firm_arch_{a}" for a in rng.integers(0, 12, len(inds))],
            "sector_id": [industry_sector(int(i)) for i in inds],
            "sector_name": [SECTORS[industry_sector(int(i))] for i in inds],
            "da_id": rng.choice(da_ids, len(inds)),
            "cma": CMA,
            "is_synthetic": "true",
        }
    )
    df.to_csv(d / "firms_2006.csv", index=False, lineterminator="\n")
    (d / "firm_location_provenance.json").write_text(
        json.dumps(
            {
                "note": "SYNTHETIC demo data — no repair performed; all firm DAs generated directly against the synthetic centroid table.",
                "n_firms": int(len(inds)),
                "source": "make_synthetic_data.py",
            },
            indent=2,
        ),
        newline="\n",
    )


def make_model_inputs(out, rng, n_workers, n_firms, da_ids):
    from hashlib import sha256

    params = out / "Data/Simulation/Parameters"
    workplace = params / "workplace"
    workplace.mkdir(exist_ok=True)
    cmas = np.array([532, 535, 537])
    cma_by_zone = dict(zip(da_ids, np.resize(cmas, len(da_ids))))
    for name, file, population in (
        ("worker_id", out / "Data_Agents/Microsim_Output/population_2006.csv", True),
        ("firm_id", out / "Data_Agents/firm_population/firms_2006.csv", False),
    ):
        table = pd.read_csv(file)
        cma_index = np.arange(len(table)) // 71 % 3 if population else np.arange(len(table)) % 3
        table["cma"] = cmas[cma_index]
        for cma in cmas:
            mask = table.cma == cma
            zones = [zone for zone in da_ids if cma_by_zone[zone] == cma]
            table.loc[mask, "da_id"] = rng.choice(zones, size=int(mask.sum()))
        if population:
            table["weight"] = 1.0
        else:
            table = table.drop(columns=["target_size"])
            prefix = dict(
                zip(
                    INDUSTRIES, [11, 21, 22, 23, 31, 41, 44, 48, 52, 54, 56, 61, 62, 51, 72, 81, 91]
                )
            )
            table["naics_6digit"] = [prefix[int(i)] * 10000 + 1100 for i in table.naics_18]
            manufacturers = table.index[table.naics_18 == 5]
            table.loc[manufacturers, "naics_6digit"] = np.resize(
                np.repeat([336110, 336210, 336310, 331110, 332110, 311110], 3), len(manufacturers)
            )
            table["archetype_name"] = ["firm_arch_" + str(value) for value in table.archetype_id]
        table.to_csv(file, index=False, lineterminator="\n")
    centroids = pd.read_csv(params / "da_centroids_gtha.csv")
    centroids["cma"] = centroids.da_id.map(cma_by_zone)
    centres = {532: (43.9, -78.9), 535: (43.7, -79.4), 537: (43.3, -79.9)}
    for cma, (lat, lon) in centres.items():
        selected = centroids.cma == cma
        centroids.loc[selected, "lat"] = lat + rng.uniform(-0.08, 0.08, selected.sum())
        centroids.loc[selected, "lon"] = lon + rng.uniform(-0.1, 0.1, selected.sum())
    centroids.to_csv(params / "da_centroids_gtha.csv", index=False, lineterminator="\n")
    cohorts = [
        (year, cma, age, round((1200 + 14 * (year - 2006)) / (6 if age >= 85 else 1)), "July 1")
        for year in range(2006, 2031)
        for cma in cmas
        for age in range(91)
    ]
    pd.DataFrame(cohorts, columns=["year", "cma", "age", "population", "reference_date"]).to_csv(
        params / "cohort_population.csv", index=False, lineterminator="\n"
    )
    industries = [
        "11",
        "21",
        "22",
        "23",
        "31-33",
        "41",
        "44-45",
        "48-49",
        "52",
        "54",
        "56",
        "61",
        "62",
        "51",
        "72",
        "81",
        "91",
    ]
    pairs = [(cma, i) for cma in cmas for i in industries]
    pd.DataFrame(
        [(c, i, "usual", "all", 100) for c, i in pairs],
        columns=["cma", "industry", "workplace_scope", "worker_class", "employed_people"],
    ).to_csv(workplace / "cma_workplace_industry_2006.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [(c, d, i, 80 if c == d else 10) for c, i in pairs for d in cmas]
        + [(c, "OUTSIDE_3_CMA", i, 5) for c, i in pairs]
        + [("OUTSIDE_3_CMA", c, i, 5) for c, i in pairs],
        columns=["residence_cma", "workplace_cma", "industry", "employed_people"],
    ).to_csv(workplace / "commuting_cma_industry_2006.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [
            (c, i, "all_employed", kind, qty)
            for c, i in pairs
            for kind, qty in [
                ("all", 100),
                ("wage_earner", 90),
                ("self_employed", 10),
                ("unpaid_family", 0),
            ]
        ],
        columns=["cma", "industry", "place_of_work", "worker_class", "employed_people"],
    ).to_csv(workplace / "resident_industry_class_place_2006.csv", index=False, lineterminator="\n")
    projection = params / "net_projection.csv"
    pd.DataFrame(
        [(y, 0 if y == 2031 else 300, "persons_age_15plus", y == 2031) for y in range(2026, 2032)],
        columns=["year", "net_people", "units", "terminal_stub"],
    ).to_csv(projection, index=False, lineterminator="\n")
    projection.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "net_population_projection_v1",
                "entrant_profile_transfer": "hold_registered_2022_2025_profile",
                "timing": "ending_year_january_december",
                "start_year": 2026,
                "end_year": 2030,
                "schedule_sha256": sha256(projection.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="An empty destination for the synthetic input tables",
    )
    ap.add_argument("--n-workers", type=int, default=9000)
    ap.add_argument("--n-firms", type=int, default=300)
    ap.add_argument("--n-das", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        ap.error("Choose a new or empty destination for the generated inputs.")
    if args.n_workers < 213 or args.n_firms < 66 or args.n_das < 3:
        ap.error("Use at least 213 workers, 66 firms and 3 geographic zones.")
    rng = np.random.default_rng(args.seed)
    out = args.out
    make_parameters(out, rng)
    da_ids = make_da_centroids(out, rng, args.n_das)
    make_population(out, rng, args.n_workers, da_ids)
    make_firms(out, rng, args.n_firms, da_ids)
    make_model_inputs(out, rng, args.n_workers, args.n_firms, da_ids)
    make_extension(out, np.random.default_rng([args.seed, 2017]))
    manifest = {
        "generator": "synthetic_data/make_synthetic_data.py",
        "seed": args.seed,
        "n_workers": args.n_workers,
        "n_firms": args.n_firms,
        "n_das": args.n_das,
        "disclaimer": "ALL VALUES SYNTHETIC. Not derived from any Statistics Canada record. Pipeline demo only; does not reproduce paper results.",
    }
    (out / "SYNTHETIC_DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2), newline="\n")
    print(f"Synthetic dataset written under {out}")
    print("  Data/Simulation/Parameters/")
    print("  Data_Agents/Microsim_Output/population_2006.csv")
    print("  Data_Agents/firm_population/firms_2006.csv")


if __name__ == "__main__":
    main()
