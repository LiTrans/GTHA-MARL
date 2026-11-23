import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoadedData:
    population: pd.DataFrame
    labor_force_summary: pd.DataFrame
    job_finding_rates: pd.DataFrame
    separation_rates: pd.DataFrame
    wage_parameters: pd.DataFrame
    mortality_rates: pd.DataFrame
    immigration_emigration: pd.DataFrame
    gtha_share: pd.DataFrame
    retirement_rates: pd.DataFrame
    recession_multipliers: pd.DataFrame
    firm_size_probabilities: pd.DataFrame
    nilf_composition: pd.DataFrame
    nilf_composition_by_sex: pd.DataFrame
    labor_force_transitions: pd.DataFrame
    da_centroids: pd.DataFrame = None
    employment_by_industry: Dict[int, int] = None
    firms: pd.DataFrame = None

    def __post_init__(self):
        if self.employment_by_industry is None:
            self.employment_by_industry = self._compute_employment_by_industry()

    def _compute_employment_by_industry(self) -> Dict[int, int]:
        employed = self.population[self.population["employment_status"] == "Employed"]
        emp_by_ind = employed.groupby("naics_18").size().to_dict()
        return {int(k): v for k, v in emp_by_ind.items() if not pd.isna(k)}


def load_csv(filepath: Path, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Required file not found: {filepath}")
    df = pd.read_csv(filepath)
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"{filepath.name}: missing required columns {sorted(missing)}. Present columns: {sorted(df.columns.tolist())}."
            )
    return df


def load_csv_with_extension(
    filepath: Path, required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    df = load_csv(filepath, required_columns=required_columns)
    if "_2006_2016" not in filepath.name:
        return df
    ext_path = filepath.parent / filepath.name.replace("_2006_2016", "_2017_2025")
    if not ext_path.exists():
        return df
    df_ext = pd.read_csv(ext_path)
    common = [c for c in df.columns if c in df_ext.columns]
    missing_ext_cols = set(df.columns) - set(df_ext.columns)
    if missing_ext_cols:
        logger.warning(
            f"The extension table {ext_path.name} lacks these columns of the main table: {missing_ext_cols}"
        )
    combined = pd.concat([df[common], df_ext[common]], ignore_index=True)
    return combined


def load_all_data(config) -> LoadedData:
    data_dir = config.data.data_dir
    population_path = config.data.get_population_path()
    population = load_csv(
        population_path,
        required_columns=[
            "worker_id",
            "age",
            "sex",
            "education",
            "employment_status",
            "naics_18",
            "hourly_wage",
            "archetype_id",
            "archetype_name",
            "da_id",
            "cma",
        ],
    )
    _nan_da = int(population["da_id"].isna().sum())
    _nan_cma = int(population["cma"].isna().sum())
    if _nan_da or _nan_cma:
        raise ValueError(
            f"{population_path.name}: {_nan_da} workers have no da_id and {_nan_cma} have no cma. Both are required for the distance lookup."
        )
    if "naics_18" in population.columns:
        n_worker_ind_6 = (population["naics_18"] == 6).sum()
        if n_worker_ind_6 > 0:
            population.loc[population["naics_18"] == 6, "naics_18"] = 5
            logger.warning(
                f"Moved {n_worker_ind_6} workers from industry 6 to industry 5 (manufacturing)"
            )
    labor_force_summary = load_csv(
        data_dir / "labor_force_summary_2006_2016.csv",
        required_columns=["year", "unemployment_rate", "employment_rate"],
    )
    job_finding_rates = load_csv_with_extension(
        data_dir / "job_finding_rates_2006_2016.csv",
        required_columns=["year", "age_group", "monthly_job_finding_rate"],
    )
    separation_rates = load_csv_with_extension(
        data_dir / "separation_rates_2006_2016.csv",
        required_columns=["year", "naics_18", "separation_rate"],
    )
    wage_parameters = load_csv(
        data_dir / "wage_parameters_2006_2016.csv",
        required_columns=["year", "naics_18", "avg_hourly_wage"],
    )
    mortality_rates = load_csv(data_dir / "mortality_rates.csv", required_columns=["age", "qx"])
    immigration_emigration = load_csv(
        data_dir / "immigration_emigration_ontario.csv",
        required_columns=["year", "immigrants", "emigrants"],
    )
    gtha_share = load_csv(
        data_dir / "gtha_share.csv", required_columns=["year", "gtha_population_share"]
    )
    retirement_rates = load_csv(
        data_dir / "retirement_rates.csv", required_columns=["age", "retirement_rate"]
    )
    recession_multipliers = load_csv(
        data_dir / "recession_multipliers.csv",
        required_columns=["year", "job_finding_multiplier", "separation_multiplier"],
    )
    firm_size_probabilities = load_csv(
        data_dir / "firm_size_probabilities.csv",
        required_columns=["naics_18", "prob_lt20", "prob_20_99"],
    )
    nilf_composition = load_csv_with_extension(
        data_dir / "nilf_composition_2006_2016.csv", required_columns=["year", "age_group"]
    )
    nilf_composition_by_sex = load_csv_with_extension(
        data_dir / "nilf_composition_by_sex_2006_2016.csv",
        required_columns=[
            "year",
            "age_group",
            "sex",
            "nilf_total",
            "pct_retirement",
            "pct_school",
            "pct_illness",
            "pct_family",
        ],
    )
    labor_force_transitions = load_csv_with_extension(
        data_dir / "labor_force_transitions_2006_2016.csv", required_columns=["year", "age_group"]
    )
    da_centroids = load_csv(
        data_dir / "da_centroids_gtha.csv", required_columns=["da_id", "lat", "lon"]
    )
    firms_path = config.data.get_firms_path()
    firms = load_csv(
        firms_path,
        required_columns=[
            "firm_id",
            "naics_18",
            "estimated_employees",
            "archetype_id",
            "archetype_name",
            "sector_id",
            "sector_name",
            "da_id",
            "cma",
        ],
    )
    if firms["da_id"].isna().any() or firms["cma"].isna().any():
        raise ValueError(f"{firms_path.name}: firm DA/CMA geography contains nulls")
    valid_da_ids = set(da_centroids["da_id"].astype(int))
    invalid_da = ~firms["da_id"].astype(int).isin(valid_da_ids)
    if invalid_da.any():
        examples = firms.loc[invalid_da, "da_id"].head(5).tolist()
        raise ValueError(f"{firms_path.name}: firm DA absent from centroid table: {examples}")
    n_industry_6 = (firms["naics_18"] == 6).sum()
    if n_industry_6 > 0:
        firms.loc[firms["naics_18"] == 6, "naics_18"] = 5
    return LoadedData(
        population=population,
        labor_force_summary=labor_force_summary,
        job_finding_rates=job_finding_rates,
        separation_rates=separation_rates,
        wage_parameters=wage_parameters,
        mortality_rates=mortality_rates,
        immigration_emigration=immigration_emigration,
        gtha_share=gtha_share,
        retirement_rates=retirement_rates,
        recession_multipliers=recession_multipliers,
        firm_size_probabilities=firm_size_probabilities,
        nilf_composition=nilf_composition,
        nilf_composition_by_sex=nilf_composition_by_sex,
        labor_force_transitions=labor_force_transitions,
        da_centroids=da_centroids,
        firms=firms,
    )


def sample_population(
    population: pd.DataFrame,
    n_workers: int,
    stratify_by: List[str] = ["archetype_id"],
    seed: int = 42,
) -> pd.DataFrame:
    if n_workers >= len(population):
        logger.warning(f"Requested {n_workers} workers but only {len(population)} available")
        return population.copy()
    if "worker_id" in population.columns:
        _stable_id_col = "worker_id"
    else:
        population = population.copy()
        population["_sample_uid"] = np.arange(len(population), dtype=np.int64)
        _stable_id_col = "_sample_uid"
    if "archetype_id" in population.columns:
        frac = n_workers / len(population)
        sampled_parts = [
            grp.sample(frac=frac, random_state=seed)
            for _, grp in population.groupby("archetype_id")
        ]
        sampled = pd.concat(sampled_parts, ignore_index=True)
        if len(sampled) < n_workers:
            _already = set(sampled[_stable_id_col].tolist())
            additional = population[~population[_stable_id_col].isin(_already)]
            n_additional = min(n_workers - len(sampled), len(additional))
            if n_additional > 0:
                sampled = pd.concat(
                    [sampled, additional.sample(n=n_additional, random_state=seed + 1)],
                    ignore_index=True,
                )
        elif len(sampled) > n_workers:
            sampled = sampled.sample(n=n_workers, random_state=seed + 2)
        _n_sampled = len(sampled)
        _n_unique = sampled[_stable_id_col].nunique()
        assert (
            _n_unique == _n_sampled
        ), f"Sampling produced duplicates: sampled={_n_sampled}, unique_{_stable_id_col}={_n_unique}"
        archetype_counts = sampled["archetype_id"].value_counts().sort_index()
        for arch_id in range(12):
            if not arch_id in archetype_counts.index:
                logger.warning(f"  Worker group {arch_id} has no workers")
        if _stable_id_col == "_sample_uid":
            sampled = sampled.drop(columns=["_sample_uid"])
        return sampled
    logger.warning(
        "The population file has no archetype_id column. Workers are drawn by simple random sampling."
    )
    sampled = population.sample(n=n_workers, random_state=seed)
    _n_unique = sampled[_stable_id_col].nunique()
    assert _n_unique == len(
        sampled
    ), f"Sampling produced duplicates: sampled={len(sampled)}, unique_{_stable_id_col}={_n_unique}"
    if _stable_id_col == "_sample_uid":
        sampled = sampled.drop(columns=["_sample_uid"])
    return sampled


def prepare_tensors(
    population: pd.DataFrame, firms: pd.DataFrame, device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    tensors = {}
    worker_features = []
    for _, row in population.iterrows():
        features = [
            row["age"] / 100.0,
            1.0 if row["sex"] == "Male" else 0.0,
            row["education"] / 6.0,
            row.get("skill_level", 2) / 4.0,
            row.get("hourly_wage", 20.0) / 50.0,
            row.get("tenure_months", 0) / 120.0,
            row.get("experience_total", 0) / 40.0,
        ]
        status = row["employment_status"]
        features.extend(
            [
                1.0 if status == "Employed" else 0.0,
                1.0 if status == "Unemployed" else 0.0,
                1.0 if status == "NILF" else 0.0,
            ]
        )
        from config.default_config import NAICS_TO_SECTOR

        naics = row.get("naics_18", 0)
        sector = NAICS_TO_SECTOR.get(int(naics) if not pd.isna(naics) else 0, 0)
        sector_onehot = [0.0] * 6
        sector_onehot[sector] = 1.0
        features.extend(sector_onehot)
        worker_features.append(features)
    tensors["worker_features"] = torch.tensor(worker_features, dtype=torch.float32, device=device)
    firm_features = []
    for _, row in firms.iterrows():
        n_emp = row.get("estimated_employees", row.get("n_employees", 10))
        features = [
            n_emp / 500.0,
            row.get("naics_18", 0) / 18.0,
            row.get("union_status", 0),
            row.get("public_sector", 0),
            row.get("multi_location", 0),
            row.get("archetype_id", 0) / 5.0,
        ]
        firm_features.append(features)
    tensors["firm_features"] = torch.tensor(firm_features, dtype=torch.float32, device=device)
    return tensors


class DataManager:

    def __init__(self, config):
        self.config = config
        self.data: Optional[LoadedData] = None
        self.sampled_population: Optional[pd.DataFrame] = None
        self.firms: Optional[pd.DataFrame] = None
        self.tensors: Optional[Dict[str, torch.Tensor]] = None

    def load(self) -> "DataManager":
        self.data = load_all_data(self.config)
        return self

    def prepare(self) -> "DataManager":
        if self.data is None:
            raise RuntimeError("Must call load() before prepare()")
        population_pool = self.data.population
        firms_pool = self.data.firms
        from age_renewal import enabled as _age_renewal_enabled, initial_population

        if _age_renewal_enabled():
            self.sampled_population = initial_population(
                population_pool, self.config.simulation.n_workers, self.config.seed
            )
        else:
            self.sampled_population = sample_population(
                population_pool, n_workers=self.config.simulation.n_workers, seed=self.config.seed
            )
        from market_snapshots import select_initial_firms

        self.firms = select_initial_firms(firms_pool, self.config)
        self._ensure_worker_industry_coverage()
        return self

    def _ensure_worker_industry_coverage(self):
        if self.sampled_population is None or self.firms is None:
            return
        worker_industries = set(
            (x for x in self.sampled_population["naics_18"].unique() if pd.notna(x))
        )
        firm_industries = set((x for x in self.firms["naics_18"].unique() if pd.notna(x)))
        missing_industries = worker_industries - firm_industries
        if not missing_industries:
            return
        logger.warning(
            f"{len(missing_industries)} worker industries have no firms: {sorted(missing_industries)}"
        )
        added_firms = []
        from config.default_config import NAICS_TO_SECTOR

        sector_to_industries = {}
        for ind, sec in NAICS_TO_SECTOR.items():
            if sec not in sector_to_industries:
                sector_to_industries[sec] = []
            sector_to_industries[sec].append(ind)
        for industry in missing_industries:
            industry_firms = self.data.firms[self.data.firms["naics_18"] == industry]
            if len(industry_firms) > 0:
                firm_to_add = industry_firms.sample(
                    n=1, random_state=self.config.seed + int(industry)
                )
                added_firms.append(firm_to_add)
            else:
                sector = NAICS_TO_SECTOR.get(industry, 0)
                related_industries = [
                    ind
                    for ind in sector_to_industries.get(sector, [])
                    if ind in firm_industries and ind != industry
                ]
                if related_industries:
                    target_industry = related_industries[0]
                    n_reassigned = len(
                        self.sampled_population[self.sampled_population["naics_18"] == industry]
                    )
                    self.sampled_population.loc[
                        self.sampled_population["naics_18"] == industry, "naics_18"
                    ] = target_industry
                    logger.warning(
                        f"  Industry {industry} has no firms. Reassigned {n_reassigned} workers to industry {target_industry} (same sector {sector})"
                    )
                else:
                    most_common = self.firms["naics_18"].mode().iloc[0]
                    n_reassigned = len(
                        self.sampled_population[self.sampled_population["naics_18"] == industry]
                    )
                    self.sampled_population.loc[
                        self.sampled_population["naics_18"] == industry, "naics_18"
                    ] = most_common
                    logger.warning(
                        f"  Industry {industry} has no firms and no related industry. Reassigned {n_reassigned} workers to industry {most_common}"
                    )
        if added_firms:
            self.firms = pd.concat([self.firms] + added_firms, ignore_index=True)

    @property
    def full_firms_pool(self) -> Optional[pd.DataFrame]:
        if self.data is not None:
            pool = self.data.firms
            return pool
        return None

    def to_tensors(self, device: str = None) -> Dict[str, torch.Tensor]:
        if self.sampled_population is None:
            raise RuntimeError("Must call prepare() before to_tensors()")
        device = device or self.config.training.device
        self.tensors = prepare_tensors(self.sampled_population, self.firms, device=device)
        return self.tensors
