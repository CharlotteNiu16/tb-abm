from __future__ import annotations
import argparse
import logging
import math
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import njit
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from tqdm import trange

# Configure logging
def configure_logging(level: str = "INFO") -> None:
    fmt = "%(levelname)s  %(message)s"
    logging.basicConfig(level=getattr(logging, level), format=fmt)

# Constants
COUNTRY_CODES = {"korea": "KOR", "south_africa": "ZAF"}
COUNTRY_FULLNAME = {"korea": "Republic of Korea", "south_africa": "South Africa"}
FILE_PATTERNS = {
    "contact": "{ISO}*{layer}*2015.csv",
    "population": "population*{ISO}.csv",
    "cases": "observed*{ISO}_2023.csv",
}
AGE_BINS = np.arange(0, 85, 5)
NUM_GROUPS = len(AGE_BINS)

# Country-specific parameters
COUNTRY_PARAMS: Dict[str, Dict] = {
    "korea": {
        "beta": dict(home=0.049 * 0.65, school=0.034 * 0.65, work=0.026 * 0.65, other=0.018 * 0.65),
        "latent_h0": 0.035,
        "latent_k": 0.20,
        "treat_success": 0.88,
        "delay_to_diag": 35,
        "report_rate": 0.88,
        "daily_contacts": 9.2,
        "contact_scale": 0.83,
        "fractions": dict(home=0.22, school=0.19, work=0.10, other=0.49),
        "incidence_100k": 29,
        "prevalence_100k": 44,
        "season_amp": 0.05,
        "season_peak_month": 11,
    },
    "south_africa": {
        "beta": dict(home=0.049 * 0.8, school=0.034 * 0.8, work=0.026 * 0.8, other=0.018 * 0.8),
        "latent_h0": 0.055,
        "latent_k": 0.18,
        "treat_success": 0.77,
        "delay_to_diag": 90,
        "report_rate": 0.78,
        "daily_contacts": 13.0,
        "contact_scale": 1.1,
        "fractions": dict(home=0.22, school=0.19, work=0.10, other=0.49),
        "incidence_100k": 427,
        "prevalence_100k": None,
        "season_amp": 0.25,
        "season_peak_month": 8,
    }
}

# Utility functions for data loading

def read_csv_as_numeric(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, dtype=str)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    return numeric


def load_contact_matrix(data_dir: pathlib.Path, iso: str, layer: str) -> np.ndarray:
    path = data_dir / FILE_PATTERNS["contact"].format(ISO=iso, layer=layer)
    df = read_csv_as_numeric(path)
    if df.empty:
        raise ValueError(f"Empty contact matrix file: {path}")
    row0 = df.notna().any(axis=1).idxmax()
    col0 = df.notna().any(axis=0).idxmax()
    mat = df.iloc[row0:, col0:].values.astype(np.float32)
    if mat.shape != (NUM_GROUPS, NUM_GROUPS):
        raise ValueError(f"Contact matrix shape mismatch: expected {(NUM_GROUPS, NUM_GROUPS)}, got {mat.shape}")
    mask_nan = np.isnan(mat)
    if mask_nan.any():
        row_mean = np.nanmean(mat, axis=1)
        col_mean = np.nanmean(mat, axis=0)
        for i, j in zip(*np.where(mask_nan)):
            candidates = [val for val in (row_mean[i], col_mean[j]) if not np.isnan(val)]
            mat[i, j] = np.mean(candidates) if candidates else 0.0
    if mat.sum() == 0:
        raise ValueError(f"Contact matrix all zeros after fill: {path}")
    return mat


def load_population(data_dir: pathlib.Path, iso: str) -> np.ndarray:
    path = data_dir / FILE_PATTERNS["population"].format(ISO=iso)
    df = pd.read_csv(path)
    for col in ["Pop", "Population", "population"]:
        if col in df.columns:
            arr = df[col].astype(int).values
            break
    else:
        arr = df.select_dtypes(include=np.number).iloc[:, 0].astype(int).values
    if arr.size != NUM_GROUPS:
        raise ValueError(f"Population rows != {NUM_GROUPS}, got {arr.size}")
    return arr


def load_observed_cases(
    data_dir: pathlib.Path,
    iso: str,
    index: pd.DatetimeIndex,
    file_override: Optional[pathlib.Path] = None,
) -> pd.Series:
    if file_override:
        df = pd.read_csv(file_override)
        df.columns = ["Date", "Cases"] if df.shape[1] >= 2 else df.columns[:2]
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Cases"] = pd.to_numeric(df["Cases"], errors='coerce')
    else:
        path = data_dir / FILE_PATTERNS["cases"].format(ISO=iso)
        df = pd.read_csv(path, parse_dates=[0], names=["Date", "Cases"], header=0)
    series = df.set_index("Date")["Cases"].reindex(index)
    return series.fillna(0.0)

# Dataclass for parameters
@dataclass
class TBParams:
    beta: Dict[str, float]
    latent_h0: float
    latent_k: float
    treat_success: float
    contact_mats: Dict[str, np.ndarray]
    daily_contacts: float
    contact_scale: float
    fractions: Dict[str, float]
    report_rate: float
    delay_to_diag: int
    incidence_100k: int
    prevalence_100k: Optional[int]
    season_amp: float
    season_peak_month: int

# Agent representing a person
class Person(Agent):
    def __init__(self, uid: int, model: TBModel, age_group: int, rng: np.random.Generator):
        super().__init__(uid, model)
        self.age_group = age_group
        self.state = "S"  # S, L, A, R, D
        self.inf_time: Optional[int] = None
        self.report_time: Optional[int] = None
        self.places: Dict[str, int] = {}
        self.rng = rng

    def progress_latent(self, current_t: int):
        years = (current_t - self.inf_time) / 365.0
        rate = self.model.params.latent_h0 * math.exp(-self.model.params.latent_k * years)
        daily = 1 - math.exp(-rate / 365.0)
        if self.rng.random() < daily:
            self.state = "A"
            self.model.new_discovery()

    def progress_active(self, current_t: int):
        params = self.model.params
        if self.report_time is None:
            delay = self.rng.poisson(params.delay_to_diag)
            self.report_time = current_t + delay
        if current_t >= self.report_time and self.rng.random() < params.report_rate:
            self.model.new_notification()
        if self.rng.random() < 1 / 180:
            self.state = "R" if self.rng.random() < params.treat_success else "D"

    def step(self):
        t = self.model.current_t
        if self.state == "L":
            self.progress_latent(t)
        elif self.state == "A":
            self.progress_active(t)

# Place where people meet
class Place:
    counter = 0

    def __init__(self, setting: str):
        self.uid = Place.counter
        Place.counter += 1
        self.setting = setting
        self.members: List[int] = []

# Infection kernel compiled by Numba\@njit
def compute_infection_probability(
    sus_ages: np.ndarray,
    contacts: np.ndarray,
    matrix: np.ndarray,
    beta: float,
    prevalence: np.ndarray,
) -> np.ndarray:
    n = sus_ages.size
    result = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        age = sus_ages[idx]
        row = matrix[age]
        total = row.sum()
        if total <= 0:
            continue
        weighted_prev = (row / total) * prevalence
        lam = beta * contacts[idx]
        result[idx] = 1 - math.exp(-lam * weighted_prev.sum())
    return result

# Main model
class TBModel(Model):
    def __init__(
        self,
        params: TBParams,
        sim_pop: int,
        population_dist: np.ndarray,
        data_dir: pathlib.Path,
        country: str,
        start_date: datetime,
        seed_active: int,
        seed_latent: int,
        rng_seed: int,
    ):
        super().__init__()
        configure_logging()
        Place.counter = 0
        self.params = params
        self.current_t = 0
        self.data_dir = data_dir
        self.country = country
        self.start_date = start_date
        self.schedule = RandomActivation(self)
        self.rng = np.random.default_rng(rng_seed)
        self.setup_places_and_agents(sim_pop, population_dist)
        self.seed_initial_cases(seed_active, seed_latent)
        self.setup_datacollector()
        self.setup_collection_timing()

    def setup_places_and_agents(self, sim_pop: int, pop_dist: np.ndarray):
        ages = self.rng.choice(NUM_GROUPS, size=sim_pop, p=pop_dist / pop_dist.sum())
        avg_household = 3.5
        households = self.create_households(sim_pop, avg_household)
        self.assign_agents_to_homes(households, ages)
        self.assign_agents_to_schools()
        self.assign_agents_to_workplaces()
        self.assign_agents_to_others(sim_pop)

    def create_households(self, sim_pop: int, avg_size: float) -> List[Place]:
        count = math.ceil(sim_pop / avg_size)
        sizes = self.rng.poisson(avg_size - 1, count) + 1
        diff = sim_pop - sizes.sum()
        if diff > 0:
            extra = self.rng.choice(count, diff, replace=True)
            for idx in extra: sizes[idx] += 1
        elif diff < 0:
            reducible = np.where(sizes > 1)[0]
            for idx in reducible[: -diff]: sizes[idx] -= 1
        return [Place("home") for _ in sizes]

    def assign_agents_to_homes(self, households: List[Place], ages: np.ndarray):
        self.places = defaultdict(list)
        self.persons = {}
        uid_iter = iter(range(len(ages)))
        for hh in households:
            size = sum(1 for p in ages if True)  # simplified equal assignment
            for _ in range(size):
                uid = next(uid_iter)
                person = Person(uid, self, int(ages[uid]), self.rng)
                person.places["home"] = hh.uid
                hh.members.append(uid)
                self.schedule.add(person)
                self.persons[uid] = person
                self.places["home"].append(hh)

    def assign_agents_to_schools(self):
        enrollment = self.fetch_enrollment()
        students = [p for p in self.persons.values() if 1 <= p.age_group <= 6]
        self.populate_places(students, "school", enrollment, sizes=(400, 600))

    def assign_agents_to_workplaces(self):
        rate = self.fetch_employment_rate()
        workers = [p for p in self.persons.values() if 4 <= p.age_group <= 13 and self.rng.random() < rate]
        self.populate_places(workers, "work", count_per_place=50)

    def assign_agents_to_others(self, sim_pop: int):
        count = math.ceil(sim_pop / 250)
        others = [Place("other") for _ in range(count)]
        people = list(self.persons.values())
        self.round_robin_assign(people, others)
        self.places["other"].extend(others)

    def populate_places(
        self,
        people: List[Person],
        layer: str,
        enrollment: Dict[str, int] = None,
        sizes: Tuple[int, int] = (1, 1),
        count_per_place: int = None
    ):
        if layer == "school":
            primary_count = int(len(people) * enrollment["primary"] / sum(enrollment.values()))
            groups = [people[:primary_count], people[primary_count:]]
            place_sizes = sizes
        else:
            groups = [people]
            place_sizes = [count_per_place or 1]
        for group, size in zip(groups, place_sizes):
            place_count = max(1, math.ceil(len(group) / size))
            places = [Place(layer) for _ in range(place_count)]
            self.round_robin_assign(group, places)
            self.places[layer].extend(places)

    def round_robin_assign(self, people: List[Person], places: List[Place]) -> None:
        for idx, person in enumerate(people):
            plc = places[idx % len(places)]
            person.places[plc.setting] = plc.uid
            plc.members.append(person.unique_id)

    def seed_initial_cases(self, active_count: int, latent_count: int) -> None:
        active = self.rng.choice(list(self.persons.values()), size=active_count, replace=False)
        for person in active:
            person.state = "A"
            offset = self.rng.integers(0, self.params.delay_to_diag)
            person.inf_time = -offset
            person.report_time = self.params.delay_to_diag - offset
        susceptible = [p for p in self.persons.values() if p.state == "S"]
        latent = self.rng.choice(susceptible, size=min(latent_count, len(susceptible)), replace=False)
        for person in latent:
            person.state = "L"
            person.inf_time = -self.rng.integers(30, 180)

    def setup_datacollector(self) -> None:
        self.discoveries = 0
        self.notifications = 0
        self.collector = DataCollector(
            model_reporters={
                "Discoveries": lambda m: m.discoveries,
                "Notifications": lambda m: m.notifications,
            }
        )

    def setup_collection_timing(self) -> None:
        self.next_collect_date = (self.start_date + pd.offsets.MonthEnd(0)).to_pydatetime()
        self.next_collect_t = (self.next_collect_date - self.start_date).days + 1

    def new_discovery(self):
        self.discoveries += 1

    def new_notification(self):
        self.notifications += 1

    def seasonal_factor(self) -> float:
        if self.params.season_amp == 0:
            return 1.0
        month = (self.start_date + timedelta(days=self.current_t)).month
        phase = 2 * math.pi * (month - self.params.season_peak_month) / 12
        return 1 + self.params.season_amp * math.cos(phase)

    def step(self) -> None:
        self.transmit_all()
        self.schedule.step()
        if self.current_t + 1 == self.next_collect_t:
            self.collector.collect(self)
            self.discoveries = 0
            self.notifications = 0
            self.next_collect_date += pd.offsets.MonthEnd(1)
            self.next_collect_t = (self.next_collect_date - self.start_date).days + 1
        self.current_t += 1

    def transmit_all(self) -> None:
        for layer, places in self.places.items():
            beta = self.params.beta[layer]
            cm = self.params.contact_mats[layer]
            k_mean = self.params.daily_contacts * self.params.fractions[layer] * self.seasonal_factor()
            for plc in places:
                if not plc.members:
                    continue
                self.transmit_in_place(plc, beta, cm, k_mean)

    def transmit_in_place(
        self,
        place: Place,
        beta: float,
        cm: np.ndarray,
        k_mean: float
    ) -> None:
        uids = np.array(place.members, dtype=int)
        persons = [self.persons[uid] for uid in uids]
        ages = np.array([p.age_group for p in persons], dtype=int)
        states = np.array([p.state for p in persons])
        infectious = states == "A"
        susceptible = states == "S"
        if not infectious.any() or not susceptible.any():
            return
        inf_counts = np.bincount(ages[infectious], minlength=NUM_GROUPS)
        tot_counts = np.bincount(ages, minlength=NUM_GROUPS)
        prevalence = inf_counts / np.maximum(tot_counts, 1)
        sus_uids = uids[susceptible]
        sus_ages = ages[susceptible]
        contacts = self.rng.poisson(k_mean, len(sus_uids))
        probs = compute_infection_probability(sus_ages, contacts, cm, beta, prevalence)
        rand = self.rng.random(len(sus_uids))
        for uid, prob, r in zip(sus_uids, probs, rand):
            if r < prob:
                person = self.persons[int(uid)]
                person.state = "L"
                person.inf_time = self.current_t
                self.new_discovery()

# Compute MAPE
def compute_mape(sim: pd.Series, obs: pd.Series) -> float:
    df = pd.concat([sim, obs], axis=1).dropna()
    if df.empty:
        return float('nan')
    return float((df.iloc[:,0] - df.iloc[:,1]).abs().div(df.iloc[:,1]).mean() * 100)

# Simulation runner
def run_sim(
    country: str,
    data_dir: pathlib.Path,
    sim_pop: int,
    months: int,
    rng_seed: int,
    init_active: int,
    real_pop: int,
    start: str,
    cases_file: Optional[pathlib.Path],
    burnin: int = 0,
    save_dir: Optional[pathlib.Path] = None,
    save_fmt: str = "png",
    dpi: int = 300,
) -> Dict[str, any]:
    iso = COUNTRY_CODES[country]
    raw = COUNTRY_PARAMS[country].copy()
    contact_mats = {layer: load_contact_matrix(data_dir, iso, layer)
                    for layer in raw["beta"]}
    if raw.get("contact_scale", 1) != 1:
        for layer in contact_mats:
            contact_mats[layer] *= raw["contact_scale"]
    params = TBParams(
        beta=raw["beta"],
        latent_h0=raw["latent_h0"],
        latent_k=raw["latent_k"],
        treat_success=raw["treat_success"],
        contact_mats=contact_mats,
        daily_contacts=raw["daily_contacts"],
        contact_scale=raw["contact_scale"],
        fractions=raw["fractions"],
        report_rate=raw["report_rate"],
        delay_to_diag=raw["delay_to_diag"],
        incidence_100k=raw["incidence_100k"],
        prevalence_100k=raw.get("prevalence_100k"),
        season_amp=raw["season_amp"],
        season_peak_month=raw["season_peak_month"],
    )
    start_date = datetime.strptime(start, "%Y-%m-%d")
    if init_active <= 0:
        if params.prevalence_100k:
            init_active = round(params.prevalence_100k / 1e5 * real_pop)
        else:
            annual_inc = params.incidence_100k / 1e5 * real_pop
            init_active = round(annual_inc * params.delay_to_diag / 365)
    scale = sim_pop / real_pop
    seed_active = max(1, round(init_active * scale))
    seed_latent = max(1, round(params.incidence_100k / 1e5 * sim_pop * params.delay_to_diag / 365))
    idx = pd.date_range(start_date, periods=burnin + months, freq='M')
    obs_idx = idx[burnin:]
    observed = load_observed_cases(data_dir, iso, idx, cases_file)

    model = TBModel(
        params, sim_pop, load_population(data_dir, iso), data_dir,
        country, start_date, seed_active, seed_latent, rng_seed
    )
    total_days = (idx[-1] - start_date).days + 1
    for _ in trange(total_days, desc=f"Running {iso} seed={rng_seed}" , leave=False):
        model.step()

    df = model.collector.get_model_vars_dataframe()
    notifications = (pd.Series(df['Notifications'].values, index=idx) * (real_pop / sim_pop))
    result = pd.concat({'Sim': notifications, 'Obs': observed}, axis=1).loc[obs_idx]
    mape = compute_mape(result['Sim'], result['Obs'])

    fig, ax = plt.subplots()
    result.plot(ax=ax, title=f"{iso} Notifications (Sim vs Obs)")
    ax.set_ylabel("Cases")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(exist_ok=True)
        fname = save_dir / f"{iso}_seed{rng_seed}.{save_fmt}"
        fig.savefig(fname, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)

    return {"result": result, "MAPE": mape}

# Batch runner and CLI
def run_multiple_sims(
    country: str,
    data_dir: pathlib.Path,
    sim_pop: int,
    months: int,
    seeds: List[int],
    init_active: int,
    real_pops: List[int],
    start: str,
    cases_files: List[Optional[pathlib.Path]],
    burnin: int = 0,
    save_dir: Optional[pathlib.Path] = None,
    save_fmt: str = "png",
    dpi: int = 300,
) -> Dict[str, any]:
    results = {}
    for pop_real, cases_file in zip(real_pops, cases_files):
        sim_df = pd.DataFrame()
        for seed in seeds:
            out = run_sim(
                country, data_dir, sim_pop, months, seed,
                init_active, pop_real, start, cases_file,
                burnin, save_dir, save_fmt, dpi
            )
            sim_df[f"seed_{seed}"] = out['result']['Sim']
        idx = sim_df.index
        obs = load_observed_cases(data_dir, COUNTRY_CODES[country], idx, cases_file)
        results[country] = {
            'simulations': sim_df,
            'observed': obs,
        }
    return results


def cli():
    parser = argparse.ArgumentParser(description="Batch TB-ABM simulation")
    parser.add_argument("--countries", nargs="+", choices=list(COUNTRY_PARAMS.keys()), required=True)
    parser.add_argument("--pop", nargs="+", type=int, required=True)
    parser.add_argument("--real_pops", nargs="+", type=int, required=True)
    parser.add_argument("--init_active", nargs="+", type=int, required=True)
    parser.add_argument("--casesfiles", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--data_dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--months", type=int, default=10)
    parser.add_argument("--burnin", type=int, default=14)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--save_dir", type=pathlib.Path)
    parser.add_argument("--save_fmt", type=str, default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()

    configure_logging(args.loglevel)

    n = len(args.countries)
    for name in ['pop','real_pops','init_active','casesfiles']:
        if len(getattr(args, name)) != n:
            parser.error(f"--{name} must have {n} values")

    for country, pop, real_pop, init_act, cases_file in zip(
        args.countries, args.pop, args.real_pops, args.init_active, args.casesfiles
    ):
        run_multiple_sims(
            country, args.data_dir, pop, args.months,
            args.seeds, init_act, [real_pop], args.start,
            [cases_file], args.burnin,
            args.save_dir, args.save_fmt, args.dpi
        )

if __name__ == '__main__':
    cli()
