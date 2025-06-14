
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

# 0  日志 & 常量
LOG_FMT = "%(levelname)s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("TB-ABM")

COUNTRY_CODES = {"korea": "KOR", "south_africa": "ZAF"}
COUNTRY_FULLNAME = {"korea": "Republic of Korea", "south_africa": "South Africa"}
FILE_MAP = {
    "contact": "{ISO}_{layer}_2015.csv",
    "pop": "population_{ISO}.csv",
    "cases": "observed_{ISO}_2023.csv",
}

#  自然史 & 国别参数
COUNTRY_SPECIFIC: Dict[str, Dict] = {
    "korea": {
        "beta": {k: v * 0.65 for k, v in
                 dict(home=0.049, school=0.034, work=0.026, other=0.018).items()},
        "latent_h0": 0.035,
        "latent_k": 0.20,
        "treat_success": 0.88,
        "delay_to_diag_days": 35,
        "reporting_rate": 0.88,
        "daily_contacts": 9.2,
        "contact_scale": 0.83,
        "fraction_of_contacts": dict(home=0.22, school=0.19, work=0.10, other=0.49),
        "incidence_per_100k": 29,
        "prevalence_per_100k": 44,
        "season_amp": 0.05,
        "season_peak_month": 11,
    },
    "south_africa": {
        "beta": {k: v * 0.8 for k, v in
                 dict(home=0.049, school=0.034, work=0.026, other=0.018).items()},
        "latent_h0": 0.055,
        "latent_k": 0.18,
        "treat_success": 0.77,
        "delay_to_diag_days": 90,
        "reporting_rate": 0.78,
        "daily_contacts": 13.0,
        "contact_scale": 1.1,
        "fraction_of_contacts": dict(home=0.22, school=0.19, work=0.10, other=0.49),
        "incidence_per_100k": 427,
        # "prevalence_per_100k": 737,
        "season_amp": 0.25,
        "season_peak_month": 8,
    },
}

# 17 个年龄段
AGE_GROUPS = np.arange(0, 85, 5)  # 0-4 … 80-84
NG = len(AGE_GROUPS)


# 1. 数据加载函数
def load_contact_matrix(data_dir: pathlib.Path, iso: str, layer: str) -> np.ndarray:
    """读取并清洗接触矩阵 CSV，返回 *NG×NG* numpy 矩阵"""
    path = data_dir / FILE_MAP["contact"].format(ISO=iso, layer=layer)
    df_raw = pd.read_csv(path, header=None, dtype=str)
    if df_raw.empty:
        raise ValueError(f"{path} is empty.")
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")
    first_row = df_num.notna().any(axis=1).idxmax()
    first_col = df_num.notna().any(axis=0).idxmax()
    mat = df_num.iloc[first_row:, first_col:].values.astype(np.float32)
    if mat.shape != (NG, NG):
        raise ValueError(f"{path}: expected {NG}×{NG}, got {mat.shape}")
    if np.isnan(mat).any():
        for i in range(NG):
            for j in range(NG):
                if np.isnan(mat[i, j]):
                    r = np.nanmean(mat[i, :]); c = np.nanmean(mat[:, j])
                    mat[i, j] = 0.0 if np.isnan(r) and np.isnan(c) else np.nanmean([r, c])
    assert mat.sum() > 0, f"{path} reduced to all-zero."
    return mat


def load_population(data_dir: pathlib.Path, iso: str) -> np.ndarray:
    """读取总人口年龄分布，输出 length==NG 的 ndarray"""
    df = pd.read_csv(data_dir / FILE_MAP["pop"].format(ISO=iso))
    col = next((c for c in ("Pop", "Population", "population") if c in df.columns), None)
    if col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][0]
        col = num_cols
    if df.shape[0] != NG:
        raise ValueError(f"population file must have {NG} rows, got {df.shape[0]}")
    return df[col].astype(int).values


def load_obs(cases_file: Optional[pathlib.Path],
             data_dir: pathlib.Path, iso: str,
             idx: pd.DatetimeIndex) -> pd.Series:
    """读取观测病例数据并重索引"""
    if cases_file is not None:
        df = pd.read_csv(cases_file)
        if df.columns.size < 2:
            df.columns = ["Date", "Cases"]
        if np.issubdtype(df.iloc[:, 0].dtype, np.integer):
            df["Date"] = pd.to_datetime("2023-" + df.iloc[:, 0].astype(str) + "-01") + pd.offsets.MonthEnd(0)
            df["Cases"] = df.iloc[:, 1].astype(float)
        else:
            df.columns = ["Date", "Cases"]; df["Date"] = pd.to_datetime(df["Date"])
        ser = df.set_index("Date")["Cases"].astype(float)
    else:
        ser = (pd.read_csv(data_dir / FILE_MAP["cases"].format(ISO=iso), parse_dates=[0])
               .set_index("Date").iloc[:, 0].astype(float))
    ser = ser.reindex(idx)
    return ser


# 2. 数据可视化 & 分析函数
def plot_monthly_comparison(result_df: pd.DataFrame) -> plt.Axes:
    ax = result_df.plot(title="Monthly Comparison of Simulated vs Observed TB Notifications")
    ax.set_ylabel("Cases")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout();
    plt.show()
    return ax


def plot_confidence_band(sim_df: pd.DataFrame, obs_series: pd.Series):
    """绘制 95% 置信带"""
    lower = sim_df.quantile(0.025, axis=1)
    upper = sim_df.quantile(0.975, axis=1)
    median = sim_df.median(axis=1)

    ax = median.plot(title="95% Confidence Band for Simulated TB Notifications")
    ax.fill_between(sim_df.index, lower, upper, color="lightgray", label="95% Confidence Interval")
    ax.plot(obs_series, label="Observed", color="red", linestyle="--")
    ax.set_ylabel("Cases")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout();
    plt.legend();
    plt.show()


def compute_mape(sim: pd.Series, obs: pd.Series) -> float:
    """计算平均绝对百分比误差 (MAPE)，忽略 NaN"""
    common = pd.concat([sim, obs], axis=1).dropna()
    if common.empty:
        return float("nan")
    return (common.iloc[:, 0].sub(common.iloc[:, 1]).abs().div(common.iloc[:, 1]).mean() * 100)


# 3. Agent & Model
@dataclass
class TBParams:
    beta: Dict[str, float]
    latent_h0: float
    latent_k: float
    treat_success: float
    daily_contacts: float
    contact_mats: Dict[str, np.ndarray]
    reporting_rate: float
    delay_to_diag_days: int
    contact_scale: float = 1.0
    fraction_of_contacts: Optional[Dict[str, float]] = None
    incidence_per_100k: Optional[int] = None
    prevalence_per_100k: Optional[int] = None
    season_amp: float = 0.0
    season_peak_month: int = 1
    latent_first2y: float = 0.03
    latent_after: float = 0.001

# Person / Place / TBModel
class Person(Agent):
    __slots__ = ("age_grp", "state", "inf_step", "report_due", "place_ids", "rng")

    def __init__(self, uid: int, model: "TBModel", age_grp: int, rng: np.random.Generator):
        super().__init__(uid, model)
        self.age_grp, self.state, self.inf_step, self.report_due = age_grp, "S", None, None
        self.place_ids, self.rng = {}, rng

    def _progress(self):
        p = self.model.params
        if self.state == "L":
            if self.inf_step == self.model.t:
                return
            years = (self.model.t - self.inf_step) / 365.0
            lam_year = p.latent_h0 * math.exp(-p.latent_k * years)
            p_day = 1.0 - math.exp(-lam_year / 365.0)
            if self.rng.random() < p_day:
                self.state = "A"
                self.model._inc_month_dis += 1
        elif self.state == "A":
            if self.report_due is None:
                delay = self.rng.poisson(p.delay_to_diag_days)
                self.report_due = self.model.t + delay
            if self.model.t == self.report_due:
                if self.rng.random() < p.reporting_rate:
                    self.model._inc_month_notif += 1
            if self.rng.random() < 1 / 180:
                self.state = "R" if self.rng.random() < p.treat_success else "D"

    def step(self):
        self._progress()


class Place:
    _uid_counter = 0
    __slots__ = ("uid", "setting", "members")

    def __init__(self, setting: str):
        self.setting, self.members = setting, []
        self.uid = Place._uid_counter
        Place._uid_counter += 1


@njit
def _compute_p_inf(sus_ages: np.ndarray, k: np.ndarray, cm: np.ndarray,
                   beta: float, prev: np.ndarray) -> np.ndarray:
    n = sus_ages.shape[0]
    p_inf = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row = cm[sus_ages[i]]
        rsum = np.sum(row)
        if rsum == 0.0:
            continue
        lam = beta * k[i]
        s = np.sum((row / rsum) * prev)
        p_inf[i] = 1.0 - math.exp(-lam * s)
    return p_inf

class TBModel(Model):
    def __init__(
            self, params: TBParams, pop_size: int, population_age: np.ndarray,
            data_dir: pathlib.Path, country: str,
            start_date: datetime,
            seed_active: int = 0, seed_latent: int = 0, seed: int = 42,   # ★ 修改
    ):
        super().__init__()
        Place._uid_counter = 0
        self.params, self.t = params, 0
        self.schedule, self.rng = RandomActivation(self), np.random.default_rng(seed)
        self.data_dir, self.country = data_dir, country
        self.start_date = start_date

        # 分层接触权重
        self.layer_weights = (params.fraction_of_contacts or
                              {l: params.contact_mats[l].mean() /
                               sum(m.mean() for m in params.contact_mats.values())
                               for l in params.contact_mats})

        # 年龄分布
        ages = self.rng.choice(np.arange(NG), size=pop_size,
                               p=population_age / population_age.sum())

        # 生成家庭 / 学校 / 工作 / 其他
        avg_hh = 3.5
        n_hh = math.ceil(pop_size / avg_hh)
        hh_sizes = self.rng.poisson(avg_hh - 1, n_hh) + 1
        diff = int(pop_size - hh_sizes.sum())
        if diff > 0:
            idx = self.rng.choice(len(hh_sizes), diff, replace=True); np.add.at(hh_sizes, idx, 1)
        elif diff < 0:
            idx = np.where(hh_sizes > 1)[0]; self.rng.shuffle(idx)
            for i in idx[: -diff]:
                hh_sizes[i] -= 1

        self.places: Dict[str, List[Place]] = defaultdict(list)
        self.uid2person: Dict[int, Person] = {}
        def new_place(layer: str) -> Place:
            plc = Place(layer); self.places[layer].append(plc); return plc

        homes = [new_place("home") for _ in hh_sizes]
        persons: List[Person] = []
        uid_iter = iter(range(pop_size))
        for home, size in zip(homes, hh_sizes):
            for _ in range(size):
                uid = next(uid_iter)
                p = Person(uid, self, int(ages[uid]), self.rng)
                p.place_ids["home"] = home.uid
                home.members.append(uid)
                self.schedule.add(p)
                persons.append(p)
                self.uid2person[uid] = p
        self.persons = persons

        # 学校
        enrol = self._get_enrollment();
        tot = enrol["primary"] + enrol["secondary"] or 1
        students = [p for p in persons if 1 <= p.age_grp <= 6];
        self.rng.shuffle(students)
        n_pr = int(round(len(students) * enrol["primary"] / tot))

        def assign(studs, list_sz):
            if not studs: return
            n = max(1, math.ceil(len(studs) / list_sz))
            sch = [new_place("school") for _ in range(n)]
            for i, stu in enumerate(studs):
                plc = sch[i % n];
                stu.place_ids["school"] = plc.uid;
                plc.members.append(stu.unique_id)

        assign(students[:n_pr], 400);
        assign(students[n_pr:], 600)

        # 工作
        emp_rate = self._get_employment_rate()
        workers = [p for p in persons if 4 <= p.age_grp <= 13 and self.rng.random() < emp_rate]
        self.rng.shuffle(workers);
        n_work = max(1, math.ceil(len(workers) / 50))
        works = [new_place("work") for _ in range(n_work)]
        for i, w in enumerate(workers):
            plc = works[i % n_work];
            w.place_ids["work"] = plc.uid;
            plc.members.append(w.unique_id)

        # 其他公共场所
        avg_other = 250;
        n_other = max(1, math.ceil(pop_size / avg_other))
        others = [new_place("other") for _ in range(n_other)]
        self.rng.shuffle(persons)
        for i, p in enumerate(persons):
            plc = others[i % n_other];
            p.place_ids["other"] = plc.uid;
            plc.members.append(p.unique_id)

        # 初始 Active & Latent
        delay = self.params.delay_to_diag_days

        init_A = self.rng.choice(self.persons, size=seed_active, replace=False)
        for p in init_A:
            p.state = "A"
            offset = self.rng.integers(0, delay)
            p.inf_step = -offset
            p.report_due = delay - offset

        remaining = [q for q in self.persons if q.state == "S"]
        if seed_latent > len(remaining):
            seed_latent = len(remaining)
        init_L = self.rng.choice(remaining, size=seed_latent, replace=False)
        for p in init_L:
            p.state = "L"
            p.inf_step = -self.rng.integers(30, 180)  # 已潜伏 1~6 个月

        # 计数器 & DataCollector
        self._inc_month_inf = self._inc_month_dis = self._inc_month_notif = 0
        self.datacollector = DataCollector({
            "IncidenceDis": lambda m: m._inc_month_dis,
            "Notifications": lambda m: m._inc_month_notif,
        })

        self._next_collect_day = (start_date + pd.offsets.MonthEnd(0)).to_pydatetime()
        self._next_collect_t = (self._next_collect_day - self.start_date).days + 1

    # [就业率]
    def _get_employment_rate(self) -> float:
        iso_code = COUNTRY_CODES[self.country]
        path = self.data_dir / "Employment.csv"
        if not path.exists():
            logger.warning("找不到 Employment.csv，使用默认就业率 0.55"); return 0.55
        df = pd.read_csv(path)
        sub = df[(df["Country"] == COUNTRY_FULLNAME[self.country]) | (df["Country"] == iso_code)]
        if sub.empty:
            logger.warning("Employment.csv 无对应国家记录，使用默认 0.55"); return 0.55
        yr = sub[sub["Year"] <= 2023]["Year"].max()
        return float(sub.loc[sub["Year"] == yr, "Employment-to-population ratio"].iloc[0] / 100)

    # [入学人数]
    def _get_enrollment(self) -> Dict[str, int]:
        iso_code = COUNTRY_CODES[self.country]; path = self.data_dir / "school.csv"
        if not path.exists():
            logger.warning("找不到 school.csv，按 0 处理入学人数"); return {"primary": 0, "secondary": 0}
        df = pd.read_csv(path)
        sub = df[(df["geoUnit"] == COUNTRY_FULLNAME[self.country]) | (df["geoUnit"] == iso_code)]
        if sub.empty:
            logger.warning("school.csv 无对应国家记录，按 0 处理"); return {"primary": 0, "secondary": 0}
        latest = sub[sub["year"] <= 2023].groupby("indicatorId")["year"].max()
        def _get(ind):
            yr = latest.get(ind); return 0 if pd.isna(yr) else int(sub[(sub["indicatorId"] == ind) & (sub["year"] == yr)]["value"].iloc[0])
        return {"primary": _get("primary education"), "secondary": _get("secondary education")}

    # [当天季节性倍率]
    def _season_factor(self) -> float:
        if self.params.season_amp == 0.0: return 1.0
        date = self.start_date + timedelta(days=self.t)
        month = date.month
        phase = (month - self.params.season_peak_month) / 12.0 * 2 * math.pi
        return 1.0 + self.params.season_amp * math.cos(phase)

    # [传播]
    def _layer_transmit(self, layer: str):
        cm = self.params.contact_mats[layer]
        beta = self.params.beta[layer]
        k_mean = self.params.daily_contacts * self.layer_weights[layer] * self._season_factor()

        for plc in self.places[layer]:
            if not plc.members: continue
            mem = np.fromiter(plc.members, dtype=np.int64)
            pers = [self.uid2person[u] for u in mem]
            states = np.array([p.state for p in pers]); ages = np.array([p.age_grp for p in pers], dtype=np.int64)
            inf_mask = states == "A"; sus_mask = states == "S"
            if not inf_mask.any() or not sus_mask.any(): continue

            inf_by_age = np.bincount(ages[inf_mask], minlength=NG)
            tot_by_age = np.bincount(ages, minlength=NG)
            prev = np.divide(inf_by_age, tot_by_age, out=np.zeros_like(inf_by_age, dtype=np.float32), where=tot_by_age > 0)

            sus_idx = mem[sus_mask]; sus_ages = ages[sus_mask]
            k = self.rng.poisson(k_mean, len(sus_idx))
            p_inf = _compute_p_inf(sus_ages, k.astype(np.int64), cm, beta, prev.astype(np.float32))
            rand_u = self.rng.random(len(sus_idx))
            for uid in sus_idx[rand_u < p_inf]:
                p = self.uid2person[int(uid)]; p.state, p.inf_step = "L", self.t; self._inc_month_inf += 1

    def step(self):
        for layer in ("school", "work", "other", "home"):
            self._layer_transmit(layer)
        self.schedule.step()

        # 月末收集
        if self.t + 1 == self._next_collect_t:
            self.datacollector.collect(self)
            self._inc_month_inf = self._inc_month_dis = self._inc_month_notif = 0
            # 下一个月末
            self._next_collect_day += pd.offsets.MonthEnd(1)
            self._next_collect_t = (self._next_collect_day - self.start_date).days + 1
        self.t += 1

# 4  run_sim
def run_sim(
        country: str,
        data_dir: pathlib.Path,
        pop: int,
        months: int,
        seed: int,
        init_active: int,
        population_real: int,
        start: str,
        cases_file: Optional[pathlib.Path],
        burnin_months: int = 0,
        show_plot: bool = True,
        # ---------- 新增 3 个可选参数 ----------
        save_dir: Optional[pathlib.Path] = None,   # 保存文件夹；None 表示不保存
        save_fmt: str = "png",                     # png / pdf / svg …
        dpi: int = 300,                            # 分辨率
):

    iso = COUNTRY_CODES[country]
    spec = COUNTRY_SPECIFIC[country].copy()

    # 可选整体调参 默认保持 1.0）
    beta_mul = 1.0
    contact_scale_mul = 1.0
    if beta_mul != 1.0:
        spec["beta"] = {k: v * beta_mul for k, v in spec["beta"].items()}
    if contact_scale_mul != 1.0:
        spec["contact_scale"] *= contact_scale_mul

    # 自动估算 init_active
    if init_active <= 0:
        if spec.get("prevalence_per_100k"):
            init_active = int(round(spec["prevalence_per_100k"] / 1e5 * population_real))
        else:
            annual_inc = spec["incidence_per_100k"] / 1e5 * population_real
            init_active = int(round(annual_inc * (spec["delay_to_diag_days"] / 365)))

    scale_real_to_sim = pop / population_real
    seed_active_sim = max(1, round(init_active * scale_real_to_sim))
    seed_latent_sim = max(1, round(
        spec["incidence_per_100k"] / 1e5 * pop *
        spec["delay_to_diag_days"] / 365))

    idx_full = pd.date_range(start, periods=burnin_months + months, freq="ME")
    idx_compare = idx_full[burnin_months:]
    mats = {lay: load_contact_matrix(data_dir, iso, lay) for lay in ("home", "school", "work", "other")}
    if spec.get("contact_scale", 1.0) != 1.0:
        mats = {l: m * spec["contact_scale"] for l, m in mats.items()}
    pop_age = load_population(data_dir, iso)
    obs_full = load_obs(cases_file, data_dir, iso, idx_full)

    params = TBParams(**spec, contact_mats=mats)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    model = TBModel(params, pop, pop_age, data_dir, country,
                    start_dt, seed_active_sim, seed_latent_sim, seed)

    total_days = (idx_full[-1] - start_dt).days + 1
    for _ in trange(total_days, desc=f"Sim {iso} seed={seed}", leave=False, unit="day"):
        model.step()

    sim_monthly_full = (pd.Series(model.datacollector.get_model_vars_dataframe()["Notifications"].values,
                                  index=idx_full) * (population_real / pop))
    result_full = pd.concat({"Sim": sim_monthly_full, "Obs": obs_full}, axis=1).sort_index()
    result = result_full.loc[idx_compare]

    common = result.dropna()
    mape = (common["Sim"].sub(common["Obs"]).abs().div(common["Obs"]).mean() * 100) if not common.empty else float("nan")

    # 画图
    ax = result.plot(
        title=f"{iso} TB Notifications vs Observed  "
              f"({idx_compare[0].strftime('%Y-%m')}–{idx_compare[-1].strftime('%Y-%m')})"
    )
    ax.set_ylabel("Cases")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", rotation=0)

    fig = ax.get_figure()

    # 保存
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{iso}_seed{seed}_{start}_m{months}.{save_fmt}"
        fig.savefig(save_dir / fname, dpi=dpi, bbox_inches="tight")

    # 弹窗 or 关闭
    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    return {"result": result, "MAPE": mape}


# 5. 批量仿真函数：[一次性跑多个随机种子]

def run_multiple_sims(
    country: str,
    data_dir: pathlib.Path,
    pop: int,
    months: int,
    seeds: List[int],
    init_active: int,
    population_real: int,
    start: str,
    cases_file: Optional[pathlib.Path],
    burnin_months: int = 0,
    beta_mul: float = 1.0,
    contact_scale_mul: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """给定多个随机种子，返回 (sim_df, obs_series)。
    *sim_df* 的列为各随机种子，行为时间索引。
    """
    sims = []
    for sd in seeds:
        res = run_sim(
            country,
            data_dir,
            pop,
            months,
            sd,
            init_active,
            population_real,
            start,
            cases_file,
            burnin_months=burnin_months,
            show_plot=False,  # 批量时不逐个绘图
            beta_mul=beta_mul,
            contact_scale_mul=contact_scale_mul,
        )
        sims.append(res["result"]["Sim"])

    sim_df = pd.DataFrame(sims, index=[f"seed_{s}" for s in seeds]).T
    iso = COUNTRY_CODES[country]
    full_idx = sim_df.index
    obs_series = load_obs(cases_file, data_dir, iso, full_idx)
    return sim_df, obs_series


# 6. 自动分析：完成批量仿真后即刻执行第二部分分析

def pipeline_batch(
    countries: List[str],
    data_dir: pathlib.Path,
    pop: int,
    months: int,
    seeds: List[int],
    init_active: int,
    population_real_list: List[int],
    start: str,
    cases_file: Optional[pathlib.Path],
    burnin_months: int = 0,
    beta_mul: float = 1.0,
    contact_scale_mul: float = 1.0,
):
    for country, pop_real in zip(countries, population_real_list):
        logger.info("=" * 70)
        logger.info(f"开始批量仿真 {country.upper()}  (seeds={seeds}) …")

        sim_df, obs_series = run_multiple_sims(
            country,
            data_dir,
            pop,
            months,
            seeds,
            init_active,
            pop_real,
            start,
            cases_file,
            burnin_months=burnin_months,
            beta_mul=beta_mul,
            contact_scale_mul=contact_scale_mul,
        )

        # —— 第二部分分析 ——
        plot_confidence_band(sim_df, obs_series)

        median_sim = sim_df.median(axis=1)
        result_df = pd.concat({"Sim": median_sim, "Obs": obs_series}, axis=1)
        plot_monthly_comparison(result_df)

        mape_val = compute_mape(median_sim, obs_series)
        logger.info(f"{country.upper()}  median-sim vs Obs  MAPE = {mape_val:.2f}%")


# 7. CLI 入口：[兼容单国单 seed，也可一次跑多国多 seed]


def cli():
    p = argparse.ArgumentParser(description="Batch TB-ABM simulation")

    # 支持多国家 & 多列表
    p.add_argument("--countries", nargs="+", choices=["korea", "south_africa"], required=True)
    p.add_argument("--pop", nargs="+", type=int, required=True)
    p.add_argument("--population_real_list", nargs="+", type=int, required=True)
    p.add_argument("--init_active", nargs="+", type=int, required=True)
    p.add_argument("--casesfile", nargs="+", type=pathlib.Path, required=True)

    #  通用单值参数
    p.add_argument("--data", type=pathlib.Path, default="data")
    p.add_argument("--start", type=str, default="2022-01-01")
    p.add_argument("--months", type=int, default=10)
    p.add_argument("--burnin", type=int, default=14)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--beta_mul", type=float, default=1.0)
    p.add_argument("--contact_scale_mul", type=float, default=1.0)

    # 保存图片
    p.add_argument("--save_dir", type=pathlib.Path, help="保存所有图到此目录；若不设则不保存")
    p.add_argument("--save_fmt", type=str, default="png")

    p.add_argument("--loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    a = p.parse_args()
    logger.setLevel(getattr(logging, a.loglevel))

    #  长度校验
    n = len(a.countries)
    for name in ("pop", "population_real_list", "init_active", "casesfile"):
        if len(getattr(a, name)) != n:
            p.error(f"--{name} 必须给 {n} 个值，与 --countries 等长")

    # 批量循环
    for i, country in enumerate(a.countries):
        for sd in a.seeds:
            run_sim(
                country=country,
                data_dir=a.data,
                pop=a.pop[i],
                months=a.months,
                seed=sd,
                init_active=a.init_active[i],
                population_real=a.population_real_list[i],
                start=a.start,
                cases_file=a.casesfile[i],
                burnin_months=a.burnin,
                show_plot=(a.save_dir is None),
                save_dir=a.save_dir,
                save_fmt=a.save_fmt,
                dpi=300,
            )
    logger.info("全部模拟完成")

if __name__ == "__main__":
    cli()
