import argparse, logging, math, pathlib
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from mesa import Agent, Model
from mesa.time import RandomActivation
from numba import njit
COUNTRY_CODES={"korea":"KOR","south_africa":"ZAF"}  # 国家
FILE_PATTERNS={"contact":"{ISO}*{layer}*2015.csv","population":"population*{ISO}.csv","cases":"observed*{ISO}_2023.csv"}
AGE_GROUPS=list(range(0,85,5)); NUM_GROUPS=len(AGE_GROUPS)  # 年龄
def configure_logging(level="INFO"):
    logging.basicConfig(level=getattr(logging,level),format="%(levelname)s %(message)s")  # 日志


def get_country_params(country):
    if country=="korea":
        return {"beta":{"home":0.049,"school":0.034,"work":0.026,"other":0.018},
                "latent_h0":0.035,"latent_k":0.20,"treat_success":0.88,"delay_to_diag":35,"report_rate":0.88,
                "daily_contacts":9.2,"fractions":{"home":0.22,"school":0.19,"work":0.10,"other":0.49},
                "season_amp":0.05,"season_peak":11}  # 韩国参数
    else:
        return {"beta":{"home":0.049,"school":0.034,"work":0.026,"other":0.018},
                "latent_h0":0.055,"latent_k":0.18,"treat_success":0.77,"delay_to_diag":90,"report_rate":0.78,
                "daily_contacts":13.0,"fractions":{"home":0.22,"school":0.19,"work":0.10,"other":0.49},
                "season_amp":0.25,"season_peak":8}  # 南非参数

def load_contact_matrix(data_dir,iso,layer):
    path=pathlib.Path(data_dir)/FILE_PATTERNS["contact"].format(ISO=iso,layer=layer)
    df=pd.read_csv(path,header=None); num=df.apply(pd.to_numeric,errors="coerce")
    i0=num.notnull().any(axis=1).idxmax(); j0=num.notnull().any(axis=0).idxmax()
    mat=num.iloc[i0:,j0:].values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i,j]):
                rowm=np.nanmean(mat[i,:]); colm=np.nanmean(mat[:,j])
                mat[i,j]=np.nanmean([rowm,colm])
    return mat.astype(float)  # 接触矩阵

def load_population(data_dir,iso):
    path=pathlib.Path(data_dir)/FILE_PATTERNS["population"].format(ISO=iso)
    df=pd.read_csv(path); col=df.select_dtypes(include=[np.number]).columns[0]; arr=df[col].values
    return arr.astype(int)  # 人口分布

def load_cases(data_dir,iso,dates,override_file=None):
    if override_file:
        df=pd.read_csv(override_file); df.columns=["Date","Cases"]
    else:
        path=pathlib.Path(data_dir)/FILE_PATTERNS["cases"].format(ISO=iso)
        df=pd.read_csv(path,names=["Date","Cases"],header=0)
    df["Date"]=pd.to_datetime(df["Date"]); series=df.set_index("Date")["Cases"]
    return series.reindex(dates,fill_value=0)  # 观测病例

class Person(Agent):
    def __init__(self,uid,model,age_group):
        super().__init__(uid,model); self.age_group=age_group; self.state="S"; self.inf_time=None; self.report_time=None
    def step(self):
        t=self.model.t
        if self.state=="L":
            years=(t-self.inf_time)/365; lam=self.model.params["latent_h0"]*math.exp(-self.model.params["latent_k"]*years)
            p=1-math.exp(-lam/365)
            if np.random.random()<p: self.state="A"  # 潜伏 活跃
        elif self.state=="A":
            if self.report_time is None: self.report_time=t+np.random.poisson(self.model.params["delay_to_diag"])
            if t>=self.report_time and np.random.random()<self.model.params["report_rate"]:
                self.model.new_reports+=1
            if np.random.random()<1/180: self.state="R"  # 康复

class Place:
    def __init__(self,layer): self.layer=layer; self.members=[]  # 场所

@njit
def compute_probs(sus_ages,contacts,cm,beta,prev):
    n=len(sus_ages); out=np.zeros(n,dtype=np.float32)
    for i in range(n):
        age=sus_ages[i]; row=cm[age]; total=row.sum()
        if total<=0: out[i]=0; continue
        mix=(row/total*prev).sum(); out[i]=1-math.exp(-beta*contacts[i]*mix)
    return out  # 感染概率

class TBModel(Model):
    def __init__(self,country,pop_size,start_date,seed_active,seed_latent,data_dir,cases_file,months,burnin):
        super().__init__(); configure_logging(); self.t=0; self.start_date=pd.to_datetime(start_date)
        iso=COUNTRY_CODES[country]; params=get_country_params(country); total=burnin+months
        dates=pd.date_range(start_date,periods=total,freq="M")
        self.obs=load_cases(data_dir,iso,dates,cases_file); self.params=params
        pop_dist=load_population(data_dir,iso)
        self.contact_mats={l:load_contact_matrix(data_dir,iso,l) for l in params["beta"]}
        self.schedule=RandomActivation(self); self.new_reports=0; self.data=[]
        ages=np.random.choice(NUM_GROUPS,pop_size,p=pop_dist/pop_dist.sum())
        for uid,age in enumerate(ages): self.schedule.add(Person(uid,self,age))
        agents=self.schedule.agents; act=np.random.choice(agents,seed_active,replace=False)
        for p in act: p.state="A"; p.inf_time=-np.random.randint(0,params["delay_to_diag"])
        sus=[p for p in agents if p.state=="S"]; lat=np.random.choice(sus,seed_latent,replace=False)
        for p in lat: p.state="L"; p.inf_time=-np.random.randint(30,180)

    def step(self):
        ages=np.array([p.age_group for p in self.schedule.agents]); states=np.array([p.state for p in self.schedule.agents])
        inf_mask=states=="A"; sus_mask=states=="S"
        prev=np.bincount(ages[inf_mask],minlength=NUM_GROUPS)/np.bincount(ages,minlength=NUM_GROUPS)
        for layer,beta in self.params["beta"].items():
            cm=self.contact_mats[layer]; frac=self.params["fractions"][layer]
            contacts=np.random.poisson(self.params["daily_contacts"]*frac,len(ages))
            probs=compute_probs(ages[sus_mask],contacts[sus_mask],cm,beta,prev)
            for p,pr in zip(np.array(self.schedule.agents)[sus_mask],probs):
                if np.random.random()<pr: p.state="L"; p.inf_time=self.t
        self.data.append(self.new_reports); self.new_reports=0
        self.schedule.step(); self.t+=1  # 一步

def run_sim(country,pop_size,months,seed_active,seed_latent,start,burnin,data_dir,cases_file,save_dir=None):
    model=TBModel(country,pop_size,start,seed_active,seed_latent,data_dir,cases_file,months,burnin)
    for _ in range(burnin+months): model.step()
    dates=pd.date_range(start,periods=burnin+months,freq="M")[burnin:]
    sim=pd.Series(model.data,dates); obs=model.obs[burnin:]
    df=pd.concat([sim,obs],axis=1); df.columns=["Sim","Obs"]
    df.plot(title=f"{country} TB 通报对比",ylabel="病例数")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); plt.tight_layout()
    if save_dir: plt.savefig(pathlib.Path(save_dir)/f"{country}.png")
    else: plt.show()

if __name__=="__main__":
    p=argparse.ArgumentParser("TB-ABM");
    p.add_argument("--country",choices=["korea","south_africa"],required=True); p.add_argument("--pop",type=int,required=True)
    p.add_argument("--months",type=int,default=12); p.add_argument("--burnin",type=int,default=3)
    p.add_argument("--seed_active",type=int,default=10); p.add_argument("--seed_latent",type=int,default=20)
    p.add_argument("--start",type=str,default="2022-01-01"); p.add_argument("--data_dir",type=str,default="data")
    p.add_argument("--cases_file",type=str,default=None); p.add_argument("--save_dir",type=str,default=None)
    args=p.parse_args(); configure_logging();
    run_sim(args.country,args.pop,args.months,args.seed_active,args.seed_latent,args.start,args.burnin,args.data_dir,args.cases_file,args.save_dir)
