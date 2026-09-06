#!/usr/bin/env python3
"""Variance rerun analysis: per-instance distributions of episode cost, E vs A8, over the
original rounds + variance reruns. Usage: anchor_variance.py (paths hardcoded below)."""
import sys, os, json, statistics as st, math, random
sys.path.insert(0, os.path.dirname(__file__))
from anchor_compare import episode_rows, patch_files
data=json.load(open(os.path.expanduser("~/Projects/refactorbench/harness/data/swe-bench-promax.json")))
gold={x["instance_id"]: patch_files(x["patch"]) for x in data}
def rows(run):
    p=f"/mnt/nas/refactorbench/runs/{run}"; return episode_rows(p, gold) if os.path.isdir(p) else {}
E_runs=["s1-arm-e-r1","s1-arm-e-r2"]+[f"s1-arm-e-var-r{i}" for i in range(1,6)]
A_runs=["s1-arm-a8-r1","s1-arm-a8-r2"]+[f"s1-arm-a8-var-r{i}" for i in range(1,6)]
E=[rows(r) for r in E_runs]; A=[rows(r) for r in A_runs]
ids=["albumentations-team__albumentations-2495","huggingface__transformers-38332","langchain-ai__langchain-32996","django__django-19643","mikf__gallery-dl-7872"]
def mw_p(x,y):
    n1,n2=len(x),len(y); allv=sorted([(v,0) for v in x]+[(v,1) for v in y]); ranks={}; i=0
    while i<len(allv):
        j=i
        while j+1<len(allv) and allv[j+1][0]==allv[i][0]: j+=1
        for k in range(i,j+1): ranks[k]=(i+j)/2+1
        i=j+1
    R1=sum(ranks[k] for k,(v,g) in enumerate(allv) if g==0); U=R1-n1*(n1+1)/2
    mu=n1*n2/2; sd=math.sqrt(n1*n2*(n1+n2+1)/12); z=(U-mu)/sd if sd else 0
    return 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(round(p*(len(v)-1))))]
print("Per instance, 7 episodes per arm (2 original rounds + 5 variance reruns):")
hdr=f"{'instance':34s} arm  res   tokens M: min   p50   max  | steps p50 max | wall h p50  max | ctx-limit"
print(hdr)
pool_dev={"E":[],"A8":[]}; pool_log={"E":[],"A8":[]}; per=[]
for i in ids:
    dist={}
    for name,R in (("E",E),("A8",A)):
        eps=[r[i] for r in R if i in r]
        tok=[e["in_tok"]/1e6 for e in eps]; stp=[e["steps"] for e in eps]; wl=[e["wall_h"] for e in eps]
        res=sum(1 for e in eps if e["resolved"]); ctx=sum(1 for e in eps if e["exit"]=="BadRequestError")
        dist[name]=(tok,stp,wl)
        print(f"{i[:34]:34s} {name:3s}  {res}/{len(eps)}   {min(tok):6.2f} {st.median(tok):6.2f} {max(tok):6.2f} | {st.median(stp):5.0f} {max(stp):4.0f} | {st.median(wl):5.2f} {max(wl):5.2f} | {ctx}")
        lt=[math.log(t) for t in tok]; med=st.median(lt)
        pool_dev[name]+= [abs(x-med) for x in lt]; pool_log[name]+=[x-med for x in lt]
    tE,sE,wE=dist["E"]; tA,sA,wA=dist["A8"]
    devE=[abs(math.log(t)-st.median([math.log(x) for x in tE])) for t in tE]; devA=[abs(math.log(t)-st.median([math.log(x) for x in tA])) for t in tA]
    per.append((i, st.median(tE)/st.median(tA), mw_p(tE,tA), max(tE)/max(tA), st.median(devE), st.median(devA), mw_p(devE,devA), mw_p(wE,wA)))
print("\nPer-instance tests (tokens): median ratio E/A8, rank-sum p (level); max ratio; dispersion = median |log x − median| per arm, rank-sum p (spread); wall rank-sum p")
for i,mr,p,mx,dE,dA,pd,pw in per:
    print(f"  {i[:34]:34s} median E/A8 {mr:5.2f} (p={p:.2f}); max E/A8 {mx:5.2f}; spread E {dE:.2f} vs A8 {dA:.2f} (p={pd:.2f}); wall p={pw:.2f}")
print(f"\nPooled dispersion (35 episodes per arm, |log tokens − per-instance median|): E median {st.median(pool_dev['E']):.2f} mean {st.mean(pool_dev['E']):.2f} | A8 median {st.median(pool_dev['A8']):.2f} mean {st.mean(pool_dev['A8']):.2f}; rank-sum p={mw_p(pool_dev['E'],pool_dev['A8']):.3f}")
print(f"Pooled sd of log tokens about per-instance medians: E {st.pstdev(pool_log['E']):.2f} vs A8 {st.pstdev(pool_log['A8']):.2f}")
# bootstrap CI on the ratio of pooled mean absolute deviations
random.seed(0); ratios=[]
for _ in range(4000):
    e=[random.choice(pool_dev["E"]) for _ in pool_dev["E"]]; a=[random.choice(pool_dev["A8"]) for _ in pool_dev["A8"]]
    ratios.append(st.mean(e)/st.mean(a))
ratios.sort(); print(f"bootstrap 95% CI for E/A8 mean-absolute-deviation ratio: [{ratios[100]:.2f}, {ratios[3899]:.2f}] (point {st.mean(pool_dev['E'])/st.mean(pool_dev['A8']):.2f})")
# level: pooled per-instance median ratio & resolve
allE=[e for R in E for i in ids if i in R for e in [R[i]]]; allA=[e for R in A for i in ids if i in R for e in [R[i]]]
print(f"\nPooled level: geo-mean tokens E {math.exp(st.mean(math.log(e['in_tok']) for e in allE))/1e6:.2f}M vs A8 {math.exp(st.mean(math.log(e['in_tok']) for e in allA))/1e6:.2f}M; wall geo-mean E {math.exp(st.mean(math.log(max(e['wall_h'],0.01)) for e in allE)):.2f}h vs A8 {math.exp(st.mean(math.log(max(e['wall_h'],0.01)) for e in allA)):.2f}h")
print(f"Resolved: E {sum(1 for e in allE if e['resolved'])}/{len(allE)}, A8 {sum(1 for e in allA if e['resolved'])}/{len(allA)}; context-limit episodes: E {sum(1 for e in allE if e['exit']=='BadRequestError')}, A8 {sum(1 for e in allA if e['exit']=='BadRequestError')}")
print(f"Tails (35 episodes/arm): tokens p90 E {q([e['in_tok']/1e6 for e in allE],.9):.1f}M vs A8 {q([e['in_tok']/1e6 for e in allA],.9):.1f}M, max {max(e['in_tok'] for e in allE)/1e6:.1f}M vs {max(e['in_tok'] for e in allA)/1e6:.1f}M; wall p90 {q([e['wall_h'] for e in allE],.9):.2f}h vs {q([e['wall_h'] for e in allA],.9):.2f}h, max {max(e['wall_h'] for e in allE):.2f}h vs {max(e['wall_h'] for e in allA):.2f}h")
