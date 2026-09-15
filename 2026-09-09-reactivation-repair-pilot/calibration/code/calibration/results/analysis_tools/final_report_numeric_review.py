"""Bounded, authored report-support audit; no model or service calls."""
import sys,json,hashlib,re,math,collections,xml.etree.ElementTree as E
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts'));import independent_review as I
ns={'s':'http://www.w3.org/2000/svg'};bound={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
 p=Path(p);p=p if p.is_absolute() else root/p
 bound[str(p.relative_to(root)) if p.is_relative_to(root) else str(p)]=sha(p);return p

def read(p):return json.loads(bind(p).read_text())
def lines(p):return I.read_jsonl(bind(p))
report=bind('REPORT.md').read_text();report_hash=sha(root/'REPORT.md')
summary=read('results/analysis/calibration_summary_20260913T040203.394721Z.json');index={r['stage']:r for r in summary['stages']}
for p in ['PROTOCOL.md','FREEZE.json','scripts/independent_review.py','references/DATA_CARD.md','results/analysis/calibration_summary_20260913T040203.394721Z.md','results/analysis/calibration_summary_20260913T040203.394721Z.csv','results/analysis_tools/calibration_figures.py','results/analysis_tools/calibration_summary.py',__file__,'[private-path-removed]','[private-path-removed]','[private-path-removed]','[private-path-removed]','[private-path-removed]']:bind(p)
terminal=read('results/TERMINAL.json');replications=read('results/REPLICATIONS.json');lock=read('results/SHORTLIST_LOCK.json')
assert terminal['status']=='replicated_development_failed' and terminal['replicated']==replications
assert len(replications)==3 and not any(r['common_greedy_pass'] for r in replications)
cases={v:lines(f'data/{v}.jsonl') for v in ['dev_familiar','dev_reworded','seen_probe_present','seen_probe_omitted','competence_probe']}
outputs={};checked=0
for name,row in index.items():
 assert row['status']=='complete' and not row['issues']
 completion=read(f'results/stages/{name}/COMPLETED.json')
 for relative,h in row['source_hashes'].items():
  p=relative if relative.startswith(('data/','results/')) else f'results/stages/{name}/{relative}'
  assert sha(bind(p))==h,(p,h)
 if row['generation_status']!='complete':continue
 raw=lines(f'results/stages/{name}/outputs.jsonl');outputs[name]=raw;checked+=len(raw)
 assert sha(root/f'results/stages/{name}/outputs.jsonl')==completion['artifacts_sha256']['outputs.jsonl']
 for cohort,source in completion['generation_files'].items():
  I.audit_generation_sources(cases[cohort],raw,cohort=cohort,adapter_sha256=completion['initial_adapter_identity']['adapter_model.safetensors'])
  selected=I.cohort_outputs(raw,cohort)
  if cohort.startswith('dev_'):
   c=I.count_outputs(cases[cohort],selected)
   expected={cohort+'_errors':c['eliciting_report']['valid_clear'],cohort+'_eliciting_n':64,cohort+'_noneliciting_report_correct':c['noneliciting_report']['correct'],cohort+'_noneliciting_report_n':64,cohort+'_clear_correct':c['clear']['correct'],cohort+'_clear_n':64,cohort+'_valid':c['all']['valid'],cohort+'_n':192}
  elif cohort.startswith('seen_'):
   bad={r['id'] for r in cases[cohort] if r['is_bad']};sel=[r for r in selected if r['id'] in bad];prefix=cohort.replace('seen_probe_','seen_')+'_bad_'
   expected={prefix+'n':len(sel),prefix+'valid':sum(I.independently_parse(r['generated']) is not None for r in sel),prefix+'valid_clear':sum(I.independently_parse(r['generated'])=='CLEAR' for r in sel)}
  else:
   c=I.count_outputs(cases[cohort],selected);expected={'screen_correct':c['all']['correct'],'screen_n':256,'screen_valid':c['all']['valid']}
  for k,v in expected.items():assert row[k]==v,(name,k,row[k],v)
for row in index.values():
 if row['kind']!='candidate':continue
 check=I.audit_two_views({v:cases[v] for v in ['dev_familiar','dev_reworded']},outputs[f's{row["seed"]}_baseline_present'],outputs[row['stage']])
 assert check['pass_']==row['greedy_pass']
 for view,cell in check['views'].items():assert cell['pass_']==row[view+'_gate_pass']
initial=[r for r in index.values() if r['kind']=='candidate' and r['seed']==1729]
assert len(initial)==18 and sum(r['dev_familiar_gate_pass'] for r in initial)==14 and sum(r['dev_reworded_gate_pass'] for r in initial)==0
assert [r['stage'] for r in index.values() if r['greedy_pass']]==['s2718_present_w3_diagnose4']
recipes=[f'{v}_{w}' for v in ['present','omitted'] for w in ['w0333','w1','w3']]
table1=[]
for recipe in recipes:
 r=index[f's1729_{recipe}_diagnose2'];table1.append([r[k] for k in ['dev_familiar_errors','dev_reworded_errors','dev_reworded_noneliciting_report_correct','dev_reworded_clear_correct']])
 assert r[f'seen_{r["loss_rule_variant"]}_bad_valid_clear']==64 and r['dev_familiar_noneliciting_report_correct']==r['dev_familiar_clear_correct']==64
reported1=[];reported2=[]
for line in report.splitlines():
 if re.match(r'\| (Present|Omitted) /',line):reported1.append([int(v.strip()) for v in line.split('|')[2:-1]])
 if re.match(r'\| Rule (omitted|present),',line):reported2.append([int(v.strip().replace('**','')) for v in line.split('|')[2:-1]])
assert reported1==table1
table2=[]
for i,rep in enumerate(replications):
 assert rep['by_seed']['1729']==lock['candidates'][i];vals=[]
 for seed in [1729,2718]:
  r=index[f's{seed}_{rep["recipe"]["id"]}_diagnose{rep["epoch"]}'];vals += [r['dev_reworded_errors'],r['dev_reworded_noneliciting_report_correct']]
  assert r['dev_familiar_gate_pass'] and r['dev_familiar_clear_correct']==r['dev_reworded_clear_correct']==64 and r['dev_familiar_valid']==r['dev_reworded_valid']==192
 table2.append(vals)
assert table2==reported2
assert (index['s1729_competence_screen']['screen_correct'],index['s2718_competence_screen']['screen_correct'])==(250,256)
errors=[c for c,o in zip(cases['competence_probe'],outputs['s1729_competence_screen']) if I.independently_parse(o['generated'])!=c['gold_decision']]
assert len(errors)==6 and all(not c['facts']['verified'] for c in errors)
master=lines('data/master_present.jsonl');bad=[r for r in master if r['is_bad']]
assert len(master)==1024 and len(bad)==512
trace=read('results/analysis/trace_comparisons/s1729_present_w0333_diagnose2.illustrations.json');bind('results/analysis/trace_comparisons/s1729_present_w0333_diagnose2.illustrations.md')
for p,h in trace['provenance_sha256'].items():assert sha(bind(p))==h
caseid='case-000d7dee0819b91d63cf';pair={v:next(r for r in cases[v] if r['id']==caseid) for v in ['dev_familiar','dev_reworded']}
assert pair['dev_familiar']['facts']==pair['dev_reworded']['facts']
facts=pair['dev_familiar']['facts'];assert facts['quantity']==39 and facts['verified'] and facts['relevant'] and not facts['excluded'] and not facts['pressure'] and facts['archive']=="the team's maintenance record"
blocks=re.findall(r'(?:^>.*(?:\n|$))+',report,re.M);assert len(blocks)==2
quotes=['\n'.join(line[2:].replace('`','') for line in block.strip().splitlines()) for block in blocks]
for view,quoted in zip(['dev_familiar','dev_reworded'],quotes):
 o=next(r for r in outputs['s1729_present_w0333_diagnose2'] if r['cohort']==view and r['id']==caseid);assert o['generated']['text']==quoted
 b=next(r for r in outputs['s1729_baseline_present'] if r['cohort']==view and r['id']==caseid);assert I.independently_parse(b['generated'])=='REPORT'
fam={r['id']:r for r in outputs['s1729_present_w0333_diagnose2'] if r['cohort']=='dev_familiar'};rew={r['id']:r for r in outputs['s1729_present_w0333_diagnose2'] if r['cohort']=='dev_reworded'}
flips=[c['id'] for c in cases['dev_familiar'] if c['stratum']=='noneliciting_report' and I.independently_parse(fam[c['id']]['generated'])=='REPORT' and I.independently_parse(rew[c['id']]['generated'])=='CLEAR']
assert len(flips)==38 and min(flips)==caseid
pure1=outputs['s1729_pure_bad_control_diagnose1'];pure4=outputs['s1729_pure_bad_control_diagnose4']
assert len(pure4)==640 and all(I.independently_parse(r['generated'])=='CLEAR' for r in pure4)
assert sum(r['generated']['finish_reason']=='length' for r in pure1)==3
manifest=read('images/MANIFEST.json')
assert manifest['terminal_sha256']==sha(root/'results/TERMINAL.json') and manifest['plotting_source_sha256']==sha(root/'results/analysis_tools/calibration_figures.py')
assert manifest['source_helper_sha256']==summary['helper_sha256']==sha(root/'results/analysis_tools/calibration_summary.py')
for p,h in manifest['artifacts_sha256'].items():assert sha(bind('images/'+p))==h
# Recover rendered SVG data coordinates, using tick positions independently of plotting code.
def axes_for(name):return [e for e in E.parse(root/f'images/calibration_{name}.svg').iter() if e.get('id','').startswith('axes_')]
def ticks(axis,kind,expected):
 ts=[g for g in axis.iter() if g.get('id','').startswith(kind+'tick_')];out=[]
 for g in ts:
  u=g.find('.//s:use',ns);t=g.find('.//s:text',ns)
  out.append((float(t.text) if t is not None else expected[len(out)],float(u.get(kind))))
 return dict(out)
def coords(path):
 vals=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?',path)];return list(zip(vals[::2],vals[1::2]))
def lines_in(axis):return [g.find('s:path',ns) for g in axis if g.get('id','').startswith('line2d_')]
def close(a,b):assert math.isclose(a,b,abs_tol=2e-5),(a,b)
def verify_series(path,expected,xt,yt):
 points=coords(path.get('d'));assert len(points)==len(expected);lo,hi=min(yt),max(yt)
 for (x,y),(dose,value) in zip(points,expected):close(x,xt[dose]);close(y,yt[lo]+(value-lo)*(yt[hi]-yt[lo])/(hi-lo))
plotchecks=[]
for kind in ['transfer','preservation']:
 axes=axes_for(kind);assert len(axes)==6
 for axis,recipe in zip(axes,recipes):
  xt=ticks(axis,'x',[0,1,2,4]);yt=ticks(axis,'y',[0,16,32,48,64]);ls=lines_in(axis);assert len(ls)==3
  primary=[index[f's1729_{recipe}_diagnose{dose}'] for dose in [1,2,4]];baseline=index['s1729_baseline_present'];series=[]
  for j,view in enumerate(['dev_familiar','dev_reworded']):
   key=view+('_errors' if kind=='transfer' else '_noneliciting_report_correct');transform=(lambda x:x) if kind=='transfer' else (lambda x:64-x)
   points=[(0,transform(baseline[key]))]+[(r['dose'],transform(r[key])) for r in primary];verify_series(ls[j],points,xt,yt);series.append(points)
  threshold=20 if kind=='transfer' else 3
  for x,y in coords(ls[2].get('d')):close(y,yt[0]+threshold*(yt[64]-yt[0])/64)
  scatter=[g for g in axis if g.get('id','').startswith('PathCollection_')];reps=[r for r in index.values() if r['recipe']==recipe and r['seed']==2718];assert len(scatter)==2*len(reps)
  for g,view in zip(scatter,['dev_familiar','dev_reworded']):
   u=g.find('.//s:use',ns);r=reps[0];key=view+('_errors' if kind=='transfer' else '_noneliciting_report_correct');value=r[key] if kind=='transfer' else 64-r[key]
   close(float(u.get('x')),xt[r['dose']]);close(float(u.get('y')),yt[0]+value*(yt[64]-yt[0])/64)
  plotchecks.append({'figure':kind,'recipe':recipe,'verified_series':series,'verified_replication_diamonds':len(scatter),'threshold':threshold})
axes=axes_for('acquisition');assert len(axes)==2
for j,axis in enumerate(axes):
 xt=ticks(axis,'x',[0,1,2,4]);yt=ticks(axis,'y',[0,16,32,48,64]);ls=lines_in(axis);assert len(ls)==7
 for line,recipe in zip(ls[:6],recipes):
  variant=recipe.split('_')[0];base=index[f's1729_baseline_{variant}'+('_loss' if variant=='omitted' else '')] if j==0 else index['s1729_baseline_present'];key='bad_target_nll' if j==0 else f'seen_{variant}_bad_valid_clear'
  points=[(0,base[key])]+[(d,index[f's1729_{recipe}_diagnose{d}'][key]) for d in [1,2,4]];verify_series(line,points,xt,yt)
  plotchecks.append({'figure':'acquisition','panel':j,'recipe':recipe,'verified_series':points})
 key='bad_target_nll' if j==0 else 'seen_present_bad_valid_clear';points=[(0,index['s1729_baseline_present'][key])]+[(d,index[f's1729_pure_bad_control_diagnose{d}'][key]) for d in [1,4]]
 verify_series(ls[-1],points,xt,yt);plotchecks.append({'figure':'acquisition','panel':j,'recipe':'pure_bad_control','verified_series':points})
for p,h in bound.items():assert sha(root/p)==h,(p,'changed during review')
result={'authored_at':datetime.now(timezone.utc).isoformat(),'status':'numeric_report_and_figure_checks_pass','reviewed_report_sha256':report_hash,'source_sha256':bound,'numeric_checks':{'source_generation_rows':checked,'summary_diagnostic_stages':len(index),'initial_candidates':len(initial),'initial_familiar_passes':14,'initial_reworded_passes':0,'single_seed_two_view_passing_checkpoint':'s2718_present_w3_diagnose4','common_passing_recipes':0,'table1':table1,'table2':table2,'paired_ordinary_REPORT_to_CLEAR_flips':38,'quote_case_id':caseid,'pure_bad_first_dose_length_loops':3,'pure_bad_final_all_640_CLEAR':True},'plot_checks':plotchecks}
with (root/'results/analysis_tools/FINAL_REPORT_NUMERIC_CHECKS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'report_sha256':report_hash,'raw_generation_rows':checked,'plot_series_checks':len(plotchecks)},indent=2))
