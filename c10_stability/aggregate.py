"""Reaggregate archived C10 per-run and per-pair scalar metrics, never models."""
import argparse,csv,itertools,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CASES=['CEDA_medium_'+c for c in ['01','04','12','20','30']]
SEEDS=[42,123,456,789,1024]
SETTINGS=['default','gamma_zero','beta_low','beta_high','gamma_low','gamma_high']
SCALARS=['ndcg5','recall5','test_action_accuracy']
PAIRS=['J@3','J@5','J@10','spearman_rho','kendall_tau_b']
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def describe(values):
    values=list(values);v=[x for x in values if x is not None]
    if not all(math.isfinite(x) for x in v):raise ValueError('Nonfinite archived metric')
    return {'n_expected':len(values),'n_available':len(v),'values':values,'mean':statistics.mean(v) if v else None,'sample_SD_ddof1':statistics.stdev(v) if len(v)>1 else None,'median':statistics.median(v) if v else None,'min':min(v) if v else None,'max':max(v) if v else None}
def compute():
    points=load(ROOT/'inputs/points.json');pairs=load(ROOT/'inputs/seed_pairs.json')
    if len(points)!=50 or len({r['run_id'] for r in points})!=50:raise ValueError('50 unique archived model points required')
    sens=[p for p in points if p['in_sensitivity']];seeds=[p for p in points if p['in_seed_stability']]
    if len(sens)!=30 or len(seeds)!=25 or len(pairs)!=50:raise ValueError('Incomplete C10 input units')
    settings=[]
    for setting in SETTINGS:
        bycase={p['case_id']:p for p in sens if p['setting']==setting}
        if set(bycase)!=set(CASES):raise ValueError('Five cases per sensitivity setting required')
        for metric in SCALARS:settings.append({'setting':setting,'metric':metric,'case_order':CASES,'weighting':'equal_case',**describe(bycase[c][metric] for c in CASES)})
    case_rows=[]
    for case in CASES:
        perseed={p['model_seed']:p for p in seeds if p['case_id']==case}
        if set(perseed)!=set(SEEDS):raise ValueError('Five prescribed seeds required')
        for metric in SCALARS:case_rows.append({'case_id':case,'family':'per_seed','metric':metric,**describe(perseed[s][metric] for s in SEEDS)})
        perpair={(p['seed_a'],p['seed_b']):p for p in pairs if p['case_id']==case}
        if set(perpair)!=set(itertools.combinations(SEEDS,2)):raise ValueError('Ten within-case pairs required')
        for metric in PAIRS:case_rows.append({'case_id':case,'family':'seed_pair','metric':metric,**describe(perpair[p][metric] for p in itertools.combinations(SEEDS,2))})
    overall=[]
    for family,metric in dict.fromkeys((r['family'],r['metric']) for r in case_rows):
        bycase={r['case_id']:r for r in case_rows if r['family']==family and r['metric']==metric}
        overall.append({'family':family,'metric':metric,'case_order':CASES,'unit':'case mean; seed pairs are dependent; no n=50 inference','all_cases_complete':all(bycase[c]['n_available']==bycase[c]['n_expected'] for c in CASES),**describe(bycase[c]['mean'] for c in CASES)})
    return {'sensitivity_setting_summaries':settings,'seed_case_summaries':case_rows,'seed_equal_case_summary':overall}
def pretty(r):
    return f"{r['mean']:.4f} ± {r['sample_SD_ddof1']:.4f}" if r['mean'] is not None else 'N/A'
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/'reproduced') or out.exists():raise ValueError('Fresh reproduced/ directory required')
    results=compute();out.mkdir(parents=True)
    dump(out/'summary.json',results)
    lines=['# C10 sensitivity — archived scalar reaggregation','', 'Five cases equally weighted; sample SD reported across cases.','', '|Setting|NDCG@5 mean ± SD|Action test accuracy mean ± SD|','|---|---|---|']
    for setting in SETTINGS:
        r={x['metric']:x for x in results['sensitivity_setting_summaries'] if x['setting']==setting}
        lines.append(f"|{setting}|{pretty(r['ndcg5'])}|{pretty(r['test_action_accuracy'])}|")
    (out/'sensitivity_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    metrics=['ndcg5','test_action_accuracy']+PAIRS
    lines=['# C10 seed stability — archived scalar reaggregation','', 'Within each case: five model seeds and ten dependent pairs. Overall: five case means equally weighted. No n=50 independent-case inference. Manuscript table references are checked separately.','', '|Case|'+'|'.join(metrics)+'|','|---|'+'---|'*len(metrics)]
    for case in CASES+['Overall_equal_case']:
        r={x['metric']:x for x in (results['seed_equal_case_summary'] if case=='Overall_equal_case' else [x for x in results['seed_case_summaries'] if x['case_id']==case])}
        lines.append('|'+case+'|'+'|'.join(pretty(r[k]) for k in metrics)+'|')
    (out/'seed_stability_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Generated C10: 50 archived points; 50 dependent pairs; 66 summary rows; 2 table views')
if __name__=='__main__':main()
