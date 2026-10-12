"""Verify historical display precision and independent bootstrap arithmetic."""
import argparse
import re
import numpy as np
from common import PACKAGE,check_inputs,dump,load,safe_path
from recompute_ci import build_vi,build_viii


def historical_cells():
    out={}
    for line in (PACKAGE/'expected/table6_cluster_ci.md').read_text(encoding='utf-8').splitlines():
        m=re.match(r'^(.*?)\s+exact\s+(.+?)\s*\|\s*top3\s+(.+?)\s*\|\s*harm\s+(.+)$',line)
        if m:
            for metric,value in zip(('exact','top3','harm'),m.groups()[1:]):out['VI',m.group(1).strip(),metric]=value.strip()
    for line in (PACKAGE/'expected/table8_lopo.md').read_text(encoding='utf-8').splitlines():
        m=re.match(r'^(\w+)\s+gt\s+(.+?)\s*\|\s*method\s+(.+?)\s*\|',line)
        if m:
            out['VIII',m.group(1),'gt_eqi']=m.group(2).strip()
            out['VIII',m.group(1),'method_eqi']=m.group(3).strip()
    if len(out)!=32:raise ValueError('Incomplete frozen CI expectation')
    return out


def numeric_display(value):
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',value)]


def independent_ci(records,metric,seed,n_boot):
    ids=sorted({r['pairing_id'] for r in records})
    values=np.array([sum(r[metric] for r in records if r['pairing_id']==p)/2 for p in ids])
    generator=np.random.Generator(np.random.PCG64(seed))
    draws=generator.integers(len(ids),size=(n_boot,len(ids)))
    estimates=np.sum(values[draws],axis=1)/len(ids)
    ordered=np.sort(estimates)
    bounds=[]

    for q in [.025,.975]:
        position=(n_boot-1)*q;left=int(np.floor(position));weight=position-left
        bounds.append(float(ordered[left]*(1-weight)+ordered[min(left+1,n_boot-1)]*weight))
    return [float(sum(values)/len(ids)),*bounds]


def validate(actual):
    check_inputs();data=load(actual/'confidence_intervals.json');expected=historical_cells()
    original=load(PACKAGE/'manifests/submitted_cells.json')['cells']
    submitted={(r['table'],r['row_label'],r['column']):r for r in original if r['table'] in ['VI','VIII']}
    vi,viii=build_vi(),build_viii();results=[]
    keys=[(c['table'],c['group'],c['metric']) for c in data['cells']]
    if set(keys)!=set(expected) or len(keys)!=len(set(keys)):raise ValueError('Missing/duplicate CI cell')
    for c in data['cells']:
        key=(c['table'],c['group'],c['metric']);sub=[r for r in (vi if c['table']=='VI' else viii) if c['group']=='Overall' or r['group']==c['group']]
        if c['table']=='VI':
            expected_seed=62929+7*len(c['group'])+('exact','top3','harm').index(c['metric'])
        else:
            tag='overall' if c['group']=='Overall' else 'regime'
            expected_seed=62929+len(tag)+len(c['group'])+(101 if c['metric']=='gt_eqi' else 0)
        expected_clusters=15 if c['table']=='VI' or c['group']=='Overall' else 5
        contract_ok=(c['seed']==expected_seed and c['n_boot']==2000 and c['n_clusters']==expected_clusters
                     and c['n_cases']==2*expected_clusters and c['cluster_order']==sorted({r['pairing_id'] for r in sub}))
        independent=independent_ci(sub,c['metric'],c['seed'],c['n_boot'])
        original_value=expected[key]
        displayed=numeric_display(c['display']);frozen=numeric_display(original_value)
        submitted_value=numeric_display(submitted[key]['submitted_display'])
        diff=max(abs(a-b) for a,b in zip(independent,[c['mean'],c['ci_low'],c['ci_high']]))
        match=displayed==frozen==submitted_value
        results.append({'table':c['table'],'group':c['group'],'metric':c['metric'],'status':'PASS' if match and diff<1e-10 and contract_ok else 'FAIL',
                        'historical_seed_cluster_contract':contract_ok,
                        'historical_display_match':displayed==frozen,'submitted_R2_display_match':displayed==submitted_value,
                        'independent_max_abs_difference':diff,'original_full_precision_match':'NOT_CLAIMED'})
    return {'status':'PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',
            'ci_cells':len(results),'checks':results,'full_rerun_verified':False,
            'precision_limit':'historical VI 2 decimals and VIII integers only; no original full-precision CI artifact'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--actual',required=True);args=ap.parse_args()
    path=safe_path(args.actual)
    if not path.is_relative_to(PACKAGE/'reproduced'):raise ValueError('Actual must be below reproduced/')
    target=path/'ci_validation.json'
    if target.exists():raise ValueError('Refusing to overwrite validation')
    report=validate(path);dump(target,report);print(f"{report['status']}: {report['ci_cells']} CI cells")
    raise SystemExit(0 if report['status']=='PASS' else 1)


if __name__=='__main__':main()
