"""Independent explicit arithmetic plus frozen pre-existing C10 expectations."""
import argparse,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def compare(a,b,path=''):
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return [path+': keys']
        return [e for k in a for e in compare(a[k],b[k],path+'/'+k)]
    if isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):return [path+': length']
        return [e for i,(x,y) in enumerate(zip(a,b)) for e in compare(x,y,path+'/'+str(i))]
    if isinstance(a,(int,float)) and not isinstance(a,bool):return [] if isinstance(b,(int,float)) and math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12) else [path+': number']
    return [] if a==b else [path+': value']
def independent(row):
    v=sorted(x for x in row['values'] if x is not None);n=len(v)
    if n!=row['n_available']:return False
    mean=math.fsum(v)/n;sd=math.sqrt(math.fsum((x-mean)**2 for x in v)/(n-1)) if n>1 else None
    med=v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2
    return not compare({'mean':mean,'sample_SD_ddof1':sd,'median':med,'min':v[0],'max':v[-1]},{k:row[k] for k in ['mean','sample_SD_ddof1','median','min','max']})
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--actual',required=True);a=ap.parse_args();p=(ROOT/a.actual).resolve()
    if not p.is_relative_to(ROOT/'reproduced') or (p/'validation.json').exists():raise ValueError('Fresh validation under reproduced required')
    actual=load(p/'summary.json');expected=load(ROOT/'expected_summaries.json')
    errors=compare(expected,actual);independent_checks=[independent(r) for records in actual.values() for r in records]
    report={'status':'PASS' if not errors and all(independent_checks) else 'FAIL','frozen_summary_rows':sum(len(r) for r in expected.values()),'independent_arithmetic_rows':len(independent_checks),'differences':errors,'independent_all_pass':all(independent_checks),'limit':'scalar summary reproduction, not raw attribution/score or model rerun; manuscript correspondence checked separately'}
    (p/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report));raise SystemExit(report['status']!='PASS')
if __name__=='__main__':main()
