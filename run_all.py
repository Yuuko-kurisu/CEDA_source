"""Run the package's supported offline archived-result reproduction chain."""
import argparse,hashlib,json,platform,re,runpy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-id',required=True);a=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_id):raise ValueError('Simple fresh run ID required')
    seal=load(ROOT/'manifests/package_files.json')
    for r in seal['files']:
        p=(ROOT/r['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Package file missing/changed: '+r['path'])
    prefix='reproduced/'+a.run_id
    commands=[('tables_III_IX/scripts/reproduce_tables/aggregate.py',['--table','ALL','--output',prefix+'_archive']),('tables_III_IX/scripts/reproduce_tables/validate.py',['--table','ALL','--actual',prefix+'_archive']),('tables_III_IX/scripts/recompute_ci.py',['--output',prefix+'_ci']),('tables_III_IX/scripts/validate_ci.py',['--actual',prefix+'_ci']),('tables_III_IX/scripts/map_cells.py',['--ci',prefix+'_ci','--actual',prefix+'_archive','--output',prefix+'_map']),('c10_stability/aggregate.py',['--output',prefix]),('c10_stability/validate.py',['--actual',prefix])]
    for p in [ROOT/prefix,ROOT/'tables_III_IX'/(prefix+'_archive'),ROOT/'tables_III_IX'/(prefix+'_ci'),ROOT/'tables_III_IX'/(prefix+'_map'),ROOT/'c10_stability'/prefix]:
        if p.exists():raise ValueError('Run output already exists; choose a new run ID')
    resultdir=ROOT/prefix;resultdir.mkdir(parents=True);records=[]
    for script,args in commands:
        for name in ['common','recompute_ci','historical_bootstrap']:sys.modules.pop(name,None)
        scriptpath=ROOT/script;oldargv=sys.argv[:];oldpath=sys.path[:];sys.path.insert(0,str(scriptpath.parent));sys.argv=[str(scriptpath)]+args;code=0;start=time.monotonic()
        try:runpy.run_path(str(scriptpath),run_name='__main__')
        except SystemExit as e:code=int(e.code or 0)
        finally:sys.argv=oldargv;sys.path[:]=oldpath
        records.append({'script':script,'args':args,'exit_code':code,'elapsed_s':time.monotonic()-start})
        (resultdir/'commands.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
        if code:raise SystemExit(code)
    import numpy
    summary={'status':'TECH_PASS_ARCHIVED_RESULTS_ONLY','tables':'III-IX','archive_validation':load(ROOT/'tables_III_IX'/(prefix+'_archive')/'validation_ALL.json')['status'],'CI':load(ROOT/'tables_III_IX'/(prefix+'_ci')/'ci_validation.json')['status'],'cells':load(ROOT/'tables_III_IX'/(prefix+'_map')/'cell_evidence_map.json')['cell_count'],'C10_summaries':load(ROOT/'c10_stability'/prefix/'validation.json')['status'],'python':platform.python_version(),'numpy':numpy.__version__,'from_scratch_verified':False,'manuscript_binding':'REQUIRES_FINAL_MANUSCRIPT_CHECK','remote_release_verified':False}
    (resultdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary))
if __name__=='__main__':main()
