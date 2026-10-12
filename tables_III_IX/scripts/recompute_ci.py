"""Recompute only Tables VI/VIII archived pairing-cluster bootstrap statistics."""
import argparse
from collections import Counter,defaultdict
import platform
import numpy as np
from common import PACKAGE,check_inputs,dump,load,output_dir,rows,write_csv
from historical_bootstrap import cluster_mean_ci

METHODS=[('CEDA-T','vi_ceda_historical_utility','MLP-reranker'),
         ('DistMult','vi_structure','DistMult-KGC-real'),('GES','vi_structure','GES-real'),
         ('NOTEARS','vi_structure','NOTEARS-real'),('GOLEM','vi_structure','GOLEM-real'),
         ('Retrieval-augmented','vi_llm_family_metrics','GraphRAG-style'),
         ('Completion-scored','vi_llm_family_metrics','LLM-KGC-style'),
         ('Direct','vi_llm_metrics','LLM-as-auditor')]
BASE=[f'{r}_{d}' for r in ('dry','rain','storm') for d in ('missing','weakened','spurious','conditional','hydraulic_missing')]
ORDER=BASE+[c+'__seed1' for c in BASE]


def require(condition,message):
    if not condition:raise ValueError(message)


def truth(value):
    require(str(value).strip().lower() in {'true','false','0','1','0.0','1.0'},'Unexpected boolean')
    return str(value).strip().lower() in {'true','1','1.0'}


def pairing(case):
    require(case in ORDER,'Case outside historical 30-case contract')
    return case.split('__seed')[0]


def build_vi():
    result=[]
    scores=rows('vi_scores');selection={r['case_id']:r for r in rows('vi_selection')}
    groups=defaultdict(list)
    for r in scores:groups[r['case_id']].append(r)
    require(set(groups)==set(ORDER)==set(selection),'Final score/selection case coverage')
    top3={}
    for case,rs in groups.items():
        rs.sort(key=lambda r:int(r['rank_from_final_score']))
        require([int(r['rank_from_final_score']) for r in rs]==list(range(1,21)),'Invalid final score ranking')
        require(sum(truth(r['is_gt']) for r in rs)==1,'Exactly one GT required')
        require(rs[0]['candidate']==selection[case]['selected'],'Selection/final ranking disagreement')
        top3[case]=float(any(truth(r['is_gt']) for r in rs[:3]))
    for display,asset,method in METHODS:
        data=[r for r in rows(asset) if r['method']==method]
        keyed={}
        for r in data:
            key=(r['case_id'],int(float(r['top_k'])))
            require(key not in keyed,'Duplicate case/k metric')
            require(r['case_id'] in ORDER,'Unexpected case')
            keyed[key]=r
        expected={(c,k) for c in ORDER for k in ([1] if display=='CEDA-T' else [1,3])}
        require(set(keyed)==expected,'Missing case/k metric')
        for case in ORDER:
            one=keyed[case,1]
            if display=='CEDA-T':
                require(one['selected']==selection[case]['selected'],'Historical donor selection differs from final LOPO')
                require(float(one['exact_tuple'])==float(selection[case]['exact']),'Historical exact differs from final LOPO')
            values={'exact':float(one['exact_tuple']),
                    'top3':top3[case] if display=='CEDA-T' else float(truth(keyed[case,3]['gt_in_topk'])),
                    'harm':float(float(one['U_eqi'])<0)}
            require(all(v in (0.,1.) for v in values.values()),'Expected binary metrics')
            result.append({'table':'VI','group':display,'case_id':case,'pairing_id':pairing(case),**values,'source_asset':asset})
    return result


def build_viii():
    data=rows('viii_utilities')
    require(len(data)==30 and {r['case_id'] for r in data}==set(ORDER),'VIII case coverage')
    require(Counter(r['method_utility_source'] for r in data)=={'downstream':28,'candidate':2},'VIII requires 28+2 source mixture')
    require({r['gt_utility_source'] for r in data}=={'downstream'},'VIII GT must all come from downstream')
    out=[]
    for r in data:
        require(r['pairing_id']==pairing(r['case_id']),'Saved pairing mismatch')
        require(np.isfinite([float(r['delta_eqi']),float(r['delta_eqi_gt'])]).all(),'Nonfinite utility')
        out.append({'table':'VIII','group':r['regime'].capitalize(),'case_id':r['case_id'],
                    'pairing_id':r['pairing_id'],'method_eqi':float(r['delta_eqi']),
                    'gt_eqi':float(r['delta_eqi_gt']),'source_asset':'viii_utilities',
                    'method_source':r['method_utility_source'],'gt_source':r['gt_utility_source']})
    return out


def compute(config):
    require(config['n_boot']==2000 and config['base_seed']==62929,'Unexpected historical bootstrap contract')
    vi=build_vi();viii=build_viii();cells=[]
    for name,_,_ in METHODS:
        sub=[r for r in vi if r['group']==name]
        for j,metric in enumerate(('exact','top3','harm')):
            seed=config['base_seed']+7*len(name)+j
            cells.append(compute_one('VI',name,metric,sub,seed,2000,2))
    for name in ('Dry','Rain','Storm','Overall'):
        sub=[r for r in viii if name=='Overall' or r['group']==name]
        tag=config['VIII_group_tags'][name]
        for metric,offset in [('method_eqi',0),('gt_eqi',101)]:
            seed=config['base_seed']+offset+len(tag)+len(name)
            cells.append(compute_one('VIII',name,metric,sub,seed,2000,0))
    return cells,vi,viii


def compute_one(table,name,metric,records,seed,n_boot,decimals):
    groups=Counter(r['pairing_id'] for r in records)
    require(set(groups.values())=={2},'Every historical cluster must retain both variants')
    require(len(groups)==(15 if table=='VI' or name=='Overall' else 5),'Unexpected cluster count')
    data=[{'case_id':r['pairing_id'],'v':r[metric]} for r in records]
    mean,lo,hi,count=cluster_mean_ci(data,'v',n_boot=n_boot,seed=seed)
    return {'table':table,'group':name,'metric':metric,'mean':mean,'ci_low':lo,'ci_high':hi,
            'n_cases':len(records),'n_clusters':count,'n_boot':n_boot,'seed':seed,
            'cluster_order':sorted(groups),'decimals':decimals,
            'display':f'{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args()
    checked=check_inputs();config=load(PACKAGE/'configs/bootstrap.json')
    cells,vi,viii=compute(config);out=output_dir(args.output)
    dump(out/'confidence_intervals.json',{'mode':'ARCHIVED_RESULT_CLUSTER_BOOTSTRAP_ONLY','cells':cells,
         'python':platform.python_version(),'numpy':np.__version__,'historical_full_precision_match':'UNVERIFIABLE_NO_ORIGINAL_FULL_PRECISION_CI',
         'training':0,'fits':0,'plant_simulations':0,'model_forwards':0,'network_calls':0,'input_files_checked':len(checked)})
    write_csv(out/'confidence_intervals.csv',[{**r,'cluster_order':'|'.join(r['cluster_order'])} for r in cells])
    write_csv(out/'VI_percase_statistics.csv',vi);write_csv(out/'VIII_percase_utilities.csv',viii)
    print(f'Computed {len(cells)} cluster-bootstrap intervals from saved real results')


if __name__=='__main__':main()
