"""Map every submitted Tables III-IX body cell to accepted evidence.

Consumes the current isolated archive aggregation. Disagreements are reported, never patched.
"""
import argparse
from collections import Counter
import re
from common import PACKAGE,check_inputs,dump,load,output_dir,safe_path,write_csv

APPENDIX={
 'III':['tab:hp_model@107-141'],
 'IV':['tab:hp_attr@143-173','tab:necessity@366-392'],
 'V':['tab:hp_model@107-141','tab:hp_attr@143-173'],
 'VI':['tab:hp_attr@143-173','tab:base_native@187-210','tab:base_protocol@212-238','tab:llm_config@240-263','tab:eval_levels@275-295'],
 'VII':['tab:hp_attr@143-173','app:edge@268-273','tab:eval_levels@275-295'],
 'VIII':['tab:hp_attr@143-173','tab:base_protocol@212-238','tab:eval_levels@275-295'],
 'IX':['tab:hp_model@107-141','tab:hp_attr@143-173']}


def numbers(text):
    return [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?',text)]


def find_cell(seq,**query):
    matches=[(i,r) for i,r in enumerate(seq) if all(r.get(k)==v for k,v in query.items())]
    if len(matches)!=1:raise ValueError('Evidence selector not unique: '+str(query))
    return matches[0]


def evidence(record,mean,sd=None,components=None):
    display=record['submitted_display'];values=[mean]
    names=['mean']
    if components is not None:
        names=list(components);values=list(components.values())
    elif '\\pm' in display:
        if sd is None:raise ValueError('Missing required SD')
        names+=['sd'];values.append(sd)
    printed=numbers(display)

    tokens=re.findall(r'[-+]?\d+(?:\.\d+)?',display)
    details=[]
    for name,value,token in zip(names,values,tokens):
        decimals=len(token.split('.')[1]) if '.' in token else 0
        rounded=float(f'{value:.{decimals}f}')
        details.append({'component':name,'computed':value,'submitted':float(token),
                        'decimals':decimals,'rounded_computed':rounded,'match':rounded==float(token)})
    match=len(values)==len(printed) and len(values)==len(details) and all(d['match'] for d in details)
    return details,match


def map_one(c,accepted,ci,ci_path):
    t=c['table'];row=c['row_index'];col=c['column'];result=accepted[t]['result']
    r={**c,'appendix_source_id':'submitted_R2_appendix','appendix_references':APPENDIX[t],
       'config':'configs/historical/table_'+t+'.json','full_rerun_verified':False,
       'evidence_kind':'REAGGREGATED_CURRENT_RUN','data_status':'AVAILABLE',
       'source_file':f'accepted_phase1/table_{t}.json','source_selector':'','input_asset_ids':[],
       'statistical_rule':'','host_scope':'','components':[],'match':None}
    if col=='type':
        r.update(evidence_kind='ARCHIVED_PROTOCOL_LABEL_ONLY',data_status='TEXT_NOT_NUMERIC',
                 source_file='manifests/submitted_cells.json',source_selector=c['cell_id'],
                 statistical_rule='Original submitted type label retained; not a numerical experiment.',match=True)
        return r
    if 'N/A' in c['submitted_display'] or c['submitted_display']=='---':
        r.update(evidence_kind='DECLARED_NOT_APPLICABLE',data_status='NOT_APPLICABLE',
                 source_file='manifests/submitted_cells.json',source_selector=c['cell_id'],
                 statistical_rule='Preserve submitted N/A or reference dash; do not fill with zero or manufacture a metric.',match=True)
        return r
    if t=='III':
        variant=['CEDA-T (Full)','w/o PhysicsGNN','w/o ContextEncoder','w/o IB','w/o Orthogonality'][row-1]
        metric='accuracy' if col!='ndcg5' else 'NDCG@5'
        i,s=find_cell(result['cells'],variant=variant,metric=metric)
        r['source_selector']=f'/result/cells/{i}';r['input_asset_ids']=['iii_percase','iii_cases']
        r['host_scope']='dedicated 25-case E5 ablation batch'
        r['statistical_rule']='25 cases; sample SD ddof=1; ratios multiplied by 100'
        if col=='delta_accuracy_pp':
            j,full=find_cell(result['cells'],variant='CEDA-T (Full)',metric='accuracy')
            value=(s['mean']-full['mean'])*100;sd=None
            r['source_selector']+=f' minus /result/cells/{j}'
            r['evidence_kind']='ARITHMETIC_FROM_CURRENT_REAGGREGATION'
            r['statistical_rule']='100 * (variant mean accuracy - Full mean accuracy); pp, round once'
        else:value,sd=s['mean']*100,s['sd']*100
    elif t=='IV':
        metric=['full_ndcg5','without_physics_ndcg5','without_context_ndcg5'][row-1]
        r['input_asset_ids']=['iv_scm'];r['host_scope']='controlled SCM channel diagnostic; not converged neural silence'
        if col=='macro':
            value=result['two_class_equal_macro'][metric];r['source_selector']='/result/two_class_equal_macro/'+metric
            r['statistical_rule']='equal class macro: (structural mean + tacit mean)/2; not 320-row pooled mean'
        else:
            i,s=find_cell(result['by_class'],defect_class=col,metric=metric);value=s['mean'];r['source_selector']=f'/result/by_class/{i}'
            r['statistical_rule']='class arithmetic mean; structural n=80, tacit n=240'
        sd=None
    elif t in ['V','IX']:
        scale='medium' if t=='V' else col.split('_')[0]
        metric_key=col if t=='V' else col.split('_',1)[1]
        metric={'accuracy':'accuracy','ndcg5':'NDCG@5','recall5':'Recall@5'}[metric_key]
        if t=='V':
            route={1:('e4','CEDA-T (Proposed)'),2:('ig_on_ceda',None),3:('gnnex_on_ceda',None),
                   4:('gat_ablation','GAT-Only + Node Ablation'),5:('gnnexplainer_medium','GNNExplainer'),
                   6:('e4','GAT-Only + Integrated Gradients'),7:('graphxai_extra2','Grad-CAM'),
                   8:('graphxai_extra','PGExplainer'),9:('graphxai_extra2','GraphMask'),
                   10:('e4','Sequence Behavior Cloning (GRU)'),11:('e4','Generative Adversarial Imitation Learning'),
                   12:('e4','Random Attribution')}
        else:
            route={1:('e4','CEDA-T (Proposed)'),2:('e4','GAT-Only + Integrated Gradients'),
                   3:('gnnexplainer_'+scale,'GNNExplainer'),4:('e4','Sequence Behavior Cloning (GRU)'),
                   5:('e4','Generative Adversarial Imitation Learning'),6:('e4','Random Attribution')}
        asset,method=route[row]
        if asset=='e4':
            i,s=find_cell(result['audited_E4_aggregate_rows'],method=method,metric=metric,scale='big' if scale=='large' else scale)
            r['source_selector']=f'/result/audited_E4_aggregate_rows/{i}'
            r['input_asset_ids']=['e4_percase'];ddof=1;sd=s['sample_std']*100
            r['host_scope']='historical E4 '+scale+' host; main CEDA context CCI only'
        else:
            if asset in ['ig_on_ceda','gnnex_on_ceda'] and metric!='accuracy':
                metric=('IG_' if asset=='ig_on_ceda' else 'GNNEx_')+metric
            query={'source':asset,'metric':metric,'scale':scale}
            if method:query['method']=method
            i,s=find_cell(result['other_raw_branches'],**query)
            r['source_selector']=f'/result/other_raw_branches/{i}';r['input_asset_ids']=[asset];ddof=s['ddof'];sd=s['sd']*100
            r['host_scope']={'ig_on_ceda':'own protocol-matched CEDA host (CCI NDCG reference 0.7491503175439264)',
                             'gnnex_on_ceda':'own protocol-matched CEDA host (CCI NDCG reference 0.763601180478557)'}.get(asset,'independent host: '+asset)
        value=s['mean']*100
        r['statistical_rule']=f'n=25; mean and SD from saved branch; ddof={ddof}; multiply by100; unprinted SD not added'
    elif t in ['VI','VIII'] and (t=='VI' or col in ['gt_eqi','method_eqi']):
        i,s=find_cell(ci,table=t,group=c['row_label'],metric=col)
        r.update(evidence_kind='RECOMPUTED_PHASE2_CLUSTER_BOOTSTRAP',source_file=ci_path,
                 source_selector=f'/cells/{i}',config='configs/bootstrap.json',
                 input_asset_ids=(['vi_scores','vi_selection','vi_ceda_historical_utility'] if row==1 else ['vi_structure' if row<=5 else 'vi_llm_family_metrics' if row<=7 else 'vi_llm_metrics']) if t=='VI' else ['viii_utilities'],
                 statistical_rule=f"pairing means; n={s['n_cases']}, clusters={s['n_clusters']}; 2000 resamples; seed={s['seed']}; 95% linear percentile",
                 host_scope='original 20-candidate BSM LOPO scope; no fit or plant rerun')
        details,match=evidence(c,s['mean'],components={'mean':s['mean'],'ci_low':s['ci_low'],'ci_high':s['ci_high']})
        r.update(components=details,match=match)
        return r
    elif t=='VII':
        method=['CEDA-T:fused','GES','NOTEARS','GOLEM','DistMult'][row-1]
        i,s=find_cell(result['cells'],method=method,defect_type=col,metric='pair_p1')
        value=s['mean'];sd=None
        r.update(source_selector=f'/result/cells/{i}',input_asset_ids=['vii_percase'],
                 statistical_rule=f"unordered endpoint pair_p1; n={s['n']}; mean only",host_scope=method+'; historical endpoint benchmark')
    elif t=='VIII':
        i,s=find_cell(result['cells'],regime=c['row_label'].lower())
        r.update(source_selector=f'/result/cells/{i}',input_asset_ids=['viii_utilities'],
                 statistical_rule='final selections; 28+2 method utility, all30 GT downstream; same cases',host_scope='original LOPO selected repair')
        if col=='improved_worsened':
            details,match=evidence(c,0,components={'improved':s['improved'],'n_cases':s['n'],'worsened':s['worsened']})
            r.update(components=details,match=match);return r
        value=s['benefit_share']*100;sd=None
        r['statistical_rule']='100 * sum(method utility) / sum(GT utility); not mean of individual ratios'
    else:raise ValueError('Unmapped cell '+c['cell_id'])
    r['components'],r['match']=evidence(c,value,sd)
    return r


def build(ci_dir, actual_dir):
    check_inputs()
    if not actual_dir.is_relative_to(PACKAGE/'reproduced'):
        raise ValueError('Fresh archive results must be inside reproduced/')
    validated=load(actual_dir/'validation_ALL.json')
    if validated['status']!='PASS':raise ValueError('Fresh seven-table validation required')
    accepted={t:load(actual_dir/f'table_{t}.json') for t in ['III','IV','V','VI','VII','VIII','IX']}
    ci_path=(ci_dir/'confidence_intervals.json').relative_to(PACKAGE).as_posix()
    ci=load(ci_dir/'confidence_intervals.json')['cells']
    original=load(PACKAGE/'manifests/submitted_cells.json')['cells']
    mapped=[]
    for c in original:
        try:mapped.append(map_one(c,accepted,ci,ci_path))
        except (KeyError,ValueError) as e:
            mapped.append({**c,'evidence_kind':'MISSING_DATA_OR_MAPPING','data_status':'BLOCKED',
                           'match':False,'blocker':str(e),'full_rerun_verified':False})
    for r in mapped:
        if r.get('source_file','').startswith('accepted_phase1/'):
            r['source_file']=(actual_dir/f"table_{r['table']}.json").relative_to(PACKAGE).as_posix()
    mismatches=[r['cell_id'] for r in mapped if not r['match']]
    return {'status':'COMPLETE_WITH_DIFFERENCES' if mismatches else 'ALL_MAPPED_CELLS_MATCH_SUBMITTED_DISPLAY',
            'cell_count':len(mapped),'numeric_cells':sum(bool(r.get('components')) for r in mapped),
            'by_table':dict(Counter(r['table'] for r in mapped)),
            'by_evidence_kind':dict(Counter(r['evidence_kind'] for r in mapped)),
            'differences_or_blockers':mismatches,'cells':mapped,
            'limits':'Body cells only: method row keys/headers are locators; table notes, surrounding prose and Appendix are protocol crossreferences, not newly audited experiments.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--ci',required=True);ap.add_argument('--actual',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    ci=safe_path(args.ci)
    validation=load(ci/'ci_validation.json')
    if validation['status']!='PASS':raise ValueError('CI validation must pass before mapping')
    result=build(ci,safe_path(args.actual));out=output_dir(args.output);dump(out/'cell_evidence_map.json',result)
    flattened=[]
    for r in result['cells']:
        flat=dict(r)
        for key,value in flat.items():
            if isinstance(value,(list,dict)):
                import json
                flat[key]=json.dumps(value,ensure_ascii=False)
        flattened.append(flat)
    columns=list(dict.fromkeys(k for r in flattened for k in r))
    flattened=[{k:r.get(k,'') for k in columns} for r in flattened]
    write_csv(out/'cell_evidence_map.csv',flattened)
    lines=['# Tables III–IX 逐 cell 证据映射','',result['status'],'',
           'Archived case-level results are reaggregated with the saved confidence-interval procedure; final manuscript references are checked separately.','',
           '|Cell|R2行/列|提交值|证据状态|显示值一致|','|---|---|---|---|---|']
    for r in result['cells']:
        lines.append(f"|{r['cell_id']}|{r['source_line']} / {r['column']}|{r['submitted_display']}|{r['evidence_kind']}|{r['match']}|")
    (out/'cell_evidence_map.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({k:v for k,v in result.items() if k!='cells'})
    raise SystemExit(1 if result['differences_or_blockers'] else 0)


if __name__=='__main__':main()
