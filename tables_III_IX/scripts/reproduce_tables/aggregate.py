"""Tables III-IX from frozen numeric archives. Never imports experiment code."""
import argparse
import statistics as st
from collections import Counter, defaultdict

from common import Archive, MODE, PACKAGE, TABLES, dump, fresh_output, load


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summaries(data, group_keys, metrics, ddof=1):
    buckets = defaultdict(list)
    for r in data:
        buckets[tuple(r[k] for k in group_keys)].append(r)
    result = []
    for group, records in sorted(buckets.items()):
        for metric in metrics:
            values = [float(r[metric]) for r in records if r.get(metric) not in ('', None)]
            if values:
                result.append({**dict(zip(group_keys, group)), 'metric': metric, 'n': len(values),
                               'mean': st.mean(values), 'sd': (st.stdev(values) if ddof else st.pstdev(values))
                               if len(values) > 1 else None, 'ddof': ddof})
    return result


def node_branches(archive, scale):
    result = []
    branches = [('gnnexplainer_' + scale, 1)]
    if scale == 'medium':
        branches += [('gat_ablation', 1), ('graphxai_extra', 0), ('graphxai_extra2', 1)]
    for asset, ddof in branches:
        data = archive.rows(asset)
        data = [{**r, 'accuracy': r.get('accuracy', r.get('Accuracy'))}
                for r in data if r.get('status', 'ok') == 'ok']
        counts = Counter(r['method'] for r in data)
        require(all(n == 25 for n in counts.values()), 'Each node branch needs 25 valid cases')
        result += [{**r, 'source': asset, 'scale': scale}
                   for r in summaries(data, ['method'], ['accuracy', 'NDCG@5', 'Recall@5'], ddof)]
    if scale == 'medium':
        for asset, prefix in [('ig_on_ceda', 'IG'), ('gnnex_on_ceda', 'GNNEx')]:
            data = [{**r, 'host': prefix + '-on-CEDA matched host'} for r in archive.rows(asset)]
            require(len(data) == 25, 'Matched host requires 25 cases')
            result += [{**r, 'source': asset, 'scale': scale} for r in summaries(data, ['host'],
                       ['accuracy', 'CCI_NDCG@5', 'CCI_Recall@5', prefix + '_NDCG@5', prefix + '_Recall@5'])]
    return result


def e4_rows(archive, table):
    data = archive.rows('e4_percase')
    counts = Counter((r['scale'], r['method']) for r in data)
    require(len(counts) == 18 and set(counts.values()) == {25}, 'E4 requires 3 x 6 x 25 rows')


    cells = summaries(data, ['scale', 'method'], ['accuracy', 'NDCG@5', 'Recall@5'])
    cells = [{'scale': r['scale'], 'method': r['method'], 'metric': r['metric'],
              'n': r['n'], 'mean': r['mean'], 'sample_std': r['sd']}
             for r in cells if table == 'IX' or r['scale'] == 'medium']
    return cells


def aggregate(table, archive):
    rows = archive.rows
    if table == 'III':
        manifest = rows('iii_cases')
        data = rows('iii_percase')
        ids = {r['case_id'] for r in manifest}
        require(len(ids) == 25 and len(data) == 125, 'III: wrong counts')
        variants = {r['variant'] for r in data}
        require(len(variants) == 5 and all({r['case_id'] for r in data if r['variant'] == v} == ids
                                         for v in variants), 'III: case/variant coverage mismatch')
        return {'cells': summaries(data, ['variant'], ['accuracy', 'NDCG@5', 'final_orth'])}
    if table == 'IV':
        data = rows('iv_scm')
        require(Counter(r['defect_class'] for r in data) == {'structural': 80, 'tacit': 240},
                'IV: expected 80/240 class rows')
        metrics = ['full_ndcg5', 'without_physics_ndcg5', 'without_context_ndcg5']
        cells = summaries(data, ['defect_class'], metrics)
        return {'by_class': cells, 'two_class_equal_macro': {
            m: st.mean(r['mean'] for r in cells if r['metric'] == m) for m in metrics}, 'n': len(data)}
    if table in ('V', 'IX'):
        scales = ['medium'] if table == 'V' else ['small', 'medium', 'large']
        return {'audited_E4_aggregate_rows': e4_rows(archive, table),
                'other_raw_branches': [r for scale in scales for r in node_branches(archive, scale)]}
    if table == 'VI':
        scores, selection = rows('vi_scores'), rows('vi_selection')
        grouped = defaultdict(list)
        for r in scores:
            grouped[r['case_id']].append(r)
        require(len(grouped) == 30 and {r['case_id'] for r in selection} == set(grouped),
                'VI: wrong case coverage')
        selected = {r['case_id']: r for r in selection}
        ceda = []
        for case, rs in sorted(grouped.items()):
            rs.sort(key=lambda r: int(r['rank_from_final_score']))
            require([int(r['rank_from_final_score']) for r in rs] == list(range(1, 21)),
                    'VI: invalid 20-candidate rank permutation')
            require(sum(int(r['is_gt']) for r in rs) == 1, 'VI: expected one GT')
            require(rs == sorted(rs, key=lambda r: (-float(r['final_lopo_score']), r['candidate'])),
                    'VI: ranks inconsistent with final scores')
            require(rs[0]['candidate'] == selected[case]['selected'], 'VI: selection mismatch')
            rank = next(int(r['rank_from_final_score']) for r in rs if int(r['is_gt']))
            require(rank == int(selected[case]['gt_rank_from_final_score']), 'VI: GT rank mismatch')
            ceda.append({'case_id': case, 'top1': int(rank == 1), 'top3': int(rank <= 3), 'MRR': 1 / rank})
        other = []
        for asset, methods in [('vi_structure', {'DistMult-KGC-real', 'GES-real', 'NOTEARS-real', 'GOLEM-real'}),
                               ('vi_llm_metrics', {'LLM-as-auditor'}),
                               ('vi_llm_family_metrics', {'GraphRAG-style', 'LLM-KGC-style'})]:
            data = [r for r in rows(asset) if r['method'] in methods]
            require(Counter((r['method'], r['top_k']) for r in data) ==
                    {(m, k): 30 for m in methods for k in ('1', '3')}, 'VI: method/top-k coverage mismatch')
            for r in data:
                r['harm_from_utility'] = int(float(r['U_eqi']) < 0)
            for k, metric in [('1', 'exact_tuple'), ('3', 'gt_in_topk'), ('1', 'harm_from_utility')]:
                other += [{**r, 'top_k': k, 'source': asset} for r in summaries(
                    [r for r in data if r['top_k'] == k], ['method'], [metric])]
        return {'CEDA_per_case': ceda, 'CEDA_top1': sum(r['top1'] for r in ceda) / len(ceda),
                'CEDA_top3': sum(r['top3'] for r in ceda) / len(ceda),
                'other_methods': other, 'clean_selection_rows': len(selection)}
    if table == 'VII':
        data = [r for r in rows('vii_percase') if r['included_for_headline'].lower() == 'true']
        counts = Counter((r['method'], r['defect_type']) for r in data)
        require(all(n == {'omitted': 24, 'weakened': 25, 'spurious': 25}[defect]
                    for (_, defect), n in counts.items()), 'VII: wrong valid denominator')
        return {'cells': summaries(data, ['method', 'defect_type'], ['pair_p1'])}
    if table == 'VIII':
        data = rows('viii_utilities')
        require(Counter(r['regime'] for r in data) == {'dry': 10, 'rain': 10, 'storm': 10},
                'VIII: expected 10 cases per regime')
        require(Counter(r['method_utility_source'] for r in data) == {'downstream': 28, 'candidate': 2},
                'VIII: expected 28+2 provenance')
        require({r['gt_utility_source'] for r in data} == {'downstream'}, 'VIII: GT provenance mismatch')
        result = []
        for regime in ('dry', 'rain', 'storm', 'overall'):
            sub = [r for r in data if regime == 'overall' or r['regime'] == regime]
            m, g = [float(r['delta_eqi']) for r in sub], [float(r['delta_eqi_gt']) for r in sub]
            result.append({'regime': regime, 'n': len(sub), 'mean_method': st.mean(m), 'mean_GT': st.mean(g),
                           'improved': sum(v > 0 for v in m), 'worsened': sum(v < 0 for v in m),
                           'benefit_share': sum(m) / sum(g)})
        return {'cells': result}
    raise ValueError('Unknown table')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', choices=[*TABLES, 'ALL'], default='ALL')
    ap.add_argument('--output', required=True, help='Fresh relative directory under reproduced/')
    args = ap.parse_args()
    tables = TABLES if args.table == 'ALL' else [args.table]
    archive = Archive()

    results = {t: aggregate(t, archive) for t in tables}
    output = fresh_output(args.output)
    for table, result in results.items():
        contract = load(PACKAGE / f'manifests/table_{table}.json')
        dump(output / f'table_{table}.json', {'table': table, 'mode': MODE, 'result': result,
             'ci_status': contract['ci_status'], 'full_rerun_verified': False})
    dump(output / 'run.json', {'tables': list(tables), 'mode': MODE, 'input_hashes': archive.used,
                             'training': 0, 'fits': 0, 'simulations': 0, 'network_calls': 0})
    print('PASS archive aggregation: ' + ', '.join(tables))


if __name__ == '__main__':
    main()
