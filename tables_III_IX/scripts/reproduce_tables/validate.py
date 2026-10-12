"""Compare all regenerated cells to archived expectations, not to new output."""
import argparse
from common import Archive, MODE, PACKAGE, TABLES, compare, dump, inside, load


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', choices=[*TABLES, 'ALL'], default='ALL')
    ap.add_argument('--actual', required=True)
    args = ap.parse_args()
    actual = inside(PACKAGE, args.actual)
    if not actual.is_relative_to(PACKAGE / 'reproduced'):
        raise ValueError('Actual must be under reproduced/')
    report_path = actual / ('validation_' + args.table + '.json')
    if report_path.exists():
        raise ValueError('Refusing to overwrite validation report')
    tables = TABLES if args.table == 'ALL' else [args.table]
    archive = Archive()
    reports = []
    for table in tables:
        contract = load(PACKAGE / f'manifests/table_{table}.json')
        for asset in contract['inputs']:
            archive.rows(asset)
        expected = load(inside(PACKAGE, contract['expected']))['result']
        got = load(actual / f'table_{table}.json')
        errors = compare(expected, got['result'])
        if got.get('mode') != MODE or got.get('full_rerun_verified') is not False:
            errors.append('Invalid reproduction claim')
        reports.append({'table': table, 'status': 'FAIL' if errors else 'PASS',
                        'errors': errors, 'comparison': 'archived cells, abs/rel tolerance 1e-12'})
    report = {'status': 'PASS' if all(r['status'] == 'PASS' for r in reports) else 'FAIL',
              'tables': reports, 'full_rerun_verified': False, 'input_hashes_checked': len(archive.used)}
    dump(report_path, report)
    print(report['status'] + ': frozen expectation comparison')
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
