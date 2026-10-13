"""Generate archived experiment statistics. Use --verify for optional saved-result comparisons."""
import argparse
import json
import platform
import re
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ('III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX')


def _execute(script, arguments):
    script_path = ROOT / script
    original_argv, original_path = sys.argv[:], sys.path[:]
    for module in ('common', 'recompute_ci', 'historical_bootstrap'):
        sys.modules.pop(module, None)
    sys.argv = [str(script_path), *arguments]
    sys.path.insert(0, str(script_path.parent))
    try:
        runpy.run_path(str(script_path), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(f'Calculation failed: {script}') from exc
    finally:
        sys.argv, sys.path[:] = original_argv, original_path


def _next_id():
    for index in range(1, 10000):
        value = f'run_{index:03d}'
        if not (ROOT / 'results' / value).exists() and not (ROOT / 'tables_III_IX/reproduced' / (value + '_archive')).exists():
            return value
    raise RuntimeError('Too many existing result directories')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', help='Optional output folder name; automatically numbered if omitted')
    parser.add_argument('--verify', action='store_true', help='Also compare statistics with saved expectations')
    args = parser.parse_args()
    run_id = args.run_id or _next_id()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        parser.error('--run-id must contain only letters, numbers, underscore or hyphen')

    relative = 'reproduced/' + run_id
    native = {
        'tables_III_IX': ROOT / 'tables_III_IX' / (relative + '_archive'),
        'confidence_intervals': ROOT / 'tables_III_IX' / (relative + '_ci'),
        'c10_stability': ROOT / 'c10_stability' / relative,
    }
    if args.verify:
        native['verification'] = ROOT / 'tables_III_IX' / (relative + '_map')
    output = ROOT / 'results' / run_id
    if output.exists() or any(folder.exists() for folder in native.values()):
        parser.error('Output already exists; select a different --run-id')

    _execute('tables_III_IX/scripts/reproduce_tables/aggregate.py',
             ['--table', 'ALL', '--output', relative + '_archive'])
    _execute('tables_III_IX/scripts/recompute_ci.py',
             ['--output', relative + '_ci'])
    _execute('c10_stability/aggregate.py', ['--output', relative])

    if args.verify:
        _execute('tables_III_IX/scripts/reproduce_tables/validate.py',
                 ['--table', 'ALL', '--actual', relative + '_archive'])
        _execute('tables_III_IX/scripts/validate_ci.py',
                 ['--actual', relative + '_ci'])
        _execute('tables_III_IX/scripts/map_cells.py',
                 ['--actual', relative + '_archive', '--ci', relative + '_ci', '--output', relative + '_map'])
        _execute('c10_stability/validate.py', ['--actual', relative])

    intervals = json.loads((native['confidence_intervals'] / 'confidence_intervals.json').read_text(encoding='utf-8'))
    stability = json.loads((native['c10_stability'] / 'summary.json').read_text(encoding='utf-8'))
    summary = {
        'source': 'archived case-level and scalar observations',
        'tables_generated': list(TABLES),
        'confidence_intervals_generated': len(intervals['cells']),
        'c10_summary_rows': sum(len(rows) for rows in stability.values()),
        'comparison_with_saved_paper_values': 'passed' if args.verify else 'not requested',
        'from_scratch_training_or_simulation': False,
        'python': platform.python_version(),
    }
    output.mkdir(parents=True)
    for name, folder in native.items():
        shutil.copytree(folder, output / name)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (output / 'README.txt').write_text(
        'CEDA-T archived-result calculations\n'
        'tables_III_IX/: recalculated Tables III-IX from archived case outputs\n'
        'confidence_intervals/: recalculated Tables VI/VIII intervals and case records\n'
        'c10_stability/: five-case stability summaries and tables\n'
        + ('verification/: optional cell-by-cell comparison\n' if args.verify else '')
        + 'No simulation, generator, neural training, or raw attribution is rerun.\n',
        encoding='utf-8',
    )
    # Temporary internal calculation folders are not part of the reviewer-facing outputs.
    for folder in native.values():
        shutil.rmtree(folder)
    print(f'\nGenerated {len(TABLES)} tables, {len(intervals["cells"])} intervals, '
          f'{summary["c10_summary_rows"]} C10 summary rows.')
    if args.verify:
        print('Optional comparison with saved expectations completed.')
    print('Results saved to:', output.relative_to(ROOT))
    print('These outputs are reaggregations of archived records, not new model runs.')


if __name__ == '__main__':
    main()
