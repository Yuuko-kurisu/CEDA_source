"""Standard-library archive I/O, integrity checks and safe fresh output paths."""
import csv
import hashlib
import json
import math
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
TABLES = ('III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX')
MODE = 'ARCHIVE_REAGGREGATION_ONLY_NOT_TRAINING_OR_FULL_RERUN'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()




def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def inside(root, rel):
    rel = Path(rel)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Non-portable path in manifest')
    p = (root / rel).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Input escapes package')
    return p


def fresh_output(relative):

    p = inside(PACKAGE, relative)
    if not p.is_relative_to(PACKAGE / 'reproduced') or p == PACKAGE / 'reproduced':
        raise ValueError('Output must be a fresh child of package/reproduced')
    if p.exists():
        raise ValueError('Output exists; choose a new output directory')
    p.mkdir(parents=True)
    return p


class Archive:
    def __init__(self, package=PACKAGE):
        self.package = Path(package).resolve()
        self.inputs = load(self.package / 'manifests/inputs.json')['assets']
        self.used = {}

    def rows(self, asset):
        spec = self.inputs[asset]
        p = inside(self.package, spec['path'])
        if not p.is_file():
            raise ValueError('Missing archive asset: ' + asset)
        with p.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != spec['columns']:
                raise ValueError('Column mismatch: ' + asset)
            records = list(reader)
        if len(records) != spec['rows']:
            raise ValueError('Row count mismatch: ' + asset)
        keys = spec.get('unique_key', [])
        seen = set()
        for row in records:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed CSV: ' + asset)
            if keys:
                key = tuple(row[k] for k in keys)
                if key in seen:
                    raise ValueError('Duplicate record: ' + asset)
                seen.add(key)
            for col in spec.get('numeric', []):
                v = row[col]
                if v == '' and col in spec.get('nullable_numeric', []):
                    continue
                if not math.isfinite(float(v)):
                    raise ValueError('Nonfinite numeric value: ' + asset)
        self.used[asset] = sha(p)
        return records


def compare(expected, actual, path='', atol=1e-12):
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return [path + ': keys differ']
        for key in expected:
            errors += compare(expected[key], actual[key], path + '/' + key, atol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [path + ': list length differs']
        for i, (e, a) in enumerate(zip(expected, actual)):
            errors += compare(e, a, path + '/' + str(i), atol)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(expected, actual, abs_tol=atol, rel_tol=atol):
            errors.append(path + ': numeric mismatch')
    elif expected != actual:
        errors.append(path + ': value differs')
    return errors
