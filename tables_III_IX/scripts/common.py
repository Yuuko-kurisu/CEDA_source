import csv
import hashlib
import json
from pathlib import Path

PACKAGE=Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()




def safe_path(relative):
    r=Path(relative)
    if r.is_absolute() or '..' in r.parts:raise ValueError('Relative package path required')
    p=(PACKAGE/r).resolve()
    if not p.is_relative_to(PACKAGE):raise ValueError('Path escapes package')
    return p


def check_inputs():
    checked=[]
    for rel in load(PACKAGE/'manifests/ci_inputs.json')['files']:
        path=safe_path(rel)
        if not path.is_file():raise ValueError('Missing input: '+rel)
        checked.append(rel)
    return checked


def rows(name):
    with safe_path('archived_results/'+name+'.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def output_dir(relative):
    p=safe_path(relative)
    if not p.is_relative_to(PACKAGE/'reproduced') or p==PACKAGE/'reproduced':raise ValueError('Output must be below reproduced/')
    if p.exists():raise ValueError('Refusing overwrite; use a fresh output directory')
    p.mkdir(parents=True)
    return p


def write_csv(path,records):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
