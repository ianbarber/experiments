"""Small CPU-only provenance helpers for the preliminary check."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs: {path}')
    return rows


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def write_rows_new(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def check_freeze():
    frozen = json.loads((ROOT / 'FREEZE.json').read_text())
    for name, digest in frozen['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'Frozen input changed: {name}')
    return frozen


def event(name, **details):
    record = {'at': now(), 'event': name, **details}
    (ROOT / 'results').mkdir(exist_ok=True)
    with (ROOT / 'results/events.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def notebook(text):
    with (ROOT / 'LABNOTES.md').open('a') as handle:
        handle.write(f'\n## {now()}\n\n{text}\n')
