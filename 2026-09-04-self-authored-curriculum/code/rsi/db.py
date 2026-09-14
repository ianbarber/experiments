"""SQLite log: single source of truth for tasks, episodes, turns, gates, iterations, events (docs/03 §9-10)."""
from __future__ import annotations
import json, sqlite3, time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY, name TEXT, path TEXT, source TEXT, domain TEXT, tier TEXT,
  iteration_added INTEGER, status TEXT, provenance TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS gates (
  id INTEGER PRIMARY KEY, task_id TEXT, iteration INTEGER, check_name TEXT, verdict TEXT, detail TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY, task_id TEXT, job TEXT, trial TEXT, policy TEXT, purpose TEXT, iteration INTEGER,
  reward REAL, verified INTEGER, turns INTEGER, stop_reason TEXT, compactions INTEGER,
  n_input_tokens INTEGER, n_output_tokens INTEGER, wall_s REAL, started_at TEXT, finished_at TEXT,
  exception TEXT, quarantined INTEGER DEFAULT 0, flags TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS turns (
  episode_id TEXT, segment INTEGER, turn INTEGER, prompt_tokens INTEGER, gen_tokens INTEGER,
  finish_reason TEXT, command TEXT, exit_code INTEGER, obs_chars INTEGER, latency_s REAL,
  record_path TEXT, PRIMARY KEY (episode_id, segment, turn));
CREATE TABLE IF NOT EXISTS iterations (
  k INTEGER, step TEXT, status TEXT, started_at REAL, finished_at REAL, notes TEXT, PRIMARY KEY (k, step));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, kind TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY, ts REAL, iteration INTEGER, step TEXT, name TEXT, value REAL, extra TEXT);
CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_id);
CREATE INDEX IF NOT EXISTS idx_episodes_job ON episodes(job);
CREATE INDEX IF NOT EXISTS idx_gates_task ON gates(task_id);
"""

class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback(); raise

    def event(self, kind: str, **payload):
        with self.tx() as c:
            c.execute("INSERT INTO events(ts, kind, payload) VALUES (?,?,?)", (time.time(), kind, json.dumps(payload, default=str)))

    def metric(self, name: str, value: float, iteration: int | None = None, step: str | None = None, **extra):
        with self.tx() as c:
            c.execute("INSERT INTO metrics(ts, iteration, step, name, value, extra) VALUES (?,?,?,?,?,?)",
                      (time.time(), iteration, step, name, float(value), json.dumps(extra, default=str) if extra else None))

    def upsert_task(self, task_id, name, path, source, domain=None, tier=None, iteration_added=None, status="candidate", provenance=None):
        with self.tx() as c:
            c.execute("""INSERT INTO tasks(task_id,name,path,source,domain,tier,iteration_added,status,provenance,created_at)
                         VALUES(?,?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(task_id) DO UPDATE SET name=COALESCE(excluded.name, tasks.name), path=COALESCE(excluded.path, tasks.path), source=COALESCE(excluded.source, tasks.source),
                           domain=COALESCE(excluded.domain, tasks.domain), tier=COALESCE(excluded.tier, tasks.tier),
                           status=excluded.status, provenance=COALESCE(excluded.provenance, tasks.provenance)""",
                      (task_id, name, str(path), source, domain, tier, iteration_added, status,
                       json.dumps(provenance) if isinstance(provenance, dict) else provenance, time.time()))

    def gate(self, task_id, check_name, verdict, detail=None, iteration=None):
        with self.tx() as c:
            c.execute("INSERT INTO gates(task_id,iteration,check_name,verdict,detail,ts) VALUES(?,?,?,?,?,?)",
                      (task_id, iteration, check_name, verdict, json.dumps(detail, default=str) if detail is not None else None, time.time()))

    def upsert_episode(self, **row):
        cols = ["episode_id","task_id","job","trial","policy","purpose","iteration","reward","verified","turns","stop_reason",
                "compactions","n_input_tokens","n_output_tokens","wall_s","started_at","finished_at","exception","quarantined","flags"]
        vals = [row.get(k) for k in cols]
        if vals[cols.index("quarantined")] is None: vals[cols.index("quarantined")] = 0     # INSERT OR REPLACE skips the column default
        if isinstance(vals[-1], (list, dict)): vals[-1] = json.dumps(vals[-1])
        with self.tx() as c:
            c.execute(f"INSERT OR REPLACE INTO episodes({','.join(cols)},ts) VALUES({','.join('?'*len(cols))},?)", (*vals, time.time()))

    def insert_turns(self, rows):
        with self.tx() as c:
            c.executemany("""INSERT OR REPLACE INTO turns(episode_id,segment,turn,prompt_tokens,gen_tokens,finish_reason,command,exit_code,obs_chars,latency_s,record_path)
                             VALUES(?,?,?,?,?,?,?,?,?,?,?)""", rows)

    def q(self, sql, *args):
        return self.conn.execute(sql, args).fetchall()

    def set_step(self, k: int, step: str, status: str, notes: str | None = None):
        with self.tx() as c:
            if status == "running":
                c.execute("INSERT OR REPLACE INTO iterations(k,step,status,started_at,finished_at,notes) VALUES(?,?,?,?,NULL,?)", (k, step, status, time.time(), notes))
            else:
                c.execute("""INSERT INTO iterations(k,step,status,started_at,finished_at,notes) VALUES(?,?,?,?,?,?)
                             ON CONFLICT(k,step) DO UPDATE SET status=excluded.status, finished_at=excluded.finished_at, notes=COALESCE(excluded.notes, iterations.notes)""",
                          (k, step, status, time.time(), time.time(), notes))
