"""SQLite metadata store. Per D8: local filesystem for images/models, SQLite for metadata."""
import sqlite3, json, time
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS classes (name TEXT PRIMARY KEY, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT, class_name TEXT NOT NULL,
                filename TEXT NOT NULL, path TEXT NOT NULL, created_at REAL NOT NULL,
                FOREIGN KEY(class_name) REFERENCES classes(name));
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, started_at REAL NOT NULL, finished_at REAL,
                class_names TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'running');
            CREATE TABLE IF NOT EXISTS run_models (
                run_id TEXT NOT NULL, model_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', artifact_path TEXT, accuracy REAL,
                confusion_matrix TEXT, labels TEXT, error TEXT, finished_at REAL,
                PRIMARY KEY (run_id, model_type), FOREIGN KEY(run_id) REFERENCES runs(id));
        """)

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def add_class(name):
    with get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO classes(name, created_at) VALUES (?, ?)", (name, time.time()))

def list_classes():
    with get_conn() as conn:
        return [r["name"] for r in conn.execute("SELECT name FROM classes ORDER BY name").fetchall()]

def delete_class(name):
    with get_conn() as conn:
        conn.execute("DELETE FROM images WHERE class_name = ?", (name,))
        conn.execute("DELETE FROM classes WHERE name = ?", (name,))

def add_image(class_name, filename, path):
    with get_conn() as conn:
        conn.execute("INSERT INTO images(class_name, filename, path, created_at) VALUES (?, ?, ?, ?)",
                     (class_name, filename, path, time.time()))

def list_images(class_name=None):
    with get_conn() as conn:
        if class_name:
            rows = conn.execute("SELECT * FROM images WHERE class_name = ? ORDER BY id", (class_name,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM images ORDER BY class_name, id").fetchall()
        return [dict(r) for r in rows]

def class_counts():
    with get_conn() as conn:
        rows = conn.execute("SELECT class_name, COUNT(*) as n FROM images GROUP BY class_name").fetchall()
        return {r["class_name"]: r["n"] for r in rows}

def create_run(run_id, class_names):
    with get_conn() as conn:
        conn.execute("INSERT INTO runs(id, started_at, class_names, status) VALUES (?, ?, ?, 'running')",
                     (run_id, time.time(), json.dumps(class_names)))
        for model_type in ("logistic_regression", "random_forest", "cnn"):
            conn.execute("INSERT INTO run_models(run_id, model_type, status) VALUES (?, ?, 'pending')",
                         (run_id, model_type))

def update_run_model(run_id, model_type, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values())
    with get_conn() as conn:
        conn.execute(f"UPDATE run_models SET {cols} WHERE run_id = ? AND model_type = ?",
                     vals + [run_id, model_type])

def finish_run(run_id):
    with get_conn() as conn:
        conn.execute("UPDATE runs SET finished_at = ?, status = 'done' WHERE id = ?", (time.time(), run_id))

def get_run(run_id):
    with get_conn() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return None
        models = conn.execute("SELECT * FROM run_models WHERE run_id = ?", (run_id,)).fetchall()
        run = dict(run)
        run["models"] = [dict(m) for m in models]
        return run

def list_runs():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()]

def latest_done_run():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE status='done' ORDER BY started_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None

_init_db()
