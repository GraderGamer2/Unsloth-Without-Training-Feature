# SPDX-License-Identifier: AGPL-3.0-only
"""Project-scoped research migration regression, no model or network required."""
import importlib.util
import sqlite3
from pathlib import Path

spec = importlib.util.spec_from_file_location("research_project_scope", Path(__file__).resolve().parents[1] / "storage/research_project_scope.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
conn = sqlite3.connect(":memory:")
conn.executescript("""
PRAGMA foreign_keys=ON;
CREATE TABLE chat_threads(id TEXT PRIMARY KEY);
CREATE TABLE chat_messages(id TEXT PRIMARY KEY);
CREATE TABLE chat_projects(id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS research_runs(
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    user_message_id TEXT NOT NULL REFERENCES chat_messages(id),
    assistant_message_id TEXT REFERENCES chat_messages(id),
    status TEXT NOT NULL CHECK(status IN ('running','completed'))
);
CREATE INDEX run_status ON research_runs(status);
CREATE TABLE sources(id TEXT PRIMARY KEY,run_id TEXT REFERENCES research_runs(id) ON DELETE CASCADE);
INSERT INTO chat_threads VALUES('t');
INSERT INTO chat_messages VALUES('m');
INSERT INTO chat_projects VALUES('p');
INSERT INTO research_runs VALUES('old','t','m',NULL,'completed');
INSERT INTO sources VALUES('s','old');
""")
module.ensure_project_scope(conn)
module.ensure_project_scope(conn)
assert conn.execute("SELECT thread_id,user_message_id FROM research_runs WHERE id='old'").fetchone() == ("t","m")
assert conn.execute("SELECT * FROM sources").fetchall() == [("s","old")]
assert conn.execute("SELECT name FROM sqlite_master WHERE name='run_status'").fetchone()
conn.execute("INSERT INTO research_runs(id,project_id,status) VALUES (?,?,?)", ("new","p","running"))
try:
    conn.execute("INSERT INTO research_runs(id,status) VALUES (?,?)", ("bad","running"))
    raise AssertionError("missing scope accepted")
except sqlite3.IntegrityError:
    pass
conn.execute("INSERT INTO sources VALUES (?,?)", ("new-source","new"))
conn.execute("DELETE FROM chat_projects WHERE id='p'")
assert conn.execute("SELECT id FROM research_runs").fetchall() == [("old",)]
assert conn.execute("SELECT * FROM sources").fetchall() == [("s","old")]
assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)
print("PASS: migration preserves legacy records, ownership, indexes and foreign keys")
