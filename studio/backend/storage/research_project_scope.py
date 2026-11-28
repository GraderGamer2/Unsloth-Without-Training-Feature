# SPDX-License-Identifier: AGPL-3.0-only
"""Add project-owned runs to the existing durable research store, without dummy chats."""
import sqlite3


def ensure_project_scope(conn: sqlite3.Connection) -> None:
    if any(row[1] == "project_id" for row in conn.execute("PRAGMA table_info(research_runs)")):
        return
    conn.commit()
    foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        columns = [row[1] for row in conn.execute("PRAGMA table_info(research_runs)")]
        if "project_id" in columns:
            conn.commit()
            return
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='research_runs'").fetchone()[0]
        for old, new in (
            ("CREATE TABLE research_runs", "CREATE TABLE research_runs_new"),
            ("thread_id TEXT NOT NULL REFERENCES chat_threads", "thread_id TEXT REFERENCES chat_threads"),
            ("user_message_id TEXT NOT NULL REFERENCES chat_messages", "user_message_id TEXT REFERENCES chat_messages"),
            ("status TEXT NOT NULL CHECK", "project_id TEXT REFERENCES chat_projects(id) ON DELETE CASCADE,\n            status TEXT NOT NULL CHECK"),
        ):
            if ddl.count(old) != 1:
                raise RuntimeError("Unsupported research schema; existing data was not changed")
            ddl = ddl.replace(old, new)
        ddl = ddl.rstrip()
        if not ddl.endswith(")"):
            raise RuntimeError("Unexpected research schema")
        ddl = ddl[:-1] + """,
            CHECK ((project_id IS NULL AND thread_id IS NOT NULL AND user_message_id IS NOT NULL)
                OR (project_id IS NOT NULL AND thread_id IS NULL AND user_message_id IS NULL
                    AND assistant_message_id IS NULL))
        )"""
        objects = [row[0] for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE tbl_name='research_runs' AND type IN ('index','trigger') AND sql IS NOT NULL")]
        quoted = ", ".join('"' + col.replace('"', '""') + '"' for col in columns)
        conn.execute(ddl)
        conn.execute(f"INSERT INTO research_runs_new ({quoted}) SELECT {quoted} FROM research_runs")
        conn.execute("DROP TABLE research_runs")
        conn.execute("ALTER TABLE research_runs_new RENAME TO research_runs")
        for sql in objects:
            conn.execute(sql)
        # Check the rebuilt parent and every existing child reference before commit.
        children = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for name in children:
            ident = '"' + name.replace('"', '""') + '"'
            if name == "research_runs" or any(row[2] == "research_runs" for row in conn.execute(f"PRAGMA foreign_key_list({ident})")):
                if conn.execute(f"PRAGMA foreign_key_check({ident})").fetchone():
                    raise RuntimeError("Research migration would break existing foreign keys")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.execute(f"PRAGMA foreign_keys={int(foreign_keys)}")
