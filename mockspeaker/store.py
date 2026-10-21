import json
import sqlite3
import time
import uuid
from pathlib import Path


class Store:
    """Small SQLite transactions; no transaction crosses an await point."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS members (
            scope TEXT, user_id TEXT, name TEXT NOT NULL, total INTEGER NOT NULL DEFAULT 0,
            generated_at_count INTEGER NOT NULL DEFAULT 0, prompt TEXT NOT NULL DEFAULT '',
            updated_at REAL, error TEXT NOT NULL DEFAULT '', retry_at REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(scope, user_id));
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, user_id TEXT NOT NULL,
            message_id TEXT, text TEXT NOT NULL,
            UNIQUE(scope, message_id));
        CREATE INDEX IF NOT EXISTS samples_member ON samples(scope, user_id, id);
        CREATE TABLE IF NOT EXISTS shared_profiles (
            profile_key TEXT NOT NULL, user_id TEXT NOT NULL, source_scopes TEXT NOT NULL,
            prompt TEXT NOT NULL DEFAULT '', generated_at_count INTEGER NOT NULL DEFAULT 0,
            updated_at REAL, error TEXT NOT NULL DEFAULT '', retry_at REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(profile_key, user_id));
        CREATE TABLE IF NOT EXISTS selections (
            scope TEXT, owner TEXT, target TEXT NOT NULL, PRIMARY KEY(scope, owner));
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, scope TEXT NOT NULL, owner TEXT NOT NULL, target TEXT NOT NULL,
            origin TEXT NOT NULL, cid TEXT NOT NULL, closed INTEGER NOT NULL DEFAULT 0,
            seed TEXT NOT NULL DEFAULT '', memory TEXT NOT NULL DEFAULT '',
            summarized_through INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS sessions_owner ON sessions(scope, owner, closed);
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
            user TEXT NOT NULL, assistant TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id);
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY, origin TEXT NOT NULL, cid TEXT NOT NULL, text TEXT NOT NULL);
        PRAGMA user_version=2;
        """)

    def close(self):
        self.db.close()

    def member(self, scope, user):
        row = self.db.execute("SELECT * FROM members WHERE scope=? AND user_id=?", (scope, user)).fetchone()
        return dict(row) if row else None

    def members(self, scope):
        return [
            dict(r)
            for r in self.db.execute("SELECT * FROM members WHERE scope=? ORDER BY total DESC, user_id", (scope,))
        ]

    def collect(self, scope, user, name, message_id, text, max_samples):
        with self.db:
            inserted = self.db.execute(
                "INSERT OR IGNORE INTO samples(scope,user_id,message_id,text) VALUES(?,?,?,?)",
                (scope, user, message_id or None, text),
            ).rowcount
            if not inserted:
                return False
            self.db.execute(
                """INSERT INTO members(scope,user_id,name,total) VALUES(?,?,?,1)
                ON CONFLICT(scope,user_id) DO UPDATE SET name=excluded.name,total=total+1""",
                (scope, user, name),
            )
            self.db.execute(
                """DELETE FROM samples WHERE scope=? AND user_id=? AND id NOT IN
                (SELECT id FROM samples WHERE scope=? AND user_id=? ORDER BY id DESC LIMIT ?)""",
                (scope, user, scope, user, max_samples),
            )
        return True

    def samples(self, scope, user):
        return [
            r[0]
            for r in self.db.execute("SELECT text FROM samples WHERE scope=? AND user_id=? ORDER BY id", (scope, user))
        ]

    def total_messages(self, scopes, user):
        if not scopes:
            return 0
        placeholders = ",".join("?" for _ in scopes)
        return self.db.execute(
            f"SELECT COALESCE(SUM(total),0) FROM members WHERE scope IN ({placeholders}) AND user_id=?",
            (*scopes, user),
        ).fetchone()[0]

    def shared_profile(self, key, user):
        row = self.db.execute("SELECT * FROM shared_profiles WHERE profile_key=? AND user_id=?", (key, user)).fetchone()
        return dict(row) if row else None

    def save_shared_profile(self, key, user, sources, prompt, count):
        with self.db:
            self.db.execute(
                """INSERT INTO shared_profiles(profile_key,user_id,source_scopes,prompt,generated_at_count,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(profile_key,user_id) DO UPDATE SET
                prompt=excluded.prompt,generated_at_count=excluded.generated_at_count,
                updated_at=excluded.updated_at,error='',retry_at=0""",
                (key, user, json.dumps(sources), prompt, count, time.time()),
            )

    def shared_profile_error(self, key, user, sources, message):
        with self.db:
            self.db.execute(
                """INSERT INTO shared_profiles(profile_key,user_id,source_scopes,error,retry_at) VALUES(?,?,?,?,?)
                ON CONFLICT(profile_key,user_id) DO UPDATE SET error=excluded.error,retry_at=excluded.retry_at""",
                (key, user, json.dumps(sources), message[:300], time.time() + 300),
            )

    def save_profile(self, scope, user, prompt, count):
        with self.db:
            self.db.execute(
                """UPDATE members SET prompt=?,generated_at_count=?,updated_at=?,error='',retry_at=0
                WHERE scope=? AND user_id=?""",
                (prompt, count, time.time(), scope, user),
            )

    def profile_error(self, scope, user, message):
        with self.db:
            self.db.execute(
                "UPDATE members SET error=?,retry_at=? WHERE scope=? AND user_id=?",
                (message[:300], time.time() + 300, scope, user),
            )

    def selected(self, scope, owner):
        row = self.db.execute("SELECT target FROM selections WHERE scope=? AND owner=?", (scope, owner)).fetchone()
        return row[0] if row else None

    def select(self, scope, owner, target):
        with self.db:
            self.db.execute("UPDATE sessions SET closed=1 WHERE scope=? AND owner=? AND closed=0", (scope, owner))
            if target:
                self.db.execute("INSERT OR REPLACE INTO selections VALUES(?,?,?)", (scope, owner, target))
            else:
                self.db.execute("DELETE FROM selections WHERE scope=? AND owner=?", (scope, owner))

    def session(self, sid):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(row) if row else None

    def active_session(self, scope, owner, target, origin, cid):
        row = self.db.execute(
            """SELECT * FROM sessions WHERE scope=? AND owner=? AND target=?
            AND origin=? AND cid=? AND closed=0 ORDER BY rowid DESC LIMIT 1""",
            (scope, owner, target, origin, cid),
        ).fetchone()
        return dict(row) if row else None

    def create_session(self, scope, owner, target, origin, cid, seed="", closed=False):
        sid = uuid.uuid4().hex
        with self.db:
            if not closed:
                self.db.execute("UPDATE sessions SET closed=1 WHERE scope=? AND owner=? AND closed=0", (scope, owner))
            self.db.execute(
                "INSERT INTO sessions(id,scope,owner,target,origin,cid,seed,closed) VALUES(?,?,?,?,?,?,?,?)",
                (sid, scope, owner, target, origin, cid, seed, int(closed)),
            )
        return self.session(sid)

    def add_turn(self, sid, user, assistant):
        with self.db:
            self.db.execute("INSERT INTO turns(session_id,user,assistant) VALUES(?,?,?)", (sid, user, assistant))

    def turns(self, sid, pending=False):
        after = self.session(sid)["summarized_through"] if pending else 0
        return [
            dict(r)
            for r in self.db.execute("SELECT * FROM turns WHERE session_id=? AND id>? ORDER BY id", (sid, after))
        ]

    def pending_sessions(self, origin=None, cid=None, scope=None, owner=None, closed_only=False):
        sql = """SELECT s.* FROM sessions s WHERE EXISTS
            (SELECT 1 FROM turns t WHERE t.session_id=s.id AND t.id>s.summarized_through)"""
        args = []
        if closed_only:
            sql += " AND s.closed=1"
        for key, value in (("origin", origin), ("cid", cid), ("scope", scope), ("owner", owner)):
            if value is not None:
                sql += f" AND s.{key}=?"
                args.append(value)
        return [dict(r) for r in self.db.execute(sql, args)]

    def compact(self, sid, through, memory, note, keep_turns):
        session = self.session(sid)
        with self.db:
            self.db.execute("UPDATE sessions SET memory=?, summarized_through=? WHERE id=?", (memory, through, sid))
            self.db.execute(
                "INSERT OR IGNORE INTO notes VALUES(?,?,?,?)",
                (f"{sid}:{through}", session["origin"], session["cid"], note),
            )
            self.db.execute(
                """DELETE FROM turns WHERE session_id=? AND id<=? AND id NOT IN
                (SELECT id FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?)""",
                (sid, through, sid, keep_turns),
            )

    def notes(self, origin, cid):
        return [
            dict(r)
            for r in self.db.execute("SELECT * FROM notes WHERE origin=? AND cid=? ORDER BY rowid", (origin, cid))
        ]

    def acknowledge(self, ids):
        with self.db:
            self.db.executemany("DELETE FROM notes WHERE id=?", [(i,) for i in ids])

    def forget(self, scope, user):
        """Erase samples, profile, selections, raw imitation turns and undelivered notes."""
        shared_keys = [
            row[0]
            for row in self.db.execute("SELECT profile_key,source_scopes FROM shared_profiles WHERE user_id=?", (user,))
            if scope in json.loads(row[1])
        ]
        with self.db:
            self.db.executemany(
                "DELETE FROM shared_profiles WHERE profile_key=? AND user_id=?", [(key, user) for key in shared_keys]
            )
            ids = [
                r[0]
                for r in self.db.execute(
                    "SELECT id FROM sessions WHERE scope=? AND (owner=? OR target=?)", (scope, user, user)
                )
            ]
            for sid in ids:
                self.db.execute("DELETE FROM notes WHERE id LIKE ?", (sid + ":%",))
                self.db.execute("DELETE FROM turns WHERE session_id=?", (sid,))
                self.db.execute("DELETE FROM sessions WHERE id=?", (sid,))
            self.db.execute("DELETE FROM selections WHERE scope=? AND (owner=? OR target=?)", (scope, user, user))
            self.db.execute("DELETE FROM samples WHERE scope=? AND user_id=?", (scope, user))
            self.db.execute("DELETE FROM members WHERE scope=? AND user_id=?", (scope, user))
        return shared_keys


def scope_key(platform, group):
    return json.dumps([platform, group], ensure_ascii=False, separators=(",", ":"))
