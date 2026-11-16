"""SQLite persistence for the pathology slide allocation service.

Design rules enforced here:
  * one short explicit transaction per state transition;
  * a plan is written together with the *full frozen membership* of both splits
    in a single transaction, so a failure can never leave a half-written list;
  * a delivery row (with its own frozen copy of the slide rows) exists *before*
    any network call, so a crash can never lose a restriction or double-send;
  * deliveries are append-only facts; a delivery id maps to exactly one frozen
    content checksum and exactly one terminal state;
  * every delivery records its *owning receiver* (the URL used at send time and
    the receiver's persistent instance id, captured before any content leaves);
    legacy rows without attribution keep their ban and are never re-sent.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

DB_PATH = os.environ.get("APP_DB_PATH", "/data/pathology.sqlite3")
MAX_SLIDES = 100

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slides (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL,
    repeat_group TEXT,
    malignancy TEXT NOT NULL CHECK (malignancy IN ('benign', 'malignant')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_slides_patient ON slides(patient_id);
CREATE INDEX IF NOT EXISTS idx_slides_group ON slides(repeat_group);

CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    req_benign INTEGER NOT NULL,
    req_malignant INTEGER NOT NULL,
    test_benign INTEGER NOT NULL,
    test_malignant INTEGER NOT NULL,
    test_count INTEGER NOT NULL,
    train_count INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    snapshot_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'valid' CHECK (status IN ('valid', 'suspended')),
    status_reason TEXT,
    checked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_slides (
    plan_id INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    slide_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'test')),
    patient_id TEXT NOT NULL,
    repeat_group TEXT,
    malignancy TEXT NOT NULL,
    PRIMARY KEY (plan_id, slide_id)
);
CREATE INDEX IF NOT EXISTS idx_plan_slides_slide ON plan_slides(slide_id);

CREATE TABLE IF NOT EXISTS deliveries (
    id TEXT PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES plans(id),
    split TEXT NOT NULL CHECK (split IN ('train', 'test')),
    checksum TEXT NOT NULL,
    slide_count INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'rejected')),
    receiver_status TEXT NOT NULL DEFAULT 'unknown',
    receiver_url TEXT,
    receiver_id TEXT,
    receipt_ref TEXT,
    receipt_json TEXT,
    created_at TEXT NOT NULL,
    last_checked_at TEXT,
    last_error TEXT,
    UNIQUE (plan_id, checksum, attempt)
);
CREATE INDEX IF NOT EXISTS idx_deliveries_plan ON deliveries(plan_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_status ON deliveries(status);

CREATE TABLE IF NOT EXISTS delivery_slides (
    delivery_id TEXT NOT NULL REFERENCES deliveries(id) ON DELETE CASCADE,
    slide_id TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    repeat_group TEXT,
    malignancy TEXT NOT NULL,
    PRIMARY KEY (delivery_id, slide_id)
);

CREATE TABLE IF NOT EXISTS receipts (
    receipt_ref TEXT PRIMARY KEY,
    delivery_id TEXT NOT NULL UNIQUE REFERENCES deliveries(id),
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


class CapacityError(Exception):
    pass


def _migrate(conn: sqlite3.Connection) -> None:
    """Additively upgrade older database files.

    Delivery-attribution columns were introduced after the first deployment;
    rows created before them keep NULL receiver_url / receiver_id and are
    treated as *unattributed* (their ban is kept, never lifted by a
    substitute receiver).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(deliveries)")}
    if "receiver_url" not in cols:
        conn.execute("ALTER TABLE deliveries ADD COLUMN receiver_url TEXT")
    if "receiver_id" not in cols:
        conn.execute("ALTER TABLE deliveries ADD COLUMN receiver_id TEXT")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_init_lock = threading.Lock()
_initialized: set[str] = set()


def init_db(path: str) -> None:
    if path in _initialized:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with _init_lock:
        if path in _initialized:
            return
        conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
            conn.executescript(SCHEMA)
            _migrate(conn)
            conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES ('receiver_url', ?)",
                (os.environ.get("RECEIVER_URL", "http://receiver:8000"),),
            )
        finally:
            conn.close()
        _initialized.add(path)


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or DB_PATH
    init_db(path)
    # One connection per request. FastAPI may enter the dependency in one
    # thread and run the sync endpoint body in another, so the connection must
    # be usable across threads; it is never shared between requests.
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    return conn


def reset_init_cache() -> None:
    _initialized.clear()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


@contextmanager
def read_snapshot(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        conn.execute("COMMIT")


def log_event(conn: sqlite3.Connection, kind: str, message: str) -> None:
    conn.execute(
        "INSERT INTO events(created_at, kind, message) VALUES (?,?,?)",
        (now(), kind, message),
    )


def get_settings(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings")}


def get_setting(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def all_slides(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT seq, slide_id, patient_id, repeat_group, malignancy, created_at "
        "FROM slides ORDER BY seq"
    ).fetchall()
    return [dict(r) for r in rows]


def count_slides(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM slides").fetchone()["n"]


def insert_slides(
    conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]
) -> tuple[int, list[dict[str, Any]], int]:
    """Append parsed rows. Caller must provide the transaction.

    Duplicates (by slide id, including ids repeated inside this batch) are
    skipped, never overwritten. Capacity is checked before the first INSERT,
    so an over-capacity batch writes nothing. Returns
    (inserted, skipped_duplicates, total_before).
    """
    rows = list(rows)
    before = count_slides(conn)
    existing = {r["slide_id"] for r in conn.execute("SELECT slide_id FROM slides")}
    seen: set[str] = set()
    fresh: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for r in rows:
        sid = r["slide_id"]
        if sid in existing or sid in seen:
            duplicates.append(r)
            continue
        seen.add(sid)
        fresh.append(r)
    if before + len(fresh) > MAX_SLIDES:
        raise CapacityError(
            f"库容量上限 {MAX_SLIDES} 条：现有 {before} 条，本批新增 {len(fresh)} 条，"
            f"将超出 {before + len(fresh) - MAX_SLIDES} 条；整批未写入。"
        )
    for r in fresh:
        conn.execute(
            "INSERT INTO slides(slide_id, patient_id, repeat_group, malignancy, created_at) "
            "VALUES (?,?,?,?,?)",
            (
                r["slide_id"],
                r["patient_id"],
                r.get("repeat_group") or None,
                r["malignancy"],
                now(),
            ),
        )
    return len(fresh), duplicates, before


# ------------------------------------------------------------------- plans


def insert_plan(
    conn: sqlite3.Connection,
    *,
    created_at: str,
    req_benign: int,
    req_malignant: int,
    test_benign: int,
    test_malignant: int,
    test_count: int,
    train_count: int,
    checksum: str,
    fingerprint: str,
    rows: list[dict[str, Any]],
) -> int:
    cursor = conn.execute(
        "INSERT INTO plans(created_at, req_benign, req_malignant, test_benign, "
        "test_malignant, test_count, train_count, checksum, snapshot_fingerprint, "
        "status, status_reason, checked_at) VALUES (?,?,?,?,?,?,?,?,?, 'valid', NULL, ?)",
        (
            created_at,
            req_benign,
            req_malignant,
            test_benign,
            test_malignant,
            test_count,
            train_count,
            checksum,
            fingerprint,
            now(),
        ),
    )
    plan_id = int(cursor.lastrowid)
    conn.executemany(
        "INSERT INTO plan_slides(plan_id, slide_id, split, patient_id, repeat_group, malignancy) "
        "VALUES (?,?,?,?,?,?)",
        [
            (
                plan_id,
                r["slide_id"],
                r["split"],
                r["patient_id"],
                r.get("repeat_group") or None,
                r["malignancy"],
            )
            for r in rows
        ],
    )
    return plan_id


def get_plan(conn: sqlite3.Connection, plan_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()
    return dict(row) if row else None


def list_plans(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM plans ORDER BY id DESC")]


def plan_membership(conn: sqlite3.Connection, plan_id: int) -> dict[str, str]:
    return {
        r["slide_id"]: r["split"]
        for r in conn.execute(
            "SELECT slide_id, split FROM plan_slides WHERE plan_id=?", (plan_id,)
        )
    }


def plan_slides(
    conn: sqlite3.Connection, plan_id: int, split: str | None = None
) -> list[dict[str, Any]]:
    sql = (
        "SELECT slide_id, split, patient_id, repeat_group, malignancy "
        "FROM plan_slides WHERE plan_id=?"
    )
    args: list[Any] = [plan_id]
    if split:
        sql += " AND split=?"
        args.append(split)
    sql += " ORDER BY slide_id"
    return [dict(r) for r in conn.execute(sql, args)]


def update_plan_status(
    conn: sqlite3.Connection, plan_id: int, status: str, reason: str | None
) -> None:
    conn.execute(
        "UPDATE plans SET status=?, status_reason=?, checked_at=? WHERE id=?",
        (status, reason, now(), plan_id),
    )


# -------------------------------------------------------------- deliveries


def insert_delivery(
    conn: sqlite3.Connection,
    *,
    delivery_id: str,
    plan_id: int,
    split: str,
    checksum: str,
    attempt: int,
    rows: list[dict[str, Any]],
    receiver_url: str | None = None,
    receiver_id: str | None = None,
    created_at: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO deliveries(id, plan_id, split, checksum, slide_count, attempt, "
        "status, receiver_status, receiver_url, receiver_id, created_at) "
        "VALUES (?,?,?,?,?,?,'pending','unknown',?,?,?)",
        (
            delivery_id,
            plan_id,
            split,
            checksum,
            len(rows),
            attempt,
            receiver_url,
            receiver_id,
            created_at or now(),
        ),
    )
    conn.executemany(
        "INSERT INTO delivery_slides(delivery_id, slide_id, patient_id, repeat_group, malignancy) "
        "VALUES (?,?,?,?,?)",
        [
            (
                delivery_id,
                r["slide_id"],
                r["patient_id"],
                r.get("repeat_group") or None,
                r["malignancy"],
            )
            for r in rows
        ],
    )


def get_delivery(conn: sqlite3.Connection, delivery_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    return dict(row) if row else None


def list_deliveries(
    conn: sqlite3.Connection, plan_id: int | None = None
) -> list[dict[str, Any]]:
    if plan_id is None:
        rows = conn.execute(
            "SELECT * FROM deliveries ORDER BY created_at DESC, id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM deliveries WHERE plan_id=? ORDER BY created_at DESC, id",
            (plan_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def find_delivery(
    conn: sqlite3.Connection, plan_id: int, checksum: str, attempt: int
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM deliveries WHERE plan_id=? AND checksum=? AND attempt=?",
        (plan_id, checksum, attempt),
    ).fetchone()
    return dict(row) if row else None


def delivery_rows(conn: sqlite3.Connection, delivery_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT slide_id, patient_id, repeat_group, malignancy "
        "FROM delivery_slides WHERE delivery_id=? ORDER BY slide_id",
        (delivery_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delivery_slide_ids(conn: sqlite3.Connection, delivery_id: str) -> set[str]:
    return {
        r["slide_id"]
        for r in conn.execute(
            "SELECT slide_id FROM delivery_slides WHERE delivery_id=?", (delivery_id,)
        )
    }


def set_delivery_receiver_id(
    conn: sqlite3.Connection, delivery_id: str, receiver_id: str
) -> None:
    """Bind the receiver instance id once; never overwrite an existing one."""
    conn.execute(
        "UPDATE deliveries SET receiver_id=? WHERE id=? AND receiver_id IS NULL",
        (receiver_id, delivery_id),
    )


def set_delivery_receiver_url(
    conn: sqlite3.Connection, delivery_id: str, receiver_url: str
) -> None:
    """Record a verified address migration of the *same* receiver identity."""
    conn.execute(
        "UPDATE deliveries SET receiver_url=? WHERE id=?",
        (receiver_url, delivery_id),
    )


def set_delivery_terminal(
    conn: sqlite3.Connection,
    delivery_id: str,
    *,
    status: str,
    receiver_status: str,
    receipt_ref: str | None,
    receipt_json: str | None,
) -> None:
    conn.execute(
        "UPDATE deliveries SET status=?, receiver_status=?, last_checked_at=?, "
        "last_error=NULL, receipt_ref=?, receipt_json=? WHERE id=?",
        (status, receiver_status, now(), receipt_ref, receipt_json, delivery_id),
    )


def touch_delivery(
    conn: sqlite3.Connection,
    delivery_id: str,
    *,
    receiver_status: str,
    error: str | None,
) -> None:
    conn.execute(
        "UPDATE deliveries SET receiver_status=?, last_checked_at=?, last_error=? WHERE id=?",
        (receiver_status, now(), error, delivery_id),
    )


def store_receipt(
    conn: sqlite3.Connection, receipt_ref: str, delivery_id: str, payload: str
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO receipts(receipt_ref, delivery_id, payload, created_at) "
        "VALUES (?,?,?,?)",
        (receipt_ref, delivery_id, payload, now()),
    )


def list_events(conn: sqlite3.Connection, limit: int = 30) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        )
    ]
