"""Delivery ownership / substitute-receiver tests.

Scenario under test (the reported bug):

    a pathologist delivers a training list to receiver A; A accepts and the
    receipt is lost.  The operator then switches the configured address to a
    *different* receiver B (or replaces the data volume behind the same
    address) and presses "retry".  Previously the old delivery was re-sent to
    B, which accepted it; the local delivery then flipped to ``accepted`` and
    the already-trained slides became eligible for the blind test again.

Required behaviour:

* an old delivery may only be confirmed / continued by its *original*
  receiver; a changed address or a fresh volume behind the same address must
  never receive it;
* the original receiver, after a normal restart (same volume), still works;
* verification results re-judge old plans immediately and a failure suspends
  them;
* concurrent retries / late receipts / app restarts never lose a training
  exposure restriction;
* legacy rows that cannot prove their receiver stay barred, with a reason;
  an already-accepted legacy row keeps its permanent ban; a legacy "rejected"
  row does not lift anything.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import csvio, receiver_client, service, storage  # noqa: E402
from app.main import app as app_app  # noqa: E402
import receiver.main as rm  # noqa: E402

SAMPLE = """slide_id,patient_id,repeat_group,malignancy
S1,P1,G1,benign
S2,P2,G1,malignant
S3,P3,,benign
S4,P4,,malignant
S5,P5,,benign
S6,P6,,malignant
S7,P7,,benign
S8,P8,,malignant
"""


class FakeReceiver:
    """An independent receiver volume with its own SQLite file and identity."""

    def __init__(self, monkeypatch, path: str, refuse: bool = False):
        self.path = path
        self._use()
        conn = rm.connect()
        conn.execute(
            "INSERT INTO controls(key,value) VALUES ('refuse',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("1" if refuse else "0",),
        )
        conn.execute(
            "INSERT INTO controls(key,value) VALUES ('fault','none') "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
        )
        conn.close()
        self.client = TestClient(rm.app)
        self.identity = self.client.get("/api/identity").json()["receiver_id"]

    def _use(self) -> None:
        rm.DB_PATH = self.path

    def set_refuse(self, refuse: bool) -> None:
        self._use()
        conn = rm.connect()
        conn.execute(
            "INSERT INTO controls(key,value) VALUES ('refuse',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("1" if refuse else "0",),
        )
        conn.close()

    def records(self):
        self._use()
        return self.client.get("/api/deliveries").json()


@pytest.fixture()
def env(monkeypatch):
    d = tempfile.mkdtemp(prefix="own-")
    app_db = os.path.join(d, "app.sqlite3")
    monkeypatch.setenv("APP_DB_PATH", app_db)
    storage.reset_init_cache()
    yield app_db
    storage.reset_init_cache()


def route(monkeypatch, url_to_receiver: dict[str, FakeReceiver]):
    """Route the client's httpx calls by URL into the right ASGI receiver."""

    def _receiver(receiver_url):
        return url_to_receiver.get(receiver_url.rstrip("/"))

    def fetch_identity(receiver_url, timeout=10.0):
        recv = _receiver(receiver_url)
        return recv.identity if recv else None

    def send(
        receiver_url,
        delivery_id,
        checksum,
        slides,
        timeout=10.0,
        expected_receiver_id=None,
    ):
        recv = _receiver(receiver_url)
        if recv is None:
            return receiver_client.DeliveryOutcome(
                state="unknown", error="连接被拒（无此接收方）"
            )
        payload = {
            "delivery_id": delivery_id,
            "checksum": checksum,
            "slides": slides,
        }
        if expected_receiver_id:
            payload["expected_receiver_id"] = expected_receiver_id
        recv._use()
        resp = recv.client.post("/api/deliveries", json=payload)
        if resp.status_code not in (200, 201):
            return receiver_client.DeliveryOutcome(
                state="unknown",
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                http_status=resp.status_code,
            )
        return receiver_client._normalise(resp.json(), checksum)

    def query(
        receiver_url,
        delivery_id,
        expected_checksum=None,
        timeout=10.0,
        expected_receiver_id=None,
    ):
        recv = _receiver(receiver_url)
        if recv is None:
            return receiver_client.DeliveryOutcome(
                state="unknown", error="连接被拒（无此接收方）"
            )
        recv._use()
        resp = recv.client.get(f"/api/deliveries/{delivery_id}")
        if resp.status_code == 404:
            return receiver_client.DeliveryOutcome(
                state="unknown", error="查无记录", http_status=404
            )
        if resp.status_code != 200:
            return receiver_client.DeliveryOutcome(
                state="unknown", error=f"HTTP {resp.status_code}"
            )
        data = resp.json()
        if expected_receiver_id and data.get("receiver_id") != expected_receiver_id:
            return receiver_client.DeliveryOutcome(
                state="unknown", error="接收方身份不一致", payload=data
            )
        return receiver_client._normalise(data, expected_checksum or "")

    for module in (receiver_client, service.receiver_client):
        monkeypatch.setattr(module, "fetch_identity", fetch_identity)
        monkeypatch.setattr(module, "send_delivery", send)
        monkeypatch.setattr(module, "query_delivery", query)


def seed(app_db) -> tuple[sqlite3.Connection, int]:
    conn = storage.connect(app_db)
    rows = csvio.parse_csv_text(SAMPLE)
    service.append_slides(conn, [r.__dict__ for r in rows])
    plan_id = service.publish_plan(conn, 2, 2)["plan_id"]
    return conn, plan_id


def train_ids(conn, plan_id):
    return {r["slide_id"] for r in storage.plan_slides(conn, plan_id, "train")}


# ------------------------------------------------------------ core scenario


def test_lost_receipt_then_address_switch_never_reaches_substitute(env, monkeypatch):
    """The exact reported bug: A accepted, receipt lost, operator switches to B."""
    d = tempfile.mkdtemp(prefix="recv-")
    a = FakeReceiver(monkeypatch, os.path.join(d, "a.sqlite3"))
    b = FakeReceiver(monkeypatch, os.path.join(d, "b.sqlite3"))
    url_a, url_b = "http://127.0.0.1:8001", "http://127.0.0.1:8002"
    route(monkeypatch, {url_a: a, url_b: b})

    conn, plan_id = seed(env)
    # Simulate a lost response: A stores the list (accepted) but the sender
    # sees an unknown result, leaving the local delivery pending.
    real_send = service.receiver_client.send_delivery

    def lost_response(receiver_url, delivery_id, checksum, slides, timeout=10.0,
                      expected_receiver_id=None):
        real_send(receiver_url, delivery_id, checksum, slides, timeout=timeout,
                  expected_receiver_id=expected_receiver_id)
        return receiver_client.DeliveryOutcome(state="unknown", error="响应丢失")

    monkeypatch.setattr(service.receiver_client, "send_delivery", lost_response)
    res = service.start_delivery(conn, plan_id, url_a)
    pid = res["delivery"]["id"]
    assert res["delivery"]["status"] == "pending"
    assert res["delivery"]["receiver_id"] == a.identity
    assert len(a.records()) == 1

    # Operator switches the configured address and presses "retry". The retry
    # must go to the *original* receiver A, never to B.
    monkeypatch.setattr(service.receiver_client, "send_delivery", real_send)
    out = service.retry_delivery(conn, pid, url_b, timeout=2)
    assert out["delivery"]["status"] == "accepted"  # confirmed by A
    assert b.records() == []  # B never received anything
    assert len(a.records()) == 1  # idempotent replay, not a second record
    assert out["delivery"]["receiver_id"] == a.identity
    slide_ids = storage.delivery_slide_ids(conn, pid)
    assert slide_ids <= service.current_restrictions(conn).accepted
    conn.close()


def test_verify_uses_recorded_receiver_not_configured_address(env, monkeypatch):
    """A bound delivery is always verified at its recorded receiver.

    Even if the substitute receiver B holds a look-alike record under the same
    delivery id, confirming against B is never attempted; the recorded
    receiver A answers and B's record is ignored.
    """
    d = tempfile.mkdtemp(prefix="recv-")
    url_a, url_b = "http://127.0.0.1:8001", "http://127.0.0.1:8002"
    a = FakeReceiver(monkeypatch, os.path.join(d, "a.sqlite3"))
    b = FakeReceiver(monkeypatch, os.path.join(d, "b.sqlite3"))
    route(monkeypatch, {url_a: a, url_b: b})
    conn, plan_id = seed(env)

    real_send = service.receiver_client.send_delivery

    def lost(receiver_url, delivery_id, checksum, slides, timeout=10.0,
             expected_receiver_id=None):
        real_send(receiver_url, delivery_id, checksum, slides)
        return receiver_client.DeliveryOutcome(state="unknown", error="响应丢失")

    monkeypatch.setattr(service.receiver_client, "send_delivery", lost)
    pid = service.start_delivery(conn, plan_id, url_a)["delivery"]["id"]
    rows = [
        {**r, "split": "train"}
        for r in storage.delivery_rows(conn, pid)
    ]
    checksum = storage.get_delivery(conn, pid)["checksum"]

    # B happens to hold a record under the same delivery id (e.g. it was
    # cloned/replaced with stale data). It must never be consulted.
    b._use()
    b.client.post(
        "/api/deliveries",
        json={"delivery_id": pid, "checksum": checksum, "slides": rows},
    )
    assert len(b.records()) == 1

    seen_urls: list[str] = []
    orig_fetch = service.receiver_client.fetch_identity
    orig_query = service.receiver_client.query_delivery

    def spy_fetch(receiver_url, timeout=10.0):
        seen_urls.append(receiver_url.rstrip("/"))
        return orig_fetch(receiver_url, timeout=timeout)

    def spy_query(receiver_url, delivery_id, expected_checksum=None, timeout=10.0,
                  expected_receiver_id=None):
        seen_urls.append(receiver_url.rstrip("/"))
        return orig_query(
            receiver_url, delivery_id, expected_checksum, timeout,
            expected_receiver_id,
        )

    monkeypatch.setattr(service.receiver_client, "fetch_identity", spy_fetch)
    monkeypatch.setattr(service.receiver_client, "query_delivery", spy_query)

    out = service.verify_delivery(conn, pid, url_b, timeout=2)
    assert out["delivery"]["status"] == "accepted"
    assert out["delivery"]["receiver_id"] == a.identity
    assert set(seen_urls) == {url_a}  # B was never contacted
    conn.close()


def test_same_address_new_volume_is_not_the_original_receiver(env, monkeypatch):
    """A fresh data volume mounted at the same address still counts as other."""
    d = tempfile.mkdtemp(prefix="recv-")
    url = "http://127.0.0.1:8001"
    first = FakeReceiver(monkeypatch, os.path.join(d, "vol1.sqlite3"))
    route(monkeypatch, {url: first})
    conn, plan_id = seed(env)

    # Receiver accepts, but the sender only sees a lost response.
    real_send = service.receiver_client.send_delivery

    def lost(receiver_url, delivery_id, checksum, slides, timeout=10.0,
             expected_receiver_id=None):
        real_send(receiver_url, delivery_id, checksum, slides)
        return receiver_client.DeliveryOutcome(state="unknown", error="响应丢失")

    monkeypatch.setattr(service.receiver_client, "send_delivery", lost)
    res = service.start_delivery(conn, plan_id, url)
    pid = res["delivery"]["id"]
    assert res["delivery"]["receiver_id"] == first.identity

    # Replace the volume behind the same address: new identity, no record.
    second = FakeReceiver(monkeypatch, os.path.join(d, "vol2.sqlite3"))
    route(monkeypatch, {url: second})
    assert second.identity != first.identity

    monkeypatch.setattr(service.receiver_client, "send_delivery", real_send)
    out = service.retry_delivery(conn, pid, url, timeout=2)
    assert out["delivery"]["status"] == "pending"
    assert second.records() == []
    out = service.verify_delivery(conn, pid, url, timeout=2)
    assert out["delivery"]["status"] == "pending"
    assert second.records() == []

    # Restoring the original volume lets verification succeed.
    route(monkeypatch, {url: first})
    out = service.verify_delivery(conn, pid, url, timeout=2)
    assert out["delivery"]["status"] == "accepted"
    assert slide_ids_of(conn, pid) <= service.current_restrictions(conn).accepted
    conn.close()


def slide_ids_of(conn, pid):
    return storage.delivery_slide_ids(conn, pid)


def test_original_receiver_restart_same_volume_still_verifies(env, monkeypatch):
    """A normal restart of the original receiver (same data volume) must work."""
    d = tempfile.mkdtemp(prefix="recv-")
    path = os.path.join(d, "r.sqlite3")
    url = "http://127.0.0.1:8001"
    first = FakeReceiver(monkeypatch, path)
    route(monkeypatch, {url: first})
    conn, plan_id = seed(env)

    real_send = service.receiver_client.send_delivery

    def lost(receiver_url, delivery_id, checksum, slides, timeout=10.0,
             expected_receiver_id=None):
        real_send(receiver_url, delivery_id, checksum, slides)
        return receiver_client.DeliveryOutcome(state="unknown", error="响应丢失")

    monkeypatch.setattr(service.receiver_client, "send_delivery", lost)
    pid = service.start_delivery(conn, plan_id, url)["delivery"]["id"]
    conn.close()

    # "Restart" the app and the receiver against the same volume.
    restarted = FakeReceiver(monkeypatch, path)
    assert restarted.identity == first.identity
    route(monkeypatch, {url: restarted})
    conn2 = storage.connect(env)
    service.reconcile_pending(conn2, url, timeout=2)
    delivery = storage.get_delivery(conn2, pid)
    assert delivery["status"] == "accepted"
    assert storage.delivery_slide_ids(conn2, pid) <= service.current_restrictions(conn2).accepted
    conn2.close()


def test_verification_rejudges_plans_immediately(env, monkeypatch):
    """A verify result must re-judge old plans at once, not only on restart.

    * bound rejection  -> only that restriction lifts, plan becomes valid;
    * bound acceptance -> slides are barred, plan stays suspended.
    """
    d = tempfile.mkdtemp(prefix="recv-")
    a = FakeReceiver(monkeypatch, os.path.join(d, "a.sqlite3"))
    url = "http://127.0.0.1:8001"
    route(monkeypatch, {url: a})
    conn, plan1 = seed(env)
    m1 = storage.plan_membership(conn, plan1)
    test_slide = next(s for s, split in m1.items() if split == "test")
    slide = next(s for s in storage.all_slides(conn) if s["slide_id"] == test_slide)
    row = {
        "slide_id": slide["slide_id"],
        "patient_id": slide["patient_id"],
        "repeat_group": slide.get("repeat_group") or None,
        "malignancy": slide["malignancy"],
    }
    checksum = service.allocator.canonical_checksum(
        [{**row, "split": "train"}]
    )
    with storage.transaction(conn):
        storage.insert_delivery(
            conn,
            delivery_id="DLV-LATE",
            plan_id=plan1,
            split="train",
            checksum=checksum,
            attempt=1,
            receiver_id=a.identity,
            receiver_url=url,
            binding="bound",
            rows=[row],
        )
        service.refresh_plans(conn)
    assert storage.get_plan(conn, plan1)["status"] == "suspended"

    # Receiver accepted it while the app crashed before recording the answer.
    a._use()
    a.client.post(
        "/api/deliveries",
        json={
            "delivery_id": "DLV-LATE",
            "checksum": checksum,
            "slides": [{**row, "repeat_group": row["repeat_group"] or "", "split": "train"}],
        },
    )
    out = service.verify_delivery(conn, "DLV-LATE", url, timeout=2)
    assert out["delivery"]["status"] == "accepted"
    plan = storage.get_plan(conn, plan1)
    assert plan["status"] == "suspended"  # acceptance keeps it out of test
    with pytest.raises(service.PlanNotDeliverable):
        service.plan_download(conn, plan1, "test")
    assert test_slide in service.current_restrictions(conn).accepted
    conn.close()


def test_bound_rejection_restores_plan(env, monkeypatch):
    d = tempfile.mkdtemp(prefix="recv-")
    a = FakeReceiver(monkeypatch, os.path.join(d, "a.sqlite3"), refuse=True)
    url = "http://127.0.0.1:8001"
    route(monkeypatch, {url: a})
    conn, plan1 = seed(env)
    m1 = storage.plan_membership(conn, plan1)
    test_slide = next(s for s, split in m1.items() if split == "test")
    slide = next(s for s in storage.all_slides(conn) if s["slide_id"] == test_slide)
    row = {
        "slide_id": slide["slide_id"],
        "patient_id": slide["patient_id"],
        "repeat_group": slide.get("repeat_group") or None,
        "malignancy": slide["malignancy"],
    }
    checksum = service.allocator.canonical_checksum([{**row, "split": "train"}])
    with storage.transaction(conn):
        storage.insert_delivery(
            conn,
            delivery_id="DLV-REJ",
            plan_id=plan1,
            split="train",
            checksum=checksum,
            attempt=1,
            receiver_id=a.identity,
            receiver_url=url,
            binding="bound",
            rows=[row],
        )
        service.refresh_plans(conn)
    assert storage.get_plan(conn, plan1)["status"] == "suspended"

    # The receiver stored a *rejection* while the sender's reply was lost.
    a._use()
    posted = a.client.post(
        "/api/deliveries",
        json={
            "delivery_id": "DLV-REJ",
            "checksum": checksum,
            "slides": [{**row, "repeat_group": row["repeat_group"] or "", "split": "train"}],
        },
    )
    assert posted.status_code == 201 and posted.json()["status"] == "rejected"

    out = service.verify_delivery(conn, "DLV-REJ", url, timeout=2)
    assert out["delivery"]["status"] == "rejected"
    assert storage.get_plan(conn, plan1)["status"] == "valid"
    service.plan_download(conn, plan1, "test")
    assert test_slide not in service.current_restrictions(conn).barred
    conn.close()



# ------------------------------------------------------------- legacy rows


LEGACY_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE slides (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL,
    repeat_group TEXT,
    malignancy TEXT NOT NULL CHECK (malignancy IN ('benign', 'malignant')),
    created_at TEXT NOT NULL
);
CREATE TABLE plans (
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
    status TEXT NOT NULL DEFAULT 'valid',
    status_reason TEXT,
    checked_at TEXT NOT NULL
);
CREATE TABLE plan_slides (
    plan_id INTEGER NOT NULL,
    slide_id TEXT NOT NULL,
    split TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    repeat_group TEXT,
    malignancy TEXT NOT NULL,
    PRIMARY KEY (plan_id, slide_id)
);
CREATE TABLE deliveries (
    id TEXT PRIMARY KEY,
    plan_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    checksum TEXT NOT NULL,
    slide_count INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    receiver_status TEXT NOT NULL DEFAULT 'unknown',
    receipt_ref TEXT,
    receipt_json TEXT,
    created_at TEXT NOT NULL,
    last_checked_at TEXT,
    last_error TEXT,
    UNIQUE (plan_id, checksum, attempt)
);
CREATE TABLE delivery_slides (
    delivery_id TEXT NOT NULL,
    slide_id TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    repeat_group TEXT,
    malignancy TEXT NOT NULL,
    PRIMARY KEY (delivery_id, slide_id)
);
CREATE TABLE receipts (
    receipt_ref TEXT PRIMARY KEY,
    delivery_id TEXT NOT NULL UNIQUE,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


def build_legacy_db(path: str, deliveries: list[tuple[str, str, list[str]]]) -> None:
    """Write an old-format DB (no receiver_id / binding columns)."""
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO settings(key,value) VALUES ('receiver_url','http://old-receiver')"
    )
    for i, sid in enumerate(sorted({s for _, _, ids in deliveries for s in ids})):
        conn.execute(
            "INSERT INTO slides(slide_id,patient_id,repeat_group,malignancy,created_at) "
            "VALUES (?,?,?,?,?)",
            (sid, f"P{sid}", None, "benign" if i % 2 == 0 else "malignant", now),
        )
    conn.execute(
        "INSERT INTO plans(id,created_at,req_benign,req_malignant,test_benign,"
        "test_malignant,test_count,train_count,checksum,snapshot_fingerprint,status,"
        "status_reason,checked_at) VALUES (1,?,0,0,0,0,0,2,'c','f','valid',NULL,?)",
        (now, now),
    )
    for i, (did, status, ids) in enumerate(deliveries):
        conn.execute(
            "INSERT INTO deliveries(id,plan_id,split,checksum,slide_count,attempt,"
            "status,receiver_status,receipt_ref,created_at) "
            "VALUES (?,1,'train',?,?,1,?,?,?,?)",
            (did, f"sum{i}", len(ids), status, status, f"RCPT-{i}", now),
        )
        for sid in ids:
            row = conn.execute(
                "SELECT slide_id,patient_id,repeat_group,malignancy FROM slides "
                "WHERE slide_id=?",
                (sid,),
            ).fetchone()
            conn.execute(
                "INSERT INTO delivery_slides(delivery_id,slide_id,patient_id,"
                "repeat_group,malignancy) VALUES (?,?,?,?,?)",
                (did, *row),
            )
        conn.execute(
            "INSERT INTO plan_slides(plan_id,slide_id,split,patient_id,repeat_group,"
            "malignancy) VALUES (1,?, 'train', (SELECT patient_id FROM slides "
            "WHERE slide_id=?), NULL, (SELECT malignancy FROM slides WHERE slide_id=?))",
            (ids[0], ids[0], ids[0]),
        )
    conn.commit()
    conn.close()


def test_legacy_rows_keep_restriction_with_reason(env):
    """Unknown / rejected legacy rows keep the ban; accepted stays permanent."""
    build_legacy_db(
        env,
        [
            ("DLV-LEG-ACC", "accepted", ["S1"]),
            ("DLV-LEG-REJ", "rejected", ["S2"]),
            ("DLV-LEG-PEND", "pending", ["S3"]),
        ],
    )
    conn = storage.connect(env)
    # Columns were added by migration; old rows are marked unproven.
    rows = {d["id"]: d for d in storage.list_deliveries(conn)}
    assert rows["DLV-LEG-ACC"]["binding"] == "legacy_unproven"
    assert rows["DLV-LEG-REJ"]["binding"] == "legacy_unproven"
    assert rows["DLV-LEG-ACC"]["status"] == "accepted"  # not downgraded

    state = service.current_restrictions(conn)
    assert "S1" in state.accepted and "S1" in state.barred
    # Rejected legacy row cannot lift its training exposure.
    assert "S2" in state.barred
    assert "S2" in state.unproven
    assert "原接收方" in state.explain("S2")
    assert "S3" in state.barred  # unconfirmed stays uncertain/unproven

    # Legacy rows are never networked. A terminal one is simply reported as
    # terminal (no address is contacted); a pending one is refused outright.
    out = service.verify_delivery(conn, "DLV-LEG-REJ", "http://old-receiver")
    assert out["reused"] is True and out["outcome"] is None
    out = service.retry_delivery(conn, "DLV-LEG-REJ", "http://old-receiver")
    assert out["reused"] is True and out["outcome"] is None
    with pytest.raises(service.ReceiverOwnershipError):
        service.verify_delivery(conn, "DLV-LEG-PEND", "http://old-receiver")
    with pytest.raises(service.ReceiverOwnershipError):
        service.retry_delivery(conn, "DLV-LEG-PEND", "http://old-receiver")
    # The unproven "rejected" row is not silently turned into an acceptance
    # or dropped: its bar remains (checked above) and its status is unchanged.
    assert storage.get_delivery(conn, "DLV-LEG-REJ")["status"] == "rejected"
    assert storage.get_delivery(conn, "DLV-LEG-PEND")["status"] == "pending"
    conn.close()


def test_legacy_plan_keeps_serving_when_only_accepted_train_rows_exist(env):
    build_legacy_db(env, [("DLV-A", "accepted", ["S1"])])
    conn = storage.connect(env)
    with storage.transaction(conn):
        service.refresh_plans(conn)
    # S1 is in the train split of the frozen plan; accepted train rows do not
    # invalidate it.
    assert storage.get_plan(conn, 1)["status"] == "valid"
    conn.close()


# ------------------------------------------------- concurrency / late reply


def test_concurrent_retries_persist_single_terminal_fact(env, monkeypatch):
    """Two threads verify/retry at once: one terminal fact, ban never lost."""
    d = tempfile.mkdtemp(prefix="recv-")
    a = FakeReceiver(monkeypatch, os.path.join(d, "a.sqlite3"))
    url = "http://127.0.0.1:8001"
    route(monkeypatch, {url: a})
    conn, plan_id = seed(env)
    real_send = service.receiver_client.send_delivery

    def lost(receiver_url, delivery_id, checksum, slides, timeout=10.0,
             expected_receiver_id=None):
        real_send(receiver_url, delivery_id, checksum, slides)
        return receiver_client.DeliveryOutcome(state="unknown", error="响应丢失")

    monkeypatch.setattr(service.receiver_client, "send_delivery", lost)
    pid = service.start_delivery(conn, plan_id, url)["delivery"]["id"]
    monkeypatch.setattr(service.receiver_client, "send_delivery", real_send)
    conn.close()

    errors: list[Exception] = []

    def worker(kind: str) -> None:
        c = storage.connect(env)
        try:
            if kind == "verify":
                service.verify_delivery(c, pid, url, timeout=3)
            else:
                service.retry_delivery(c, pid, url, timeout=3)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            c.close()

    threads = [
        threading.Thread(target=worker, args=("verify",)),
        threading.Thread(target=worker, args=("retry",)),
        threading.Thread(target=worker, args=("verify",)),
        threading.Thread(target=worker, args=("retry",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    conn2 = storage.connect(env)
    delivery = storage.get_delivery(conn2, pid)
    assert delivery["status"] == "accepted"
    assert len(a.records()) == 1
    assert storage.delivery_slide_ids(conn2, pid) <= service.current_restrictions(conn2).accepted
    # Receipts table has exactly one row for this delivery.
    n = conn2.execute(
        "SELECT COUNT(*) AS n FROM receipts WHERE delivery_id=?", (pid,)
    ).fetchone()["n"]
    assert n == 1
    conn2.close()


def test_late_rejection_cannot_undo_recorded_acceptance(env, monkeypatch):
    """A late/duplicate 'rejected' answer must not flip a recorded acceptance."""
    d = tempfile.mkdtemp(prefix="recv-")
    a = FakeReceiver(monkeypatch, os.path.join(d, "a.sqlite3"))
    url = "http://127.0.0.1:8001"
    route(monkeypatch, {url: a})
    conn, plan_id = seed(env)
    res = service.start_delivery(conn, plan_id, url)
    pid = res["delivery"]["id"]
    assert res["delivery"]["status"] == "accepted"
    # Force a contradictory late outcome through the same machinery.
    late = receiver_client.DeliveryOutcome(
        state="rejected", receipt_ref="RCPT-LATE", payload={"status": "rejected"}
    )
    out = service._apply_outcome(
        conn, pid, late, origin="late", receiver_id=a.identity, receiver_url=url
    )
    assert out["delivery"]["status"] == "accepted"
    assert out["delivery"]["receipt_ref"] == res["delivery"]["receipt_ref"]
    assert storage.delivery_slide_ids(conn, pid) <= service.current_restrictions(conn).accepted
    conn.close()


def test_restart_after_switch_keeps_restriction_and_does_not_contact_substitute(
    env, monkeypatch
):
    d = tempfile.mkdtemp(prefix="recv-")
    url_a, url_b = "http://127.0.0.1:8001", "http://127.0.0.1:8002"
    a = FakeReceiver(monkeypatch, os.path.join(d, "a.sqlite3"))
    b = FakeReceiver(monkeypatch, os.path.join(d, "b.sqlite3"))
    route(monkeypatch, {url_a: a, url_b: b})
    conn, plan_id = seed(env)
    real_send = service.receiver_client.send_delivery

    def lost(receiver_url, delivery_id, checksum, slides, timeout=10.0,
             expected_receiver_id=None):
        real_send(receiver_url, delivery_id, checksum, slides)
        return receiver_client.DeliveryOutcome(state="unknown", error="响应丢失")

    monkeypatch.setattr(service.receiver_client, "send_delivery", lost)
    pid = service.start_delivery(conn, plan_id, url_a)["delivery"]["id"]
    conn.close()

    # After "restart", the configured URL is B and A is offline.
    route(monkeypatch, {url_b: b})
    conn2 = storage.connect(env)
    service.startup_reconcile(conn2, url_b, timeout=1)
    delivery = storage.get_delivery(conn2, pid)
    assert delivery["status"] == "pending"
    assert delivery["receiver_id"] == a.identity
    assert storage.delivery_slide_ids(conn2, pid) <= (
        service.current_restrictions(conn2).uncertain
        | service.current_restrictions(conn2).accepted
    )
    assert b.records() == []
    conn2.close()


def test_receiver_identity_survives_process_restart(env, monkeypatch):
    d = tempfile.mkdtemp(prefix="recv-")
    path = os.path.join(d, "r.sqlite3")
    first = FakeReceiver(monkeypatch, path)
    conn = rm.connect()
    ident_a = rm.receiver_identity(conn)["receiver_id"]
    conn.close()
    assert ident_a == first.identity
    # Fresh connection (i.e. process restart) on the same file.
    conn = rm.connect()
    assert rm.receiver_identity(conn)["receiver_id"] == ident_a
    conn.close()
    # Fresh volume => different identity.
    other = FakeReceiver(monkeypatch, os.path.join(d, "other.sqlite3"))
    assert other.identity != ident_a


def test_receiver_rejects_post_with_foreign_expected_receiver_id(env, monkeypatch):
    """The receiver itself refuses to store a delivery bound to another volume."""
    d = tempfile.mkdtemp(prefix="recv-")
    a = FakeReceiver(monkeypatch, os.path.join(d, "a.sqlite3"))
    b = FakeReceiver(monkeypatch, os.path.join(d, "b.sqlite3"))
    slides = [
        {
            "slide_id": "S1",
            "patient_id": "P1",
            "repeat_group": "",
            "malignancy": "benign",
            "split": "train",
        }
    ]
    checksum = rm.checksum_of(slides)
    b._use()
    resp = b.client.post(
        "/api/deliveries",
        json={
            "delivery_id": "DLV-FOREIGN",
            "checksum": checksum,
            "slides": slides,
            "expected_receiver_id": a.identity,
        },
    )
    assert resp.status_code == 409
    assert resp.json()["receiver_id"] == b.identity
    assert b.records() == []

    # Without the foreign hint B would accept it; the sender still refuses to
    # treat a B receipt as confirmation of an A-bound delivery.
    ok = b.client.post(
        "/api/deliveries",
        json={"delivery_id": "DLV-OTHER", "checksum": checksum, "slides": slides},
    )
    assert ok.status_code == 201
    outcome = receiver_client._normalise(ok.json(), checksum)
    assert outcome.state == "accepted"
    assert outcome.receiver_id == b.identity


def test_send_response_from_wrong_receiver_is_unknown(monkeypatch):
    """A receipt carrying another volume's identity never confirms a delivery."""
    import httpx

    payload = {
        "delivery_id": "D1",
        "checksum": "abc",
        "status": "accepted",
        "receipt_ref": "RCPT-X",
        "receiver_id": "RCV-OTHER",
    }

    class Resp:
        status_code = 201
        text = ""

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **k):
            return Resp()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    out = receiver_client.send_delivery(
        "http://r", "D1", "abc", [{"slide_id": "S1"}], expected_receiver_id="RCV-MINE"
    )
    assert out.state == "unknown"
    assert "不一致" in (out.error or "")


def test_offline_original_does_not_block_other_deliveries(env, monkeypatch):
    """One receiver being offline must not stop deliveries to another."""
    d = tempfile.mkdtemp(prefix="recv-")
    url_a, url_b = "http://127.0.0.1:8001", "http://127.0.0.1:8002"
    a = FakeReceiver(monkeypatch, os.path.join(d, "a.sqlite3"))
    b = FakeReceiver(monkeypatch, os.path.join(d, "b.sqlite3"))
    route(monkeypatch, {url_a: a, url_b: b})
    conn, plan1 = seed(env)

    # Deliver to A while it is up, with a lost reply -> pending.
    real_send = service.receiver_client.send_delivery

    def lost(receiver_url, delivery_id, checksum, slides, timeout=10.0,
             expected_receiver_id=None):
        real_send(receiver_url, delivery_id, checksum, slides)
        return receiver_client.DeliveryOutcome(state="unknown", error="响应丢失")

    monkeypatch.setattr(service.receiver_client, "send_delivery", lost)
    pid_a = service.start_delivery(conn, plan1, url_a)["delivery"]["id"]
    monkeypatch.setattr(service.receiver_client, "send_delivery", real_send)

    # A goes offline; a new plan is still delivered to B without trouble.
    route(monkeypatch, {url_b: b})
    plan2 = service.publish_plan(conn, 1, 1)["plan_id"]
    res = service.start_delivery(conn, plan2, url_b)
    assert res["delivery"]["status"] == "accepted"
    assert len(b.records()) == 1
    # A's pending delivery is untouched and still bars its slides.
    pending = storage.get_delivery(conn, pid_a)
    assert pending["status"] == "pending"
    assert storage.delivery_slide_ids(conn, pid_a) <= service.current_restrictions(conn).barred
    # Reconcile while A is offline does not break anything.
    service.reconcile_pending(conn, url_b, timeout=1)
    assert storage.get_delivery(conn, pid_a)["status"] == "pending"
    assert len(b.records()) == 1
    conn.close()
