"""Delivery attribution (交付归属) tests.

The regression being pinned down: a training list is delivered to receiver A,
A accepts but the receipt is lost; the operator switches to receiver B (which
rejects) and hits "retry" — the old system let B's answer lift the test ban.
The same happened when the receiver's data volume was swapped at the same
address. These tests pin the fixed behaviour:

1. an old delivery is only ever confirmed/continued by its *owning* receiver —
   a changed address or a swapped volume never receives a re-send, and a
   substitute's accept/reject is ignored;
2. the original receiver, after a normal restart, still verifies/retries;
3. the same receiver identity at a *new* address is a relocation, not a
   substitute;
4. legacy rows without provable attribution keep their ban with reasons;
   existing acceptances stay permanent; an attributed terminal rejection
   lifts only its own restriction;
5. concurrent retries, late receipts and restarts never lose a restriction.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import csvio, receiver_client, service, storage  # noqa: E402
from receiver.main import app as receiver_app  # noqa: E402
import receiver.main as rm  # noqa: E402

SIMPLE = """slide_id,patient_id,repeat_group,malignancy
S1,P1,G1,benign
S2,P2,G1,malignant
S3,P3,,benign
S4,P4,,malignant
S5,P5,,benign
S6,P6,,malignant
"""


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def env(monkeypatch):
    d = tempfile.mkdtemp(prefix="attrib-")
    app_db = os.path.join(d, "app.sqlite3")
    monkeypatch.setenv("APP_DB_PATH", app_db)
    storage.reset_init_cache()
    yield app_db, d
    storage.reset_init_cache()


class ReceiverSim:
    """One independent receiver: own SQLite file, own instance identity."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.client = TestClient(receiver_app)
        self.online = True

    def request(self, method: str, path: str, **kwargs):
        old = rm.DB_PATH
        rm.DB_PATH = self.db_path
        try:
            return getattr(self.client, method)(path, **kwargs)
        finally:
            rm.DB_PATH = old

    def identity(self) -> str | None:
        if not self.online:
            return None
        return self.request("get", "/api/identity").json().get("receiver_id")

    def deliveries(self) -> list[dict]:
        return self.request("get", "/api/deliveries").json()

    def log(self) -> list[dict]:
        return self.request("get", "/api/log").json()


def wire_router(monkeypatch, mapping: dict[str, ReceiverSim]):
    """Route the sender's client calls to the simulated receiver for the URL."""

    def sim_for(url: str) -> ReceiverSim | None:
        sim = mapping.get(url.rstrip("/"))
        if sim is None or not sim.online:
            return None
        return sim

    def send(receiver_url, delivery_id, checksum, slides, timeout=10.0):
        sim = sim_for(receiver_url)
        if sim is None:
            return receiver_client.DeliveryOutcome(state="unknown", error="连接被拒")
        resp = sim.request(
            "post",
            "/api/deliveries",
            json={"delivery_id": delivery_id, "checksum": checksum, "slides": slides},
        )
        if resp.status_code not in (200, 201):
            return receiver_client.DeliveryOutcome(
                state="unknown",
                error=f"HTTP {resp.status_code}",
                http_status=resp.status_code,
            )
        return receiver_client._normalise(resp.json(), checksum)

    def query(receiver_url, delivery_id, expected_checksum=None, timeout=10.0):
        sim = sim_for(receiver_url)
        if sim is None:
            return receiver_client.DeliveryOutcome(state="unknown", error="连接被拒")
        resp = sim.request("get", f"/api/deliveries/{delivery_id}")
        if resp.status_code == 404:
            return receiver_client.DeliveryOutcome(
                state="unknown", error="查无记录", http_status=404
            )
        if resp.status_code != 200:
            return receiver_client.DeliveryOutcome(
                state="unknown",
                error=f"HTTP {resp.status_code}",
                http_status=resp.status_code,
            )
        return receiver_client._normalise(resp.json(), expected_checksum or "")

    def ident(receiver_url, timeout=10.0):
        sim = sim_for(receiver_url)
        if sim is None:
            return receiver_client.IdentityOutcome(error="连接被拒")
        resp = sim.request("get", "/api/identity")
        if resp.status_code != 200:
            return receiver_client.IdentityOutcome(
                error=f"HTTP {resp.status_code}", http_status=resp.status_code
            )
        return receiver_client.IdentityOutcome(
            receiver_id=resp.json().get("receiver_id")
        )

    monkeypatch.setattr(receiver_client, "send_delivery", send)
    monkeypatch.setattr(receiver_client, "query_delivery", query)
    monkeypatch.setattr(receiver_client, "fetch_identity", ident)
    monkeypatch.setattr(service.receiver_client, "send_delivery", send)
    monkeypatch.setattr(service.receiver_client, "query_delivery", query)
    monkeypatch.setattr(service.receiver_client, "fetch_identity", ident)


def seed(app_db):
    conn = storage.connect(app_db)
    rows = csvio.parse_csv_text(SIMPLE)
    service.append_slides(conn, [r.__dict__ for r in rows])
    plan_id = service.publish_plan(conn, 1, 1)["plan_id"]
    return conn, plan_id


# -------------------------------------------------------------------- tests


def test_lost_receipt_then_retry_to_substitute_is_blocked(env, monkeypatch):
    """A accepted but the receipt was lost; switching to a rejecting B and
    retrying must NOT deliver to B, must NOT lift the ban, and must still
    succeed once A is reachable again."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    a = ReceiverSim(os.path.join(d, "a.sqlite3"))
    b = ReceiverSim(os.path.join(d, "b.sqlite3"))
    a.request("post", "/api/simulate", json={"fault": "drop_after_store", "delay": "0.01"})
    b.request("post", "/api/simulate", json={"refuse": "1"})
    wire_router(monkeypatch, {"http://a": a, "http://b": b})

    res = service.start_delivery(conn, plan_id, "http://a")
    dlv = res["delivery"]
    assert dlv["status"] == "pending"
    assert dlv["receiver_url"] == "http://a"
    assert dlv["receiver_id"] == a.identity()
    # A really stored and accepted it; only the response was lost
    stored = a.deliveries()
    assert len(stored) == 1 and stored[0]["status"] == "accepted"

    train_ids = set(storage.delivery_slide_ids(conn, dlv["id"]))
    assert train_ids <= service.current_restrictions(conn).uncertain

    # operator switches the configured address to B and hits 重试/核实
    a.online = False  # original receiver is offline right now
    retried = service.retry_delivery(conn, dlv["id"], "http://b")
    assert retried["delivery"]["status"] == "pending"
    assert "替代方" in retried["message"]
    assert b.deliveries() == []  # substitute was never contacted with content
    assert b.log() == []  # not even a verification request reached B
    # B's refusal did not lift the training-exposure ban
    assert train_ids <= service.current_restrictions(conn).uncertain

    verified = service.verify_delivery(conn, dlv["id"], "http://b")
    assert verified["delivery"]["status"] == "pending"
    assert b.deliveries() == []

    # the original receiver comes back: verify/retry complete normally
    a.online = True
    a.request("post", "/api/simulate", json={"fault": "none"})
    ok = service.verify_delivery(conn, dlv["id"], "http://b")  # current URL is B!
    assert ok["delivery"]["status"] == "accepted"
    assert ok["delivery"]["receipt_ref"] == stored[0]["receipt_ref"]
    assert train_ids <= service.current_restrictions(conn).accepted
    conn.close()


def test_same_address_volume_swap_is_blocked(env, monkeypatch):
    """Same address, swapped receiver data volume => different identity =>
    never re-send; restoring the original volume lets verification through."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    a1 = ReceiverSim(os.path.join(d, "a-v1.sqlite3"))
    a1.request("post", "/api/simulate", json={"fault": "drop_after_store", "delay": "0.01"})
    mapping = {"http://a": a1}
    wire_router(monkeypatch, mapping)

    res = service.start_delivery(conn, plan_id, "http://a")
    dlv = res["delivery"]
    assert dlv["status"] == "pending"
    rid_v1 = dlv["receiver_id"]
    assert rid_v1

    # the volume behind http://a is replaced: new identity, simulating refusal
    a2 = ReceiverSim(os.path.join(d, "a-v2.sqlite3"))
    a2.request("post", "/api/simulate", json={"refuse": "1"})
    assert a2.identity() != rid_v1
    mapping["http://a"] = a2

    retried = service.retry_delivery(conn, dlv["id"], "http://a")
    assert retried["delivery"]["status"] == "pending"
    assert "身份不符" in retried["message"]
    assert a2.deliveries() == []  # nothing was re-sent to the substitute volume
    train_ids = set(storage.delivery_slide_ids(conn, dlv["id"]))
    assert train_ids <= service.current_restrictions(conn).uncertain

    verified = service.verify_delivery(conn, dlv["id"], "http://a")
    assert verified["delivery"]["status"] == "pending"
    assert a2.deliveries() == []

    # original volume restored -> verification succeeds against the true owner
    mapping["http://a"] = a1
    a1.request("post", "/api/simulate", json={"fault": "none"})
    ok = service.verify_delivery(conn, dlv["id"], "http://a")
    assert ok["delivery"]["status"] == "accepted"
    assert ok["delivery"]["receipt_ref"] == a1.deliveries()[0]["receipt_ref"]
    conn.close()


def test_original_receiver_restart_still_verifies(env, monkeypatch):
    """A normal restart of the original receiver (same volume) keeps the
    identity, so pending deliveries verify and retry as before."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    db_a = os.path.join(d, "a.sqlite3")
    a = ReceiverSim(db_a)
    a.request("post", "/api/simulate", json={"fault": "timeout", "delay": "0.01"})
    mapping = {"http://a": a}
    wire_router(monkeypatch, mapping)

    res = service.start_delivery(conn, plan_id, "http://a")
    dlv = res["delivery"]
    assert dlv["status"] == "pending"
    rid = dlv["receiver_id"]
    receipt = a.deliveries()[0]["receipt_ref"]

    # "restart": a fresh process over the same data volume
    a2 = ReceiverSim(db_a)
    assert a2.identity() == rid
    mapping["http://a"] = a2

    ok = service.verify_delivery(conn, dlv["id"], "http://a")
    assert ok["delivery"]["status"] == "accepted"
    assert ok["delivery"]["receipt_ref"] == receipt
    # retry after terminal state is a no-op; still exactly one receiver record
    again = service.retry_delivery(conn, dlv["id"], "http://a")
    assert again["reused"] is True
    assert len(a2.deliveries()) == 1
    conn.close()


def test_same_identity_at_new_address_is_relocation(env, monkeypatch):
    """Moving the *same* receiver volume to a new address is not a substitute:
    the pending delivery follows the proven identity and the recorded address
    is updated."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    db_a = os.path.join(d, "a.sqlite3")
    a = ReceiverSim(db_a)
    a.request("post", "/api/simulate", json={"fault": "drop_after_store", "delay": "0.01"})
    mapping = {"http://a": a}
    wire_router(monkeypatch, mapping)

    res = service.start_delivery(conn, plan_id, "http://a")
    dlv = res["delivery"]
    assert dlv["status"] == "pending"

    # receiver moves to a new address with the same volume (same identity)
    moved = ReceiverSim(db_a)
    mapping.clear()
    mapping["http://a-new"] = moved

    ok = service.verify_delivery(conn, dlv["id"], "http://a-new")
    assert ok["delivery"]["status"] == "accepted"
    assert ok["delivery"]["receiver_url"] == "http://a-new"
    kinds = [e["kind"] for e in storage.list_events(conn, 50)]
    assert "receiver_relocated" in kinds
    conn.close()


def test_legacy_unattributed_rows_keep_ban_with_reasons(env, monkeypatch):
    """Old-DB rows without provable attribution: unknown *and* rejected keep
    the ban (with reasons), accepted stays permanent, and they are never
    verified/re-sent anywhere."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    slides = {s["slide_id"]: s for s in storage.all_slides(conn)}

    def row(sid):
        s = slides[sid]
        return {
            "slide_id": s["slide_id"],
            "patient_id": s["patient_id"],
            "repeat_group": s["repeat_group"],
            "malignancy": s["malignancy"],
        }

    with storage.transaction(conn):
        # legacy pending (unknown) — no receiver_url / receiver_id
        storage.insert_delivery(
            conn, delivery_id="DLV-LEG-PEND", plan_id=plan_id, split="train",
            checksum="c1", attempt=1, rows=[row("S1")],
        )
        # legacy rejected — under the old bug this would have lifted the ban
        storage.insert_delivery(
            conn, delivery_id="DLV-LEG-REJ", plan_id=plan_id, split="train",
            checksum="c2", attempt=1, rows=[row("S3")],
        )
        storage.set_delivery_terminal(
            conn, "DLV-LEG-REJ", status="rejected", receiver_status="rejected",
            receipt_ref="RCPT-OLD", receipt_json="{}",
        )
        # legacy accepted — permanent, as before
        storage.insert_delivery(
            conn, delivery_id="DLV-LEG-ACC", plan_id=plan_id, split="train",
            checksum="c3", attempt=1, rows=[row("S5")],
        )
        storage.set_delivery_terminal(
            conn, "DLV-LEG-ACC", status="accepted", receiver_status="accepted",
            receipt_ref="RCPT-OLD2", receipt_json="{}",
        )
        service.refresh_plans(conn)

    state = service.current_restrictions(conn)
    assert {"S1", "S2"} <= state.barred  # pending drags its connected group
    assert "S3" in state.barred  # unattributed rejection does NOT lift the ban
    assert "S5" in state.accepted  # existing acceptance stays permanent
    assert "无法证明" in (state.explain("S1") or "")
    assert "拒收" in (state.explain("S3") or "") and "无法证明" in (state.explain("S3") or "")

    def forbidden(*args, **kwargs):  # no network may happen for legacy rows
        raise AssertionError("不得为无归属交付发起任何网络请求")

    monkeypatch.setattr(service.receiver_client, "send_delivery", forbidden)
    monkeypatch.setattr(service.receiver_client, "query_delivery", forbidden)
    monkeypatch.setattr(service.receiver_client, "fetch_identity", forbidden)

    res = service.verify_delivery(conn, "DLV-LEG-PEND", "http://a")
    assert res["delivery"]["status"] == "pending"
    assert "归属" in res["message"]
    res2 = service.retry_delivery(conn, "DLV-LEG-PEND", "http://a")
    assert res2["delivery"]["status"] == "pending"
    assert "归属" in res2["message"]

    # startup reconcile reports the legacy rows and keeps every ban
    service.startup_reconcile(conn, "http://a", timeout=0.1)
    events = storage.list_events(conn, 50)
    assert any(e["kind"] == "legacy_deliveries" for e in events)
    state = service.current_restrictions(conn)
    assert {"S1", "S2", "S3"} <= state.barred
    conn.close()


def test_attributed_rejection_lifts_only_its_own(env, monkeypatch):
    """A terminal rejection *from the owning receiver* lifts exactly its own
    restriction; the delivery row carries the receiver identity."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    a = ReceiverSim(os.path.join(d, "a.sqlite3"))
    a.request("post", "/api/simulate", json={"refuse": "1"})
    wire_router(monkeypatch, {"http://a": a})

    res = service.start_delivery(conn, plan_id, "http://a")
    dlv = res["delivery"]
    assert dlv["status"] == "rejected"
    assert dlv["receiver_id"] == a.identity()
    train_ids = set(storage.delivery_slide_ids(conn, dlv["id"]))
    state = service.current_restrictions(conn)
    assert not (train_ids & state.barred)
    conn.close()


def test_reconcile_uses_recorded_receiver_not_current_url(env, monkeypatch):
    """Startup/restart reconciliation queries each pending delivery at its
    owning receiver, regardless of the currently configured address."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    a = ReceiverSim(os.path.join(d, "a.sqlite3"))
    b = ReceiverSim(os.path.join(d, "b.sqlite3"))
    a.request("post", "/api/simulate", json={"fault": "drop_after_store", "delay": "0.01"})
    wire_router(monkeypatch, {"http://a": a, "http://b": b})

    res = service.start_delivery(conn, plan_id, "http://a")
    dlv = res["delivery"]
    assert dlv["status"] == "pending"
    a.request("post", "/api/simulate", json={"fault": "none"})

    # configured address is now B; reconciliation must still talk to A
    results = service.reconcile_pending(conn, "http://b")
    assert results and results[0]["delivery"]["status"] == "accepted"
    assert results[0]["delivery"]["receipt_ref"] == a.deliveries()[0]["receipt_ref"]
    assert b.deliveries() == [] and b.log() == []
    conn.close()


def test_concurrent_retries_keep_single_terminal(env, monkeypatch):
    """Two concurrent retries of the same pending delivery: exactly one
    receiver record, one terminal state, and the ban is never lost."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    a = ReceiverSim(os.path.join(d, "a.sqlite3"))
    a.request("post", "/api/simulate", json={"fault": "drop_after_store", "delay": "0.01"})
    wire_router(monkeypatch, {"http://a": a})

    res = service.start_delivery(conn, plan_id, "http://a")
    dlv = res["delivery"]
    assert dlv["status"] == "pending"
    a.request("post", "/api/simulate", json={"fault": "slow", "delay": "0.05"})

    barrier = threading.Barrier(2)
    results, errors = [], []

    def worker():
        c = storage.connect(app_db)
        try:
            barrier.wait(timeout=10)
            results.append(service.retry_delivery(c, dlv["id"], "http://a"))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)
        finally:
            c.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert len(results) == 2
    final = storage.get_delivery(conn, dlv["id"])
    assert final["status"] == "accepted"
    recs = a.deliveries()
    assert len(recs) == 1
    assert final["receipt_ref"] == recs[0]["receipt_ref"]
    train_ids = set(storage.delivery_slide_ids(conn, dlv["id"]))
    assert train_ids <= service.current_restrictions(conn).accepted
    conn.close()


def test_late_conflicting_receipt_never_overrides_terminal(env, monkeypatch):
    """A late, contradictory receipt (even from the owning receiver) must not
    rewrite an already-terminal delivery or its restriction."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    a = ReceiverSim(os.path.join(d, "a.sqlite3"))
    wire_router(monkeypatch, {"http://a": a})

    res = service.start_delivery(conn, plan_id, "http://a")
    dlv = res["delivery"]
    assert dlv["status"] == "accepted"
    receipt = dlv["receipt_ref"]
    train_ids = set(storage.delivery_slide_ids(conn, dlv["id"]))

    late = receiver_client.DeliveryOutcome(
        state="rejected", receipt_ref="RCPT-LATE", receiver_id=a.identity()
    )
    out = service._apply_outcome(conn, dlv["id"], late, origin="retry")
    final = out["delivery"]
    assert final["status"] == "accepted"
    assert final["receipt_ref"] == receipt
    assert train_ids <= service.current_restrictions(conn).accepted
    kinds = [e["kind"] for e in storage.list_events(conn, 50)]
    assert "delivery_conflict" in kinds
    conn.close()


def test_response_from_non_owner_is_ignored(env, monkeypatch):
    """Defence in depth: even if a response somehow arrives from a receiver
    whose identity does not match the delivery's owner, it is ignored."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    a = ReceiverSim(os.path.join(d, "a.sqlite3"))
    a.request("post", "/api/simulate", json={"fault": "drop_after_store", "delay": "0.01"})
    wire_router(monkeypatch, {"http://a": a})

    res = service.start_delivery(conn, plan_id, "http://a")
    dlv = res["delivery"]
    assert dlv["status"] == "pending"

    forged = receiver_client.DeliveryOutcome(
        state="rejected", receipt_ref="RCPT-FORGED", receiver_id="RCV-SOMEONE-ELSE"
    )
    out = service._apply_outcome(conn, dlv["id"], forged, origin="verify")
    assert out["delivery"]["status"] == "pending"
    train_ids = set(storage.delivery_slide_ids(conn, dlv["id"]))
    assert train_ids <= service.current_restrictions(conn).uncertain
    kinds = [e["kind"] for e in storage.list_events(conn, 50)]
    assert "delivery_blocked" in kinds
    conn.close()


def test_owner_offline_does_not_block_other_deliveries(env, monkeypatch):
    """The original receiver being offline only affects its own deliveries:
    new deliveries (other plans) to the current address keep working."""
    app_db, d = env
    conn, plan_id = seed(app_db)
    a = ReceiverSim(os.path.join(d, "a.sqlite3"))
    b = ReceiverSim(os.path.join(d, "b.sqlite3"))
    a.request("post", "/api/simulate", json={"fault": "drop_after_store", "delay": "0.01"})
    wire_router(monkeypatch, {"http://a": a, "http://b": b})

    first = service.start_delivery(conn, plan_id, "http://a")
    assert first["delivery"]["status"] == "pending"
    a.online = False  # original owner goes offline

    # a different plan's delivery to the *current* address still works
    plan2 = service.publish_plan(conn, 1, 1)["plan_id"]
    second = service.start_delivery(conn, plan2, "http://b")
    assert second["delivery"]["status"] == "accepted"
    assert second["delivery"]["receiver_id"] == b.identity()
    assert len(b.deliveries()) == 1
    # the first delivery is untouched and still bars its slides
    assert storage.get_delivery(conn, first["delivery"]["id"])["status"] == "pending"
    conn.close()
