"""Recovery / crash tests.

The point is the *durable* part of the protocol: a pending record must exist
before the network call, and a restart must be able to verify it, without
duplicating an acceptance or losing a restriction.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import csvio, receiver_client, service, storage  # noqa: E402
from receiver.main import app as receiver_app  # noqa: E402
import receiver.main as rm  # noqa: E402


SAMPLE = """slide_id,patient_id,repeat_group,malignancy
S1,P1,G1,benign
S2,P2,G1,malignant
S3,P3,,benign
S4,P4,,malignant
S5,P5,,benign
S6,P6,,malignant
"""


@pytest.fixture()
def env(monkeypatch):
    d = tempfile.mkdtemp(prefix="recover-")
    app_db = os.path.join(d, "app.sqlite3")
    recv_db = os.path.join(d, "receiver.sqlite3")
    monkeypatch.setenv("APP_DB_PATH", app_db)
    monkeypatch.setenv("RECEIVER_DB_PATH", recv_db)
    storage.reset_init_cache()
    rm.DB_PATH = recv_db
    rm.connect().close()
    yield app_db, recv_db
    storage.reset_init_cache()


def _wire(monkeypatch, fault="none", refuse=False):
    recv = TestClient(receiver_app)
    recv.post(
        "/api/simulate",
        json={"fault": fault, "refuse": "1" if refuse else "0", "delay": "0.01"},
    )
    identity = recv.get("/api/identity").json()["receiver_id"]

    def fetch_identity(receiver_url, timeout=10.0):
        return identity

    def send(
        receiver_url,
        delivery_id,
        checksum,
        slides,
        timeout=10.0,
        expected_receiver_id=None,
    ):
        payload = {
            "delivery_id": delivery_id,
            "checksum": checksum,
            "slides": slides,
        }
        if expected_receiver_id:
            payload["expected_receiver_id"] = expected_receiver_id
        resp = recv.post("/api/deliveries", json=payload)
        if resp.status_code not in (200, 201):
            return receiver_client.DeliveryOutcome(
                state="unknown", error=f"HTTP {resp.status_code}", http_status=resp.status_code
            )
        return receiver_client._normalise(resp.json(), checksum)

    def query(
        receiver_url,
        delivery_id,
        expected_checksum=None,
        timeout=10.0,
        expected_receiver_id=None,
    ):
        resp = recv.get(f"/api/deliveries/{delivery_id}")
        if resp.status_code == 404:
            return receiver_client.DeliveryOutcome(state="unknown", error="查无记录", http_status=404)
        if resp.status_code != 200:
            return receiver_client.DeliveryOutcome(
                state="unknown", error=f"HTTP {resp.status_code}", http_status=resp.status_code
            )
        data = resp.json()
        if expected_receiver_id and data.get("receiver_id") != expected_receiver_id:
            return receiver_client.DeliveryOutcome(state="unknown", error="接收方身份不一致")
        return receiver_client._normalise(data, expected_checksum or "")

    monkeypatch.setattr(receiver_client, "fetch_identity", fetch_identity)
    monkeypatch.setattr(receiver_client, "send_delivery", send)
    monkeypatch.setattr(receiver_client, "query_delivery", query)
    monkeypatch.setattr(service.receiver_client, "fetch_identity", fetch_identity)
    monkeypatch.setattr(service.receiver_client, "send_delivery", send)
    monkeypatch.setattr(service.receiver_client, "query_delivery", query)
    return recv


def _seed(app_db):
    conn = storage.connect(app_db)
    rows = csvio.parse_csv_text(SAMPLE)
    service.append_slides(conn, [r.__dict__ for r in rows])
    plan_id = service.publish_plan(conn, 1, 1)["plan_id"]
    return conn, plan_id


def test_crash_before_post_leaves_verifiable_pending(env, monkeypatch):
    """The process dies after the pending row but before the POST."""
    app_db, _ = env
    conn, plan_id = _seed(app_db)
    recv_db = env[1]
    import receiver.main as rm

    rm.DB_PATH = recv_db

    recv = TestClient(receiver_app)
    identity = recv.get("/api/identity").json()["receiver_id"]
    monkeypatch.setattr(
        service.receiver_client, "fetch_identity", lambda url, timeout=10.0: identity
    )

    def boom(*args, **kwargs):
        raise KeyboardInterrupt("simulated crash during POST")

    monkeypatch.setattr(service.receiver_client, "send_delivery", boom)
    with pytest.raises(KeyboardInterrupt):
        service.start_delivery(conn, plan_id, "http://receiver")

    pending = [d for d in storage.list_deliveries(conn) if d["status"] == "pending"]
    assert len(pending) == 1
    pid = pending[0]["id"]
    assert pending[0]["receiver_id"] == identity
    # restriction survived: slides of that delivery are barred from test
    state = service.current_restrictions(conn)
    assert set(storage.delivery_slide_ids(conn, pid)) <= state.uncertain
    conn.close()

    # "restart": new connection, reconcile against a receiver that has no record
    _wire(monkeypatch, fault="none")
    monkeypatch.setattr(
        receiver_client, "fetch_identity", lambda url, timeout=10.0: identity
    )
    conn2 = storage.connect(app_db)
    results = service.reconcile_pending(conn2, "http://receiver")
    assert results and results[0]["delivery"]["status"] == "pending"  # 404 => unknown
    assert recv.get(f"/api/deliveries/{pid}").status_code == 404
    # then an explicit retry delivers idempotently
    res = service.retry_delivery(conn2, pid, "http://receiver")
    assert res["delivery"]["status"] == "accepted"
    assert len(recv.get("/api/deliveries").json()) == 1
    conn2.close()


def test_crash_after_receiver_store_before_local_commit(env, monkeypatch):
    """Receiver stored and returned; the sender dies before recording it."""
    app_db, _ = env
    conn, plan_id = _seed(app_db)

    def crash_after_send(
        receiver_url,
        delivery_id,
        checksum,
        slides,
        timeout=10.0,
        expected_receiver_id=None,
    ):
        # perform the real POST through the receiver ASGI app, then "crash"
        recv = TestClient(receiver_app)
        payload = {
            "delivery_id": delivery_id,
            "checksum": checksum,
            "slides": slides,
        }
        if expected_receiver_id:
            payload["expected_receiver_id"] = expected_receiver_id
        recv.post("/api/deliveries", json=payload)
        raise KeyboardInterrupt("simulated crash after receiver stored the delivery")

    monkeypatch.setattr(service.receiver_client, "send_delivery", crash_after_send)
    recv_probe = TestClient(receiver_app)
    identity = recv_probe.get("/api/identity").json()["receiver_id"]
    monkeypatch.setattr(
        service.receiver_client, "fetch_identity", lambda url, timeout=10.0: identity
    )
    with pytest.raises(KeyboardInterrupt):
        service.start_delivery(conn, plan_id, "http://receiver")
    pending = [d for d in storage.list_deliveries(conn) if d["status"] == "pending"]
    assert len(pending) == 1
    pid = pending[0]["id"]
    conn.close()

    recv = TestClient(receiver_app)
    stored = recv.get(f"/api/deliveries/{pid}")
    assert stored.status_code == 200
    receipt = stored.json()["receipt_ref"]

    def query(
        receiver_url,
        delivery_id,
        expected_checksum=None,
        timeout=10.0,
        expected_receiver_id=None,
    ):
        resp = recv.get(f"/api/deliveries/{delivery_id}")
        if resp.status_code == 404:
            return receiver_client.DeliveryOutcome(state="unknown", error="查无记录")
        data = resp.json()
        if expected_receiver_id and data.get("receiver_id") != expected_receiver_id:
            return receiver_client.DeliveryOutcome(state="unknown", error="接收方身份不一致")
        return receiver_client._normalise(data, expected_checksum or "")

    monkeypatch.setattr(receiver_client, "query_delivery", query)
    monkeypatch.setattr(service.receiver_client, "query_delivery", query)

    conn2 = storage.connect(app_db)
    results = service.reconcile_pending(conn2, "http://receiver")
    delivery = results[0]["delivery"]
    assert delivery["status"] == "accepted"
    assert delivery["receipt_ref"] == receipt
    # exactly one receiver record, one receipt
    assert len(recv.get("/api/deliveries").json()) == 1
    # accepted => permanent ban derived from DB, no cached state needed
    state = service.current_restrictions(conn2)
    assert set(storage.delivery_slide_ids(conn2, pid)) <= state.accepted
    conn2.close()


def test_restart_with_receiver_down_keeps_restrictions(env, monkeypatch):
    app_db, _ = env
    conn, plan_id = _seed(app_db)

    def down(*args, **kwargs):
        return receiver_client.DeliveryOutcome(state="unknown", error="连接被拒")

    monkeypatch.setattr(service.receiver_client, "send_delivery", down)
    res = service.start_delivery(conn, plan_id, "http://receiver")
    pid = res["delivery"]["id"]
    assert res["delivery"]["status"] == "pending"
    conn.close()

    # restart with the receiver still unreachable
    def query_down(*args, **kwargs):
        return receiver_client.DeliveryOutcome(state="unknown", error="连接被拒")

    monkeypatch.setattr(receiver_client, "query_delivery", query_down)
    monkeypatch.setattr(service.receiver_client, "query_delivery", query_down)
    conn2 = storage.connect(app_db)
    service.reconcile_pending(conn2, "http://receiver")
    delivery = storage.get_delivery(conn2, pid)
    assert delivery["status"] == "pending"
    assert delivery["receiver_status"] == "unknown"
    state = service.current_restrictions(conn2)
    assert set(storage.delivery_slide_ids(conn2, pid)) <= state.uncertain
    conn2.close()


def test_same_content_is_not_delivered_twice(env, monkeypatch):
    app_db, _ = env
    conn, plan_id = _seed(app_db)
    recv = _wire(monkeypatch, fault="none")
    first = service.start_delivery(conn, plan_id, "http://receiver")
    assert first["delivery"]["status"] == "accepted"
    second = service.start_delivery(conn, plan_id, "http://receiver")
    assert second["reused"] is True
    assert second["delivery"]["id"] == first["delivery"]["id"]
    assert len(recv.get("/api/deliveries").json()) == 1
    assert len(storage.list_deliveries(conn)) == 1


def test_rejected_then_redelivered_creates_new_delivery(env, monkeypatch):
    app_db, _ = env
    conn, plan_id = _seed(app_db)
    _wire(monkeypatch, fault="none", refuse=True)
    first = service.start_delivery(conn, plan_id, "http://receiver")
    assert first["delivery"]["status"] == "rejected"
    _wire(monkeypatch, fault="none", refuse=False)
    second = service.start_delivery(conn, plan_id, "http://receiver")
    assert second["delivery"]["status"] == "accepted"
    assert second["delivery"]["id"] != first["delivery"]["id"]
    statuses = {d["id"]: d["status"] for d in storage.list_deliveries(conn)}
    assert statuses[first["delivery"]["id"]] == "rejected"
    assert statuses[second["delivery"]["id"]] == "accepted"


def test_unknown_then_verify_timeout_stays_unknown(env, monkeypatch):
    app_db, _ = env
    conn, plan_id = _seed(app_db)
    _wire(monkeypatch, fault="http404")
    res = service.start_delivery(conn, plan_id, "http://receiver")
    pid = res["delivery"]["id"]
    assert res["delivery"]["status"] == "pending"
    _wire(monkeypatch, fault="http_verify500")
    again = service.verify_delivery(conn, pid, "http://receiver")
    assert again["delivery"]["status"] == "pending"
    assert again["delivery"]["receiver_status"] == "unknown"
    state = service.current_restrictions(conn)
    assert set(storage.delivery_slide_ids(conn, pid)) <= state.uncertain
