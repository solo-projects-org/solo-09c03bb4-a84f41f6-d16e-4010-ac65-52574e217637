"""Integration tests: real allocation app + real receiver, both in-process
ASGI apps talking over an in-memory HTTP transport.

Covered scenarios
-----------------
1. exact split honours transitive groups and both malignancy quotas;
2. no solution is reported with a concrete reason;
3. undelivered slides (and their whole group) cannot enter the test split;
4. pending record exists before the receiver call; timeout => unknown + bar;
5. retry after a timeout is idempotent (one receipt, one acceptance);
6. accepted => permanent test ban, plan re-judged and suspended;
7. rejected only lifts its own restriction;
8. later upload that breaks a frozen group suspends the plan, restore re-enables;
9. capacity limit is enforced without partial writes;
10. stale computation is never published.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import allocator, csvio, receiver_client, service, storage  # noqa: E402
from app.main import app as app_app  # noqa: E402
from receiver.main import app as receiver_app  # noqa: E402


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def tmpdb(monkeypatch):
    directory = tempfile.mkdtemp(prefix="path-test-")
    app_db = os.path.join(directory, "app.sqlite3")
    recv_db = os.path.join(directory, "receiver.sqlite3")
    monkeypatch.setenv("APP_DB_PATH", app_db)
    monkeypatch.setenv("RECEIVER_DB_PATH", recv_db)
    storage.reset_init_cache()
    import receiver.main as rm

    rm.DB_PATH = recv_db
    rm.connect().close()
    yield app_db, recv_db
    storage.reset_init_cache()


@pytest.fixture()
def client(tmpdb):
    return TestClient(app_app)


@pytest.fixture()
def conn(tmpdb):
    c = storage.connect(tmpdb[0])
    yield c
    c.close()


def wire(tmpdb, monkeypatch, fault="none", refuse=False):
    """Route the sender's httpx calls into the receiver ASGI app."""
    recv = TestClient(receiver_app)
    recv.post(
        "/api/simulate",
        json={"fault": fault, "refuse": "1" if refuse else "0", "delay": "0.01"},
    )
    real_send = receiver_client.send_delivery
    real_query = receiver_client.query_delivery

    def send(receiver_url, delivery_id, checksum, slides, timeout=10.0):
        try:
            resp = recv.post(
                "/api/deliveries",
                json={
                    "delivery_id": delivery_id,
                    "checksum": checksum,
                    "slides": slides,
                },
            )
        except Exception as exc:  # pragma: no cover
            return receiver_client.DeliveryOutcome(state="unknown", error=str(exc))
        if resp.status_code not in (200, 201):
            return receiver_client.DeliveryOutcome(
                state="unknown",
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                http_status=resp.status_code,
            )
        return receiver_client._normalise(resp.json(), checksum)

    def query(receiver_url, delivery_id, expected_checksum=None, timeout=10.0):
        resp = recv.get(f"/api/deliveries/{delivery_id}")
        if resp.status_code == 404:
            return receiver_client.DeliveryOutcome(
                state="unknown", error="查无记录", http_status=404
            )
        if resp.status_code != 200:
            return receiver_client.DeliveryOutcome(
                state="unknown", error=f"HTTP {resp.status_code}", http_status=resp.status_code
            )
        return receiver_client._normalise(resp.json(), expected_checksum or "")

    monkeypatch.setattr(receiver_client, "send_delivery", send)
    monkeypatch.setattr(receiver_client, "query_delivery", query)
    import app.service as svc

    monkeypatch.setattr(svc.receiver_client, "send_delivery", send)
    monkeypatch.setattr(svc.receiver_client, "query_delivery", query)
    return recv, real_send, real_query


def upload(conn, text: str):
    rows = csvio.parse_csv_text(text)
    return service.append_slides(conn, [r.__dict__ for r in rows])


SIMPLE = """slide_id,patient_id,repeat_group,malignancy
S1,P1,G1,benign
S2,P2,G1,malignant
S3,P3,,benign
S4,P4,,malignant
S5,P5,,benign
S6,P6,,malignant
"""


# -------------------------------------------------------------------- tests


def test_transitive_grouping_and_exact_split(conn):
    upload(conn, SIMPLE)
    out = service.publish_plan(conn, 1, 1)
    res = out["result"]
    assert res.test_benign == 1 and res.test_malignant == 1
    membership = storage.plan_membership(conn, out["plan_id"])
    assert membership["S1"] == membership["S2"] == "test" or (
        membership["S1"] == membership["S2"] == "train"
    )
    test = [s for s, m in membership.items() if m == "test"]
    assert len(test) == 2
    assert set(test) not in ({"S1"}, {"S2"})
    # every slide is accounted for exactly once
    assert len(membership) == 6


def test_transitive_chain_via_two_groups():
    slides = [
        dict(slide_id="A", patient_id="P1", repeat_group="G1", malignancy="benign"),
        dict(slide_id="B", patient_id="P2", repeat_group="G1", malignancy="benign"),
        dict(slide_id="C", patient_id="P2", repeat_group="G2", malignancy="malignant"),
        dict(slide_id="D", patient_id="P3", repeat_group="G2", malignancy="benign"),
        dict(slide_id="E", patient_id="P4", repeat_group=None, malignancy="benign"),
    ]
    groups = allocator.build_groups(slides)
    ids = [g.slide_ids for g in groups]
    assert {"A", "B", "C", "D"} in ids
    assert {"E"} in ids


def test_unsolvable_reports_reason(conn):
    # one group of two benign, plus singletons; asking for 1 benign is impossible
    upload(
        conn,
        "slide_id,patient_id,repeat_group,malignancy\n"
        "A1,PA1,G9,benign\nA2,PA2,G9,benign\n"
        "B1,PB1,,malignant\n",
    )
    with pytest.raises(allocator.Unsolvable) as exc:
        service.publish_plan(conn, 1, 1)
    assert "连通组" in exc.value.reason
    assert storage.list_plans(conn) == []


def test_undelivered_slide_and_group_cannot_enter_test(conn):
    upload(conn, SIMPLE)
    # deliver S1 (which drags S2 by group) pending, then that whole group is barred
    out = service.publish_plan(conn, 1, 1)
    plan_id = out["plan_id"]
    # choose a plan where S3/S4 are test, then accept a delivery containing S1.
    with storage.transaction(conn):
        storage.insert_delivery(
            conn,
            delivery_id="DLV-TEST",
            plan_id=plan_id,
            split="train",
            checksum="x",
            attempt=1,
            rows=[
                {
                    "slide_id": "S1",
                    "patient_id": "P1",
                    "repeat_group": "G1",
                    "malignancy": "benign",
                }
            ],
        )
    state = service.current_restrictions(conn)
    assert {"S1", "S2"} <= state.barred  # group drag
    assert "S3" not in state.barred


def test_pending_then_timeout_then_idempotent_retry(conn, tmpdb, monkeypatch):
    upload(conn, SIMPLE)
    plan_id = service.publish_plan(conn, 1, 1)["plan_id"]
    recv, _, _ = wire(tmpdb, monkeypatch, fault="timeout")
    res = service.start_delivery(conn, plan_id, "http://receiver", timeout=1)
    delivery = res["delivery"]
    assert delivery["status"] == "pending"
    assert delivery["receiver_status"] == "unknown"
    # receiver DID store it despite the timeout
    recs = recv.get("/api/deliveries").json()
    assert len(recs) == 1 and recs[0]["status"] == "accepted"

    # pending delivery bars its slides (and groups) from the *test* side; the
    # frozen plan keeps those slides in train, so its own isolation still holds
    state = service.current_restrictions(conn)
    train_ids = {r["slide_id"] for r in storage.plan_slides(conn, plan_id, "train")}
    assert train_ids <= state.uncertain
    assert storage.get_plan(conn, plan_id)["status"] == "valid"
    # but any new plan may no longer put them in test
    try:
        other = service.publish_plan(conn, 1, 1)["plan_id"]
        membership = storage.plan_membership(conn, other)
        assert all(membership[sid] != "test" for sid in train_ids)
    except allocator.Unsolvable as exc:
        assert "禁入" in exc.reason or "连通组" in exc.reason

    verified = service.verify_delivery(conn, delivery["id"], "http://receiver")
    assert verified["delivery"]["status"] == "accepted"
    receipt = verified["delivery"]["receipt_ref"]
    assert receipt and receipt.startswith("RCPT-")

    # no second acceptance on retry
    again = service.retry_delivery(conn, delivery["id"], "http://receiver")
    assert again["delivery"]["status"] == "accepted"
    assert again["delivery"]["receipt_ref"] == receipt
    assert len(recv.get("/api/deliveries").json()) == 1


def test_accepted_is_permanent_test_ban_and_plan_suspended(conn, tmpdb, monkeypatch):
    upload(conn, SIMPLE)
    plan_id = service.publish_plan(conn, 1, 1)["plan_id"]
    wire(tmpdb, monkeypatch, fault="none")
    res = service.start_delivery(conn, plan_id, "http://receiver")
    assert res["delivery"]["status"] == "accepted"
    train_ids = {r["slide_id"] for r in storage.plan_slides(conn, plan_id, "train")}
    state = service.current_restrictions(conn)
    assert train_ids <= state.accepted
    # plan had those slides in train, so it stays valid (test side unaffected)
    assert storage.get_plan(conn, plan_id)["status"] == "valid"
    # any later plan must not test those slides
    for _ in range(5):
        try:
            pid = service.publish_plan(conn, 1, 1)["plan_id"]
        except allocator.Unsolvable:
            break
        m = storage.plan_membership(conn, pid)
        assert all(m[sid] != "test" for sid in train_ids)


def test_rejected_lifts_only_its_own_restriction(conn, tmpdb, monkeypatch):
    upload(conn, SIMPLE)
    plan_id = service.publish_plan(conn, 1, 1)["plan_id"]
    wire(tmpdb, monkeypatch, refuse=True)
    res = service.start_delivery(conn, plan_id, "http://receiver")
    assert res["delivery"]["status"] == "rejected"
    state = service.current_restrictions(conn)
    train_ids = {r["slide_id"] for r in storage.plan_slides(conn, plan_id, "train")}
    # rejected => not accepted, not pending => no restriction at all
    assert not (train_ids & state.accepted)
    assert not (train_ids & state.uncertain)
    # and the plan is still downloadable
    assert storage.get_plan(conn, plan_id)["status"] == "valid"


def test_late_upload_suspends_plan_and_restores(conn):
    upload(conn, SIMPLE)
    plan_id = service.publish_plan(conn, 1, 1)["plan_id"]
    membership = storage.plan_membership(conn, plan_id)
    test_slide = next(s for s, m in membership.items() if m == "test")
    # find its patient, then append a new slide joining the same test group
    with storage.transaction(conn):
        current = {s["slide_id"]: s for s in storage.all_slides(conn)}
    patient = current[test_slide]["patient_id"]
    group = current[test_slide]["repeat_group"]
    new_row = (
        f"slide_id,patient_id,repeat_group,malignancy\n"
        f"S99,{patient},{group or ''},benign\n"
    )
    upload(conn, new_row)
    plan = storage.get_plan(conn, plan_id)
    assert plan["status"] == "suspended"
    assert "连通组" in plan["status_reason"]
    with pytest.raises(service.PlanNotDeliverable):
        service.plan_download(conn, plan_id, "train")


def test_capacity_limit_is_atomic(conn):
    rows = [
        {"slide_id": f"X{i}", "patient_id": f"P{i}", "repeat_group": None, "malignancy": "benign"}
        for i in range(101)
    ]
    with pytest.raises(storage.CapacityError):
        service.append_slides(conn, rows)
    assert storage.count_slides(conn) == 0
    service.append_slides(conn, rows[:100])
    assert storage.count_slides(conn) == 100
    with pytest.raises(storage.CapacityError):
        service.append_slides(
            conn,
            [{"slide_id": "Y1", "patient_id": "PY", "repeat_group": None, "malignancy": "benign"}],
        )
    assert storage.count_slides(conn) == 100


def test_duplicate_slide_ids_are_skipped_not_overwritten(conn):
    upload(conn, SIMPLE)
    res = upload(conn, "slide_id,patient_id,repeat_group,malignancy\nS1,P-X,,malignant\nS7,P7,,benign\n")
    assert res["inserted"] == 1
    assert res["skipped_duplicates"] == ["S1"]
    by_id = {s["slide_id"]: s for s in storage.all_slides(conn)}
    assert by_id["S1"]["patient_id"] == "P1"


def test_stale_computation_not_published(conn, monkeypatch):
    upload(conn, SIMPLE)
    real_solve = allocator.solve

    def racing_solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        with storage.transaction(conn):
            storage.insert_slides(
                conn,
                [
                    {
                        "slide_id": "RACE",
                        "patient_id": "PRACE",
                        "repeat_group": None,
                        "malignancy": "benign",
                    }
                ],
            )
        return result

    monkeypatch.setattr(allocator, "solve", racing_solve)
    monkeypatch.setattr(service.allocator, "solve", racing_solve)
    with pytest.raises(service.StaleComputation):
        service.publish_plan(conn, 1, 1)
    assert storage.list_plans(conn) == []


def test_csv_validation_rejects_batch_atomically(conn):
    bad = "slide_id,patient_id,repeat_group,malignancy\nS1,P1,,benign\nS2,P2,,??\n"
    with pytest.raises(csvio.CsvError):
        upload(conn, bad)
    assert storage.count_slides(conn) == 0


def test_web_upload_and_download(client, conn):
    resp = client.post(
        "/upload",
        data={"text": SIMPLE, "has_header": "1"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert storage.count_slides(conn) == 6
    resp = client.post(
        "/plans",
        data={"benign_count": "1", "malignant_count": "1"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "error=" not in resp.headers["location"]
    plan_id = storage.list_plans(conn)[0]["id"]
    csv_resp = client.get(f"/plans/{plan_id}/download/train")
    assert csv_resp.status_code == 200
    assert "slide_id,patient_id,repeat_group,malignancy" in csv_resp.text
    assert len(csv_resp.text.strip().splitlines()) == 5  # header + 4 train rows
    assert client.get(f"/plans/{plan_id}/download/test").status_code == 200


def test_receiver_rejects_checksum_mismatch():
    tmp = tempfile.mkdtemp(prefix="recv-")
    import receiver.main as rm

    old = rm.DB_PATH
    rm.DB_PATH = os.path.join(tmp, "r.sqlite3")
    try:
        client = TestClient(receiver_app)
        resp = client.post(
            "/api/deliveries",
            json={
                "delivery_id": "D1",
                "checksum": "bogus",
                "slides": [
                    {
                        "slide_id": "S1",
                        "patient_id": "P1",
                        "repeat_group": "",
                        "malignancy": "benign",
                        "split": "train",
                    }
                ],
            },
        )
        assert resp.status_code == 409
        assert client.get("/api/deliveries/D1").status_code == 404
    finally:
        rm.DB_PATH = old


def test_receiver_terminal_state_is_immutable():
    tmp = tempfile.mkdtemp(prefix="recv-")
    import receiver.main as rm

    old = rm.DB_PATH
    rm.DB_PATH = os.path.join(tmp, "r.sqlite3")
    try:
        client = TestClient(receiver_app)
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
        first = client.post(
            "/api/deliveries",
            json={"delivery_id": "D1", "checksum": checksum, "slides": slides},
        )
        assert first.status_code == 201
        receipt = first.json()["receipt_ref"]
        client.post("/api/simulate", json={"refuse": "1"})
        second = client.post(
            "/api/deliveries",
            json={"delivery_id": "D1", "checksum": checksum, "slides": slides},
        )
        assert second.status_code == 200
        assert second.json()["receipt_ref"] == receipt
        assert second.json()["status"] == "accepted"
        assert client.post("/api/deliveries/D1/reject").status_code == 409
    finally:
        rm.DB_PATH = old


def test_pending_delivery_suspends_other_plan_and_rejection_restores(conn, tmpdb, monkeypatch):
    """A delivery touching another plan's test slide must suspend that plan;
    a terminal rejection lifts exactly that restriction and restores it."""
    upload(conn, SIMPLE)
    plan1 = service.publish_plan(conn, 1, 1)["plan_id"]
    m1 = storage.plan_membership(conn, plan1)
    test_slide = next(s for s, split in m1.items() if split == "test")

    # hand-register a pending delivery covering the *test* slide of plan1
    with storage.transaction(conn):
        storage.insert_delivery(
            conn,
            delivery_id="DLV-OTHER",
            plan_id=plan1,
            split="train",
            checksum="c1",
            attempt=1,
            rows=[
                {
                    "slide_id": test_slide,
                    "patient_id": next(
                        s["patient_id"] for s in storage.all_slides(conn) if s["slide_id"] == test_slide
                    ),
                    "repeat_group": next(
                        (s["repeat_group"] for s in storage.all_slides(conn) if s["slide_id"] == test_slide),
                        None,
                    ),
                    "malignancy": "benign",
                }
            ],
        )
        service.refresh_plans(conn)
    plan = storage.get_plan(conn, plan1)
    assert plan["status"] == "suspended"
    assert test_slide in plan["status_reason"]
    with pytest.raises(service.PlanNotDeliverable):
        service.plan_download(conn, plan1, "test")

    # terminal rejection lifts that single restriction -> plan is valid again
    with storage.transaction(conn):
        storage.set_delivery_terminal(
            conn,
            "DLV-OTHER",
            status="rejected",
            receiver_status="rejected",
            receipt_ref="RCPT-X",
            receipt_json="{}",
        )
        service.refresh_plans(conn)
    plan = storage.get_plan(conn, plan1)
    assert plan["status"] == "valid"
    text, _, _ = service.plan_download(conn, plan1, "test")
    assert test_slide in text


def test_broken_write_leaves_no_partial_plan(conn, monkeypatch):
    """If the membership insert fails, the plan row must roll back too."""
    upload(conn, SIMPLE)
    real_insert_plan = storage.insert_plan

    def failing_insert_plan(conn_, **kwargs):
        real_insert_plan(conn_, **kwargs)
        raise RuntimeError("simulated disk failure while writing membership")

    monkeypatch.setattr(storage, "insert_plan", failing_insert_plan)
    monkeypatch.setattr(service.storage, "insert_plan", failing_insert_plan)
    with pytest.raises(RuntimeError):
        service.publish_plan(conn, 1, 1)
    assert storage.list_plans(conn) == []
    assert conn.execute("SELECT COUNT(*) AS n FROM plan_slides").fetchone()["n"] == 0


def test_retry_after_missing_record_posts_matching_checksum(conn, tmpdb, monkeypatch):
    upload(conn, SIMPLE)
    plan_id = service.publish_plan(conn, 1, 1)["plan_id"]
    recv, _, _ = wire(tmpdb, monkeypatch, fault="http404")
    res = service.start_delivery(conn, plan_id, "http://receiver")
    pid = res["delivery"]["id"]
    assert recv.get("/api/deliveries").json() == []
    recv.post("/api/simulate", json={"fault": "none"})
    retried = service.retry_delivery(conn, pid, "http://receiver")
    assert retried["delivery"]["status"] == "accepted"
    stored = recv.get(f"/api/deliveries/{pid}").json()
    assert stored["checksum"] == retried["delivery"]["checksum"]
    assert stored["slide_count"] == retried["delivery"]["slide_count"]
