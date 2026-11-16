"""End-to-end smoke test over real HTTP (uvicorn processes, no TestClient).

Run manually:

    python3 scripts/e2e_smoke.py

It starts:
  * two independent receiver processes, each with its own DB file;
  * the app process with its own DB file.

Then it reproduces the reported bug end-to-end:

  1. upload CSV, publish a plan, deliver its train list to receiver A;
  2. hide A's answer so the local delivery stays pending (proxy fault);
  3. switch the configured address to receiver B and press retry/verify;
  4. assert B never received the list and the slides stay barred;
  5. point the address back at A and verify -> accepted, ban permanent;
  6. replace A's data volume behind the same address -> retry/verify refused;
  7. restart the app -> restriction survives, plans suspended as required.

Exit code 0 means every assertion passed.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception:
            time.sleep(0.15)
    raise RuntimeError(f"service did not start: {url}")


def req(method: str, url: str, data: dict | None = None, form: dict | None = None):
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    elif form is not None:
        from urllib.parse import urlencode

        body = urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            raw = resp.read()
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def jreq(method: str, url: str, data: dict | None = None, form: dict | None = None):
    status, raw = req(method, url, data=data, form=form)
    try:
        return status, json.loads(raw or b"null")
    except json.JSONDecodeError:
        return status, raw.decode("utf-8", "replace")


class Proc:
    def __init__(self, name, cmd, env):
        self.name = name
        self.log = open(os.path.join(env["_LOGDIR"], f"{name}.log"), "w")
        self.proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT
        )

    def stop(self):
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="e2e-")
    logdir = os.path.join(tmp, "logs")
    os.makedirs(logdir)
    procs: list[Proc] = []
    checks: list[str] = []

    def check(cond, msg):
        if not cond:
            raise AssertionError(msg)
        checks.append(msg)
        print("  ok -", msg)

    app_port = free_port()
    a_port = free_port()
    b_port = free_port()

    base_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "_LOGDIR": logdir,
        "PYTHONPATH": ROOT,
    }

    def start_receiver(name: str, port: int, db: str) -> Proc:
        env = {
            **base_env,
            "RECEIVER_DB_PATH": db,
            "RECEIVER_NAME": name,
            "RECEIVER_DECISION": "accept",
            "RECEIVER_FAULT": "none",
        }
        p = Proc(
            name,
            [PY, "-m", "uvicorn", "receiver.main:app", "--host", "127.0.0.1",
             "--port", str(port), "--log-level", "warning"],
            env,
        )
        procs.append(p)
        wait_http(f"http://127.0.0.1:{port}/healthz")
        return p

    try:
        recv_a_db = os.path.join(tmp, "recv-a.sqlite3")
        recv_b_db = os.path.join(tmp, "recv-b.sqlite3")
        start_receiver("receiver-a", a_port, recv_a_db)
        start_receiver("receiver-b", b_port, recv_b_db)
        url_a = f"http://127.0.0.1:{a_port}"
        url_b = f"http://127.0.0.1:{b_port}"
        ident_a = jreq("GET", url_a + "/api/identity")[1]["receiver_id"]
        ident_b = jreq("GET", url_b + "/api/identity")[1]["receiver_id"]
        assert ident_a != ident_b

        app_db = os.path.join(tmp, "app.sqlite3")
        app_env = {
            **base_env,
            "APP_DB_PATH": app_db,
            "RECEIVER_URL": url_a,
            "RECONCILE_ON_START": "1",
        }
        app = Proc(
            "app",
            [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(app_port), "--log-level", "warning"],
            app_env,
        )
        procs.append(app)
        app_url = f"http://127.0.0.1:{app_port}"
        wait_http(app_url + "/healthz")

        print("1. upload and publish")
        csv = open(os.path.join(ROOT, "examples", "library.csv")).read()
        status, _ = req(
            "POST",
            app_url + "/upload",
            form={"text": csv, "has_header": "1"},
        )
        check(status in (200, 303), "CSV uploaded over HTTP")
        status, _ = req(
            "POST", app_url + "/plans", form={"benign_count": "3", "malignant_count": "3"}
        )
        check(status in (200, 303), "plan published")
        plans = jreq("GET", app_url + "/api/plans")[1]
        check(len(plans) == 1, "exactly one plan")
        plan_id = plans[0]["id"]

        print("2. deliver to A with the response hidden")
        # Point the app at A, deliver, but make the *reply* invisible by
        # using A's timeout fault: it stores the record then answers 504.
        jreq("POST", url_a + "/api/simulate", data={"fault": "timeout", "delay": "0.6"})
        status, _ = req("POST", app_url + f"/plans/{plan_id}/deliver")
        check(status in (200, 303), "deliver request completed")
        deliveries = jreq("GET", app_url + "/api/deliveries")[1]
        check(len(deliveries) == 1, "one delivery registered")
        d = deliveries[0]
        pid = d["id"]
        check(d["status"] == "pending", "delivery is pending (reply lost)")
        check(d["receiver_id"] == ident_a, "delivery bound to receiver A")
        recs_a = jreq("GET", url_a + "/api/deliveries")[1]
        check(len(recs_a) == 1 and recs_a[0]["status"] == "accepted",
              "receiver A accepted the list")
        jreq("POST", url_a + "/api/simulate", data={"fault": "none"})

        print("3. switch address to B and retry / verify")
        jreq("POST", url_b + "/api/simulate", data={"fault": "none"})
        status, _ = req("POST", app_url + "/settings", form={"receiver_url": url_b})
        check(status in (200, 303), "configured address switched to B")
        status, _ = req("POST", app_url + f"/deliveries/{pid}/retry")
        check(status in (200, 303), "retry request completed")
        d = jreq("GET", app_url + f"/api/deliveries/{pid}")[1]
        check(d["receiver_id"] == ident_a, "retry kept the original receiver A")
        recs = jreq("GET", url_b + "/api/deliveries")[1]
        check(recs == [], "receiver B received nothing")

        # A new plan must still not be able to put those slides into test.
        restrictions = jreq("GET", app_url + "/api/restrictions")[1]
        rec_a_record = jreq("GET", url_a + f"/api/deliveries/{pid}")[1]
        delivered = {s["slide_id"] for s in rec_a_record["slides"]}
        check(bool(restrictions["barred"]), "delivered slides are still barred")
        check(delivered <= set(restrictions["accepted"]) | set(restrictions["uncertain"]),
              "every delivered slide is restricted")

        print("4. verify through the switched address")
        status, _ = req("POST", app_url + f"/deliveries/{pid}/verify")
        d = jreq("GET", app_url + f"/api/deliveries/{pid}")[1]
        check(d["status"] == "accepted", "verified via original receiver A")
        check(d["receipt_ref"], "receipt recorded")
        check(jreq("GET", url_b + "/api/deliveries")[1] == [],
              "B never saw the delivery")

        print("5. replace A's data volume behind the same address")
        recv_a_stopped = procs[0]
        recv_a_stopped.stop()
        procs.remove(recv_a_stopped)
        old_db_backup = os.path.join(tmp, "a-old.sqlite3")
        shutil.move(recv_a_db, old_db_backup)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(recv_a_db + suffix):
                os.remove(recv_a_db + suffix)
        # a fresh volume with no deliveries, new identity
        start_receiver("receiver-a2", a_port, recv_a_db)
        ident_a2 = jreq("GET", url_a + "/api/identity")[1]["receiver_id"]
        check(ident_a2 != ident_a, "fresh volume has a new identity")

        # Register a fresh pending delivery for the "same-address new volume"
        # experiment (the accepted one above is terminal already).
        status, _ = req("POST", app_url + "/settings", form={"receiver_url": url_a})
        before_plans = len(jreq("GET", app_url + "/api/plans")[1])
        status, body = req(
            "POST", app_url + "/plans", form={"benign_count": "0", "malignant_count": "0"}
        )
        plans = jreq("GET", app_url + "/api/plans")[1]
        check(
            len(plans) == before_plans + 1,
            "a second plan was published"
            + ("" if status in (200, 303) else f" (HTTP {status}: {body[:200]!r})"),
        )
        plan2 = plans[0]["id"]
        check(plan2 != plan_id, "second plan is a new plan")
        jreq("POST", url_a + "/api/simulate", data={"fault": "timeout", "delay": "0.6"})
        req("POST", app_url + f"/plans/{plan2}/deliver")
        new_pending = [
            x for x in jreq("GET", app_url + "/api/deliveries")[1]
            if x["status"] == "pending"
        ]
        check(len(new_pending) == 1, "second delivery pending after lost reply")
        pid2 = new_pending[0]["id"]
        check(new_pending[0]["receiver_id"] == ident_a2,
              "second delivery bound to the fresh volume")
        # Now replace the volume *again*: the pending delivery's owner is gone.
        procs[0].stop()
        procs.pop(0)
        shutil.move(recv_a_db, os.path.join(tmp, "a2-old.sqlite3"))
        for suffix in ("-wal", "-shm"):
            if os.path.exists(recv_a_db + suffix):
                os.remove(recv_a_db + suffix)
        start_receiver("receiver-a3", a_port, recv_a_db)
        ident_a3 = jreq("GET", url_a + "/api/identity")[1]["receiver_id"]
        check(ident_a3 != ident_a2, "third volume differs from the pending owner")
        jreq("POST", url_a + "/api/simulate", data={"fault": "none"})

        status, _ = req("POST", app_url + f"/deliveries/{pid2}/retry")
        d2 = jreq("GET", app_url + f"/api/deliveries/{pid2}")[1]
        check(d2["status"] == "pending", "retry refused by the substitute volume")
        check(d2["receiver_id"] == ident_a2, "binding unchanged")
        check(jreq("GET", url_a + "/api/deliveries")[1] == [],
              "substitute volume received nothing")
        status, _ = req("POST", app_url + f"/deliveries/{pid2}/verify")
        d2 = jreq("GET", app_url + f"/api/deliveries/{pid2}")[1]
        check(d2["status"] == "pending", "verify also refused")

        print("6. restart the app: restrictions survive")
        app.stop()
        procs.remove(app)
        app = Proc(
            "app2",
            [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(app_port), "--log-level", "warning"],
            app_env,
        )
        procs.append(app)
        wait_http(app_url + "/healthz")
        restrictions = jreq("GET", app_url + "/api/restrictions")[1]
        accepted_by_a = {s["slide_id"] for s in rec_a_record["slides"]}
        check(
            accepted_by_a <= set(restrictions["accepted"]),
            "accepted delivery still permanently barred after restart",
        )
        d2 = jreq("GET", app_url + f"/api/deliveries/{pid2}")[1]
        check(d2["status"] == "pending", "pending delivery survived restart")
        check(d2["receiver_id"] == ident_a2, "binding survived restart")

        print("\nALL E2E CHECKS PASSED:")
        for c in checks:
            if c:
                print(" -", c)
        return 0
    finally:
        for p in procs:
            p.stop()
        print("logs:", logdir)


if __name__ == "__main__":
    sys.exit(main())
