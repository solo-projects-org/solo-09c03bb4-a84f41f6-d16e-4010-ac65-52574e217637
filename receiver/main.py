"""Independent receiving service for delivered training lists.

Kept in a separate process and a separate SQLite file/volume so that
"delivered" is an external, independently verifiable fact.

Guarantees
----------
* a delivery id is bound to exactly one frozen content checksum; re-posting
  the same id and checksum returns the original receipt (idempotent), so a
  retry after a timeout can never create a second acceptance;
* terminal state (accepted / rejected) is written once and never changed,
  including by retries or by an operator API;
* every delivery is queryable by id with its frozen slide list;
* the service has a persistent instance identity (``receiver_id``), generated
  on first start and stored in its database: it survives restarts but changes
  when the data volume is replaced, so the sender can tell the original
  receiver apart from a substitute. It is exposed at ``GET /api/identity``
  and embedded in every delivery response;
* simulation endpoints can force refusal and inject faults (HTTP 500 / 404,
  delays / timeouts) so the sender's "unknown" path can be exercised.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

DB_PATH = os.environ.get("RECEIVER_DB_PATH", "/data/receiver.sqlite3")
DEFAULT_DECISION = os.environ.get("RECEIVER_DECISION", "accept")
DEFAULT_FAULT = os.environ.get("RECEIVER_FAULT", "none")
DEFAULT_DELAY = float(os.environ.get("RECEIVER_FAULT_DELAY", "3.0"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    slides_json TEXT NOT NULL,
    slide_count INTEGER NOT NULL,
    receipt_ref TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('accepted', 'rejected')),
    comment TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS controls (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    delivery_id TEXT,
    note TEXT
);
"""

VALID_FAULTS = {
    "none",
    "http500",
    "http404",
    "http_verify500",
    "timeout",
    "slow",
    "drop_after_store",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO controls(key, value) VALUES ('fault', ?)", (DEFAULT_FAULT,)
    )
    conn.execute(
        "INSERT OR IGNORE INTO controls(key, value) VALUES ('delay', ?)", (str(DEFAULT_DELAY),)
    )
    conn.execute(
        "INSERT OR IGNORE INTO controls(key, value) VALUES ('decision', ?)", (DEFAULT_DECISION,)
    )
    conn.execute(
        "INSERT OR IGNORE INTO controls(key, value) VALUES ('refuse', '0')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO controls(key, value) VALUES ('receiver_id', ?)",
        ("RCV-" + secrets.token_hex(8).upper(),),
    )
    return conn


def receiver_id() -> str:
    """Persistent instance identity of this receiver (and its data volume)."""
    return controls().get("receiver_id", "")


def controls() -> dict[str, str]:
    conn = connect()
    try:
        return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM controls")}
    finally:
        conn.close()


def log_request(method: str, path: str, delivery_id: str | None, note: str) -> None:
    try:
        conn = connect()
        try:
            conn.execute(
                "INSERT INTO request_log(created_at, method, path, delivery_id, note) "
                "VALUES (?,?,?,?,?)",
                (now(), method, path, delivery_id, note),
            )
        finally:
            conn.close()
    except Exception:
        pass


def checksum_of(slides: list[dict]) -> str:
    lines = sorted(
        "|".join(
            [
                str(s.get("slide_id", "")),
                str(s.get("patient_id", "")),
                str(s.get("repeat_group") or ""),
                str(s.get("malignancy", "")),
                str(s.get("split", "")),
            ]
        )
        for s in slides
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@asynccontextmanager
async def lifespan(_: FastAPI):
    connect().close()
    yield


app = FastAPI(title="独立接收服务", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/identity")
def identity():
    """Persistent instance identity; lets the sender pin deliveries to us."""
    return {
        "service": "pathology-receiver",
        "receiver_id": receiver_id(),
        "time": now(),
    }


def _public(row: sqlite3.Row | dict, include_slides: bool = True) -> dict:
    data = dict(row)
    out = {
        "delivery_id": data["delivery_id"],
        "checksum": data["checksum"],
        "status": data["status"],
        "receipt_ref": data["receipt_ref"],
        "comment": data["comment"],
        "created_at": data["created_at"],
        "slide_count": data["slide_count"],
        "receiver_id": receiver_id(),
    }
    if include_slides:
        out["slides"] = json.loads(data["slides_json"])
    return out


@app.get("/api/deliveries/{delivery_id}")
def get_delivery(delivery_id: str):
    cfg = controls()
    if cfg.get("fault") == "http_verify500":
        return PlainTextResponse("模拟核实接口故障", status_code=500)
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM deliveries WHERE delivery_id=?", (delivery_id,)
        ).fetchone()
    finally:
        conn.close()
    log_request("GET", f"/api/deliveries/{delivery_id}", delivery_id, "verify")
    if not row:
        raise HTTPException(status_code=404, detail="查无记录")
    return _public(row)


@app.post("/api/deliveries")
async def post_delivery(request: Request):
    cfg = controls()
    fault = cfg.get("fault", "none")
    delay = float(cfg.get("delay", "0") or 0)
    if fault == "slow":
        time.sleep(delay)
    if fault == "http500":
        log_request("POST", "/api/deliveries", None, "injected 500")
        return PlainTextResponse("模拟接收服务内部错误", status_code=500)
    if fault == "http404":
        log_request("POST", "/api/deliveries", None, "injected 404")
        return JSONResponse({"detail": "模拟查无记录"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "请求体不是合法 JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"detail": "请求体结构错误"}, status_code=400)
    delivery_id = str(body.get("delivery_id") or "").strip()
    checksum = str(body.get("checksum") or "").strip()
    slides = body.get("slides")
    if not delivery_id or not checksum or not isinstance(slides, list) or not slides:
        return JSONResponse(
            {"detail": "缺少 delivery_id / checksum / slides"}, status_code=422
        )
    if len(delivery_id) > 80:
        return JSONResponse({"detail": "delivery_id 过长"}, status_code=422)
    computed = checksum_of(slides)
    if computed != checksum:
        return JSONResponse(
            {"detail": "校验和不匹配，请求被拒", "computed": computed}, status_code=409
        )

    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM deliveries WHERE delivery_id=?", (delivery_id,)
        ).fetchone()
        if existing:
            conn.execute("COMMIT")
            if existing["checksum"] != checksum:
                log_request("POST", "/api/deliveries", delivery_id, "id conflict")
                return JSONResponse(
                    {
                        "detail": "该交付号已绑定不同内容，拒绝覆盖",
                        "receipt_ref": existing["receipt_ref"],
                        "status": existing["status"],
                    },
                    status_code=409,
                )
            log_request("POST", "/api/deliveries", delivery_id, "idempotent replay")
            return JSONResponse(_public(existing))
        status = (
            "rejected"
            if cfg.get("refuse") == "1" or cfg.get("decision") == "reject"
            else "accepted"
        )
        comment = None
        if status == "rejected":
            comment = cfg.get("comment") or "接收服务处于模拟拒收模式"
        receipt_ref = "RCPT-" + secrets.token_hex(8).upper()
        conn.execute(
            "INSERT INTO deliveries(delivery_id, checksum, slides_json, slide_count, "
            "receipt_ref, status, comment, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                delivery_id,
                checksum,
                json.dumps(slides, ensure_ascii=False),
                len(slides),
                receipt_ref,
                status,
                comment,
                now(),
            ),
        )
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        log_request("POST", "/api/deliveries", delivery_id, "store failed")
        return JSONResponse({"detail": "接收服务写入失败"}, status_code=500)
    finally:
        conn.close()

    log_request("POST", "/api/deliveries", delivery_id, f"stored {status}")
    if fault == "timeout":
        time.sleep(delay)
        return JSONResponse(
            {"detail": "模拟响应超时（服务端已写入，响应丢失）"}, status_code=504
        )
    if fault == "drop_after_store":
        # The record exists but the response is lost, exactly like a reset.
        return JSONResponse({"detail": "模拟连接中断（服务端已写入）"}, status_code=502)
    return JSONResponse(
        {
            "delivery_id": delivery_id,
            "checksum": checksum,
            "status": status,
            "receipt_ref": receipt_ref,
            "comment": comment,
            "created_at": now(),
            "receiver_id": receiver_id(),
        },
        status_code=201,
    )


@app.get("/api/deliveries")
def list_deliveries():
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT delivery_id, checksum, slide_count, status, receipt_ref, comment, created_at "
            "FROM deliveries ORDER BY created_at DESC, delivery_id"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


@app.get("/api/log")
def request_log():
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT * FROM request_log ORDER BY id DESC LIMIT 200"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


@app.post("/api/simulate")
async def simulate(request: Request):
    """Controls for demos/tests: refuse mode and fault injection."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if body.get("fault") is not None and body["fault"] not in VALID_FAULTS:
        return JSONResponse(
            {"detail": f"未知故障类型；可用 {sorted(VALID_FAULTS)}"}, status_code=422
        )
    await simulate_dict(body)
    return {"ok": True, "controls": controls()}


@app.post("/api/deliveries/{delivery_id}/reject")
def reject_after_the_fact(delivery_id: str):
    """Terminal states are immutable; refuse to flip a stored delivery."""
    return JSONResponse(
        {
            "detail": "终态不可变更。如需拒收，请在接收服务上启用 refuse 模式并新建交付。",
            "delivery_id": delivery_id,
        },
        status_code=409,
    )


@app.get("/", response_class=HTMLResponse)
def home():
    conn = connect()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT delivery_id, checksum, slide_count, status, receipt_ref, comment, created_at "
            "FROM deliveries ORDER BY created_at DESC, delivery_id LIMIT 100"
        )]
        log = [dict(r) for r in conn.execute(
            "SELECT * FROM request_log ORDER BY id DESC LIMIT 40"
        )]
    finally:
        conn.close()
    cfg = controls()
    items = "".join(
        f"<tr><td><code>{r['delivery_id']}</code></td><td>{r['slide_count']}</td>"
        f"<td class=\"{r['status']}\">{'接纳' if r['status'] == 'accepted' else '拒收'}</td>"
        f"<td><code>{r['receipt_ref']}</code></td><td><small>{r['created_at']}</small></td></tr>"
        for r in rows
    ) or "<tr><td colspan=5>暂无交付</td></tr>"
    log_items = "".join(
        f"<li><small>{r['created_at']}</small> {r['method']} {r['path']} "
        f"{r['delivery_id'] or ''} — {r['note']}</li>"
        for r in log
    ) or "<li>暂无请求</li>"
    fault_options = "".join(
        f"<option value='{f}' {'selected' if f == cfg.get('fault') else ''}>{f}</option>"
        for f in sorted(VALID_FAULTS)
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>独立接收服务</title><style>
body{{font-family:system-ui,"Noto Sans SC",sans-serif;margin:32px;color:#1d2330;line-height:1.5}}
table{{border-collapse:collapse;font-size:13px;margin-bottom:20px}}
td,th{{border-bottom:1px solid #e3e6ec;padding:6px 10px;text-align:left;vertical-align:top}}
.accepted{{color:#146c3c}}.rejected{{color:#9c2020}}
code{{font-size:11px;background:#f3f4f7;padding:1px 4px;border-radius:4px}}
form{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:13px;margin-bottom:18px}}
button{{padding:4px 12px;border-radius:6px;border:1px solid #1f5eff;background:#1f5eff;color:#fff;cursor:pointer}}
ul{{font-size:12px;color:#3a4353}}
</style></head><body>
<h1>独立接收服务</h1>
<p>终态接纳 / 拒收一经写入不可更改；同一交付号重复提交仅返回原回执，不重复接纳。</p>
<p>本接收服务身份：<code>{cfg.get('receiver_id')}</code>（随数据卷持久保存；更换数据卷即更换身份）。</p>
<p>当前控制项：故障注入 <b>{cfg.get('fault')}</b>，延迟 {cfg.get('delay')}s，
拒收模式 <b>{'开' if cfg.get('refuse') == '1' else '关'}</b>。</p>
<form method="post" action="/simulate-form">
  <label>故障 <select name="fault">{fault_options}</select></label>
  <label>延迟(秒) <input name="delay" value="{cfg.get('delay')}" size="4"></label>
  <label><input type="checkbox" name="refuse" value="1" {'checked' if cfg.get('refuse') == '1' else ''}> 模拟拒收</label>
  <button type="submit">应用</button>
</form>
<p><small>说明：<code>timeout</code> / <code>drop_after_store</code> 表示服务端已接纳但响应丢失，
发送方应保持“状态未明”，随后可核实到同一条回执。</small></p>
<h2>交付记录</h2>
<table><thead><tr><th>交付号</th><th>张数</th><th>终态</th><th>回执号</th><th>时间</th></tr></thead>
<tbody>{items}</tbody></table>
<h2>最近请求</h2><ul>{log_items}</ul>
</body></html>"""


@app.post("/simulate-form")
async def simulate_form(request: Request):
    from urllib.parse import parse_qs

    body = parse_qs((await request.body()).decode("utf-8", "replace"))
    payload = {
        "fault": (body.get("fault") or ["none"])[0],
        "delay": (body.get("delay") or ["0"])[0],
        "refuse": "1" if (body.get("refuse") or [""])[0] == "1" else "0",
    }
    await simulate_dict(payload)
    from fastapi.responses import RedirectResponse

    return RedirectResponse("/", status_code=303)


async def simulate_dict(payload: dict) -> None:
    conn = connect()
    try:
        for key in ("fault", "delay", "refuse", "comment", "decision"):
            if key in payload and payload[key] is not None:
                conn.execute(
                    "INSERT INTO controls(key, value) VALUES (?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, str(payload[key])),
                )
    finally:
        conn.close()
