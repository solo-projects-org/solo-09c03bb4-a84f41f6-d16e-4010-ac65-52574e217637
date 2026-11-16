from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import allocator, csvio, service, storage

APP_DB = os.environ.get("APP_DB_PATH", "/data/pathology.sqlite3")
RECEIVER_URL = os.environ.get("RECEIVER_URL", "http://receiver:8000")
RECONCILE_ON_START = os.environ.get("RECONCILE_ON_START", "1") == "1"


def _db_path() -> str:
    # resolved per request so tests / Compose overrides take effect
    return os.environ.get("APP_DB_PATH", APP_DB)


@asynccontextmanager
async def lifespan(_: FastAPI):
    conn = storage.connect(_db_path())
    try:
        if RECONCILE_ON_START:
            service.startup_reconcile(conn, _receiver_url(conn), timeout=1.5)
    finally:
        conn.close()
    yield


app = FastAPI(title="病理切片名单划分", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


def _receiver_url(conn) -> str:
    settings = storage.get_settings(conn)
    return settings.get("receiver_url") or RECEIVER_URL


def db(request: Request):
    conn = storage.connect(_db_path())
    try:
        yield conn
    finally:
        conn.close()


Db = Depends(db)


def _templates():
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["short"] = lambda v: (str(v)[:12] + "…") if v and len(str(v)) > 14 else (v or "")
    return env


templates = _templates()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, conn=Db):
    slides = storage.all_slides(conn)
    plans = service.deliverable_plan_summary(conn)
    deliveries = storage.list_deliveries(conn)
    restrictions = service.current_restrictions(conn)
    settings = storage.get_settings(conn)
    events = storage.list_events(conn, 25)
    groups = allocator.build_groups(slides)
    benign = sum(1 for s in slides if s["malignancy"] == "benign")
    malignant = len(slides) - benign
    eligible_benign = sum(
        1
        for s in slides
        if s["slide_id"] not in restrictions.barred and s["malignancy"] == "benign"
    )
    eligible_malignant = sum(
        1
        for s in slides
        if s["slide_id"] not in restrictions.barred and s["malignancy"] == "malignant"
    )
    html = templates.get_template("index.html").render(
        request=request,
        slides=slides,
        plans=plans,
        deliveries=deliveries,
        settings=settings,
        events=events,
        groups=groups,
        benign=benign,
        malignant=malignant,
        eligible_benign=eligible_benign,
        eligible_malignant=eligible_malignant,
        restriction_reasons={
            sid: restrictions.explain(sid) for sid in restrictions.barred
        },
        barred_count=len(restrictions.barred),
        max_slides=storage.MAX_SLIDES,
        flash=request.query_params.get("flash"),
        error=request.query_params.get("error"),
    )
    return HTMLResponse(html)


def _redirect(flash: str | None = None, error: str | None = None) -> RedirectResponse:
    from urllib.parse import quote

    params = []
    if flash:
        params.append("flash=" + quote(flash))
    if error:
        params.append("error=" + quote(error))
    return RedirectResponse("/" + ("?" + "&".join(params) if params else ""), status_code=303)


@app.post("/upload")
async def upload(
    request: Request,
    conn=Db,
    file: UploadFile | None = File(None),
    text: str = Form(""),
    has_header: str = Form("1"),
):
    raw = ""
    if file is not None and file.filename:
        data = await file.read()
        try:
            raw = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                raw = data.decode("gb18030")
            except UnicodeDecodeError:
                return _redirect(error="文件编码无法识别，请保存为 UTF-8 后重试。")
    elif text.strip():
        raw = text
    else:
        return _redirect(error="请选择 CSV 文件或粘贴 CSV 内容。")
    try:
        rows = csvio.parse_csv_text(raw, assumed_header=has_header == "1")
        result = service.append_slides(conn, [r.__dict__ for r in rows])
    except (csvio.CsvError, storage.CapacityError, service.ServiceError) as exc:
        return _redirect(error=str(exc))
    msg = f"已追加 {result['inserted']} 条切片，库中现有 {result['total']} 条。"
    if result["skipped_duplicates"]:
        msg += " 跳过重复切片号：" + "、".join(result["skipped_duplicates"][:12])
    return _redirect(flash=msg)


@app.post("/settings")
def update_settings(conn=Db, receiver_url: str = Form(...)):
    with storage.transaction(conn):
        storage.set_setting(conn, "receiver_url", receiver_url.strip())
        storage.log_event(conn, "settings", f"接收服务地址设为 {receiver_url.strip()}。")
    return _redirect(flash="接收服务地址已保存。")


@app.post("/plans")
def create_plan(conn=Db, benign_count: int = Form(...), malignant_count: int = Form(...)):
    try:
        result = service.publish_plan(conn, benign_count, malignant_count)
    except allocator.Unsolvable as exc:
        return _redirect(error=exc.reason)
    except service.StaleComputation as exc:
        return _redirect(error=str(exc))
    r = result["result"]
    return _redirect(
        flash=(
            f"已发布名单 #{result['plan_id']}：测试 {r.test_benign} 良 + {r.test_malignant} 恶 = "
            f"{r.test_benign + r.test_malignant} 张；训练 {r.train_benign + r.train_malignant} 张；"
            f"共 {len(r.groups)} 个连通组。校验和 {r.checksum[:12]}…"
        )
    )


@app.get("/plans/{plan_id}/download/{split}")
def download(conn=Db, plan_id: int = 0, split: str = "train"):
    try:
        text, filename, checksum = service.plan_download(conn, plan_id, split)
    except service.PlanNotDeliverable as exc:
        from urllib.parse import quote

        return PlainTextResponse(
            str(exc),
            status_code=409,
            headers={"X-Reason": quote(exc.args[0][:200], safe="")},
        )
    except service.ServiceError as exc:
        return PlainTextResponse(str(exc), status_code=404)
    return PlainTextResponse(
        text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Checksum-Sha256": checksum,
        },
    )


@app.post("/plans/{plan_id}/deliver")
def deliver(conn=Db, plan_id: int = 0):
    try:
        res = service.start_delivery(conn, plan_id, _receiver_url(conn))
    except service.PlanNotDeliverable as exc:
        return _redirect(error=str(exc))
    except service.ServiceError as exc:
        return _redirect(error=str(exc))
    return _redirect(flash=res["message"])


@app.post("/deliveries/{delivery_id}/verify")
def verify(conn=Db, delivery_id: str = ""):
    try:
        res = service.verify_delivery(conn, delivery_id, _receiver_url(conn))
    except service.ServiceError as exc:
        return _redirect(error=str(exc))
    return _redirect(flash=res["message"])


@app.post("/deliveries/{delivery_id}/retry")
def retry(conn=Db, delivery_id: str = ""):
    try:
        res = service.retry_delivery(conn, delivery_id, _receiver_url(conn))
    except service.ServiceError as exc:
        return _redirect(error=str(exc))
    return _redirect(flash=res["message"])


# ------------------------------------------------------------------ JSON API


@app.get("/api/library")
def api_library(conn=Db):
    return {
        "count": storage.count_slides(conn),
        "max": storage.MAX_SLIDES,
        "slides": storage.all_slides(conn),
    }


@app.get("/api/restrictions")
def api_restrictions(conn=Db):
    state = service.current_restrictions(conn)
    return {
        "accepted": sorted(state.accepted),
        "uncertain": sorted(state.uncertain),
        "barred": sorted(state.barred),
        "reasons": {
            sid: {"code": r.code, "detail": r.detail}
            for sid, r in state.reasons.items()
        },
    }


@app.get("/api/plans")
def api_plans(conn=Db):
    return service.deliverable_plan_summary(conn)


@app.get("/api/deliveries")
def api_deliveries(conn=Db):
    return storage.list_deliveries(conn)


@app.get("/api/deliveries/{delivery_id}")
def api_delivery(conn=Db, delivery_id: str = ""):
    d = storage.get_delivery(conn, delivery_id)
    if not d:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {**d, "slides": storage.delivery_rows(conn, delivery_id)}


@app.get("/api/events")
def api_events(conn=Db):
    return storage.list_events(conn, 100)
