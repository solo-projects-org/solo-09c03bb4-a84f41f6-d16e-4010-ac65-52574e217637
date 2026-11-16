"""Business operations shared by the HTML UI and the JSON API.

Transaction boundaries
----------------------
* ``publish_plan`` reads a consistent snapshot, solves outside the write lock,
  then verifies inside the write transaction that the library fingerprint and
  the delivery generation are unchanged. A concurrent upload or delivery
  transition therefore invalidates the computation ("过期计算不得发布"),
  and the plan plus its complete frozen membership are written in one
  transaction ("写入失败不留半份名单").
* ``start_delivery`` persists the pending record (and its frozen slide copy)
  *before* the first network call; a crash between the two leaves a verifiable
  pending delivery instead of a lost restriction.
* Retries of an unconfirmed delivery reuse the same delivery id and checksum,
  so the receiver's idempotency prevents a second acceptance.
* Restrictions are always derived from the delivery table; nothing is cached,
  so a restart cannot lose them and a rejection removes only its own entry.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from . import allocator, receiver_client, storage
from .allocator import RestrictionState, Unsolvable

DELIVERY_GENERATION_KEY = "delivery_generation"


class ServiceError(Exception):
    pass


class StaleComputation(ServiceError):
    pass


class PlanNotDeliverable(ServiceError):
    pass


class ConcurrentRegistration(ServiceError):
    pass


# ------------------------------------------------------------------ helpers


def _delivery_generation(conn: sqlite3.Connection) -> int:
    return int(storage.get_setting(conn, DELIVERY_GENERATION_KEY, "0") or "0")


def _bump_delivery_generation(conn: sqlite3.Connection) -> None:
    storage.set_setting(conn, DELIVERY_GENERATION_KEY, str(_delivery_generation(conn) + 1))


def restriction_sets(conn: sqlite3.Connection) -> tuple[set[str], set[str]]:
    """(accepted, uncertain) slide ids, derived from every delivery row."""
    accepted: set[str] = set()
    uncertain: set[str] = set()
    for d in conn.execute("SELECT id, status FROM deliveries"):
        ids = storage.delivery_slide_ids(conn, d["id"])
        if d["status"] == "accepted":
            accepted |= ids
        elif d["status"] == "pending":
            uncertain |= ids
    return accepted, uncertain


def current_restrictions(conn: sqlite3.Connection) -> RestrictionState:
    slides = storage.all_slides(conn)
    accepted, uncertain = restriction_sets(conn)
    return allocator.compute_restrictions(slides, accepted, uncertain)


def refresh_plans(conn: sqlite3.Connection) -> None:
    """Re-judge every stored plan against current library and deliveries.

    Caller must hold a write transaction. Plan membership is never modified;
    only the derived status / reason is updated. Suspended plans stop serving
    downloads, and are restored automatically once they pass again.
    """
    slides = storage.all_slides(conn)
    accepted, uncertain = restriction_sets(conn)
    restrictions = allocator.compute_restrictions(slides, accepted, uncertain)
    for plan in storage.list_plans(conn):
        membership = storage.plan_membership(conn, int(plan["id"]))
        frozen = storage.plan_slides(conn, int(plan["id"]))
        problems = allocator.evaluate_plan(membership, frozen, slides, restrictions)
        status = "suspended" if problems else "valid"
        reason = "；".join(problems) if problems else None
        if status != plan["status"] or (reason or "") != (plan["status_reason"] or ""):
            storage.update_plan_status(conn, int(plan["id"]), status, reason)
            if status == "suspended":
                storage.log_event(
                    conn,
                    "plan_suspended",
                    f"名单 #{plan['id']} 隔离校验失败，已停止下载：{reason}",
                )
            else:
                storage.log_event(
                    conn, "plan_restored", f"名单 #{plan['id']} 恢复合格，可再次下载。"
                )


def _plan_rows(conn: sqlite3.Connection, plan_id: int) -> list[dict[str, Any]]:
    return storage.plan_slides(conn, plan_id)


# ------------------------------------------------------------------ upload


def append_slides(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with storage.transaction(conn):
        inserted, duplicates, before = storage.insert_slides(conn, rows)
        if inserted:
            storage.log_event(
                conn, "upload", f"追加 {inserted} 条切片（库存 {before} → {before + inserted}）。"
            )
            refresh_plans(conn)
        total = storage.count_slides(conn)
    return {
        "inserted": inserted,
        "skipped_duplicates": [r["slide_id"] for r in duplicates],
        "total": total,
        "before": before,
    }


# ------------------------------------------------------------------- plans


def publish_plan(
    conn: sqlite3.Connection, req_benign: int, req_malignant: int
) -> dict[str, Any]:
    with storage.read_snapshot(conn):
        slides = storage.all_slides(conn)
        accepted, uncertain = restriction_sets(conn)
        generation = _delivery_generation(conn)
        snapshot_fingerprint = allocator.fingerprint(slides)

    try:
        result = allocator.solve(slides, accepted, uncertain, req_benign, req_malignant)
    except Unsolvable as exc:
        with storage.transaction(conn):
            storage.log_event(conn, "plan_unsolvable", f"无法划分：{exc.reason}")
        raise

    frozen_rows = [
        {
            "slide_id": s["slide_id"],
            "patient_id": s["patient_id"],
            "repeat_group": s.get("repeat_group") or None,
            "malignancy": s["malignancy"],
            "split": result.membership[s["slide_id"]],
        }
        for s in slides
    ]

    try:
        with storage.transaction(conn):
            current_fingerprint = allocator.fingerprint(storage.all_slides(conn))
            if (
                current_fingerprint != snapshot_fingerprint
                or _delivery_generation(conn) != generation
            ):
                raise StaleComputation(
                    "求解期间库内容或交付状态发生变化，本次结果已过期，未发布。请重新划分。"
                )
            plan_id = storage.insert_plan(
                conn,
                created_at=storage.now(),
                req_benign=req_benign,
                req_malignant=req_malignant,
                test_benign=result.test_benign,
                test_malignant=result.test_malignant,
                test_count=result.test_benign + result.test_malignant,
                train_count=len(slides) - result.test_benign - result.test_malignant,
                checksum=result.checksum,
                fingerprint=result.fingerprint,
                rows=frozen_rows,
            )
            storage.log_event(
                conn,
                "plan_created",
                f"发布名单 #{plan_id}：测试 {result.test_benign} 良性 + "
                f"{result.test_malignant} 恶性，训练 "
                f"{len(slides) - req_benign - req_malignant} 张，"
                f"校验和 {result.checksum[:12]}…。",
            )
            refresh_plans(conn)
    except StaleComputation:
        with storage.transaction(conn):
            storage.log_event(
                conn, "plan_stale", "求解期间状态变化，过期结果已丢弃（未发布）。"
            )
        raise

    return {
        "plan_id": plan_id,
        "result": result,
        "plan": storage.get_plan(conn, plan_id),
    }


def plan_or_raise(conn: sqlite3.Connection, plan_id: int) -> dict[str, Any]:
    plan = storage.get_plan(conn, plan_id)
    if not plan:
        raise ServiceError(f"名单 #{plan_id} 不存在。")
    return plan


def plan_download(
    conn: sqlite3.Connection, plan_id: int, split: str
) -> tuple[str, str, str]:
    """(csv text, filename, sha256). Re-judges and reads in one transaction."""
    if split not in {"train", "test"}:
        raise ServiceError("split 只能是 train 或 test。")
    from .csvio import render_csv

    with storage.transaction(conn):
        refresh_plans(conn)
        plan = plan_or_raise(conn, plan_id)
        if plan["status"] != "valid":
            raise PlanNotDeliverable(
                f"名单 #{plan_id} 当前隔离校验未通过，已停止提供下载。"
                f"原因：{plan['status_reason']}"
            )
        rows = storage.plan_slides(conn, plan_id, split)
    text = render_csv(rows, ["slide_id", "patient_id", "repeat_group", "malignancy"])
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    filename = f"plan-{plan_id}-{split}-{checksum[:8]}.csv"
    return text, filename, checksum


def deliverable_plan_summary(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    out = []
    for plan in storage.list_plans(conn):
        test = storage.plan_slides(conn, int(plan["id"]), "test")
        train = storage.plan_slides(conn, int(plan["id"]), "train")
        out.append(
            {
                **plan,
                "test_count": len(test),
                "train_count": len(train),
                "deliveries": storage.list_deliveries(conn, int(plan["id"])),
            }
        )
    return out


# -------------------------------------------------------------- deliveries


def _train_content(
    conn: sqlite3.Connection, plan_id: int
) -> tuple[list[dict[str, Any]], str]:
    rows = storage.plan_slides(conn, plan_id, "train")
    if not rows:
        raise ServiceError("训练名单为空，无需交付。")
    return rows, allocator.canonical_checksum(rows)


def _delivery_id(plan_id: int, checksum: str, attempt: int) -> str:
    digest = hashlib.sha256(f"v1|{plan_id}|{checksum}|{attempt}".encode()).hexdigest()
    return f"DLV-{digest[:20].upper()}"


def _latest_attempt(conn: sqlite3.Connection, plan_id: int, checksum: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM deliveries WHERE plan_id=? AND checksum=? "
        "ORDER BY attempt DESC LIMIT 1",
        (plan_id, checksum),
    ).fetchone()
    return dict(row) if row else None


def start_delivery(
    conn: sqlite3.Connection, plan_id: int, receiver_url: str, timeout: float = 10.0
) -> dict[str, Any]:
    """Register a pending verification record, then POST the frozen train list."""
    rows, checksum = _train_content(conn, plan_id)

    with storage.transaction(conn):
        refresh_plans(conn)
        plan = plan_or_raise(conn, plan_id)
        if plan["status"] != "valid":
            raise PlanNotDeliverable(
                f"名单 #{plan_id} 当前隔离校验未通过，禁止交付。原因：{plan['status_reason']}"
            )
        latest = _latest_attempt(conn, plan_id, checksum)
        if latest and latest["status"] == "accepted":
            return {
                "delivery": latest,
                "outcome": None,
                "reused": True,
                "message": f"该训练名单已被终态接纳（回执 {latest['receipt_ref']}），不会重复发送。",
            }
        if latest and latest["status"] == "pending":
            return {
                "delivery": latest,
                "outcome": None,
                "reused": True,
                "message": f"已存在待核实交付 {latest['id']}，请先核实或重试，未重复登记。",
            }
        attempt = (latest["attempt"] + 1) if latest else 1
        delivery_id = _delivery_id(plan_id, checksum, attempt)
        try:
            storage.insert_delivery(
                conn,
                delivery_id=delivery_id,
                plan_id=plan_id,
                split="train",
                checksum=checksum,
                attempt=attempt,
                rows=rows,
            )
        except sqlite3.IntegrityError:
            # A concurrent request registered the same attempt first. That
            # pending row is authoritative; never send a second copy.
            raise ConcurrentRegistration(
                "另一请求已登记同一交付，未重复发送，请刷新后查看。"
            )
        _bump_delivery_generation(conn)
        storage.log_event(
            conn,
            "delivery_pending",
            f"名单 #{plan_id} 训练名单（{len(rows)} 张）登记为待核实交付 {delivery_id}，"
            f"发送前状态已落库。",
        )
        refresh_plans(conn)

    outcome = receiver_client.send_delivery(
        receiver_url, delivery_id, checksum, rows, timeout=timeout
    )
    return _apply_outcome(conn, delivery_id, outcome, origin="send")


def verify_delivery(
    conn: sqlite3.Connection, delivery_id: str, receiver_url: str, timeout: float = 10.0
) -> dict[str, Any]:
    delivery = storage.get_delivery(conn, delivery_id)
    if not delivery:
        raise ServiceError(f"交付 {delivery_id} 不存在。")
    if delivery["status"] != "pending":
        return {
            "delivery": delivery,
            "outcome": None,
            "reused": True,
            "message": f"该交付已是终态 {delivery['status']}，不会因核实而改变。",
        }
    outcome = receiver_client.query_delivery(
        receiver_url, delivery_id, delivery["checksum"], timeout=timeout
    )
    return _apply_outcome(conn, delivery_id, outcome, origin="verify")


def retry_delivery(
    conn: sqlite3.Connection, delivery_id: str, receiver_url: str, timeout: float = 10.0
) -> dict[str, Any]:
    delivery = storage.get_delivery(conn, delivery_id)
    if not delivery:
        raise ServiceError(f"交付 {delivery_id} 不存在。")
    if delivery["status"] != "pending":
        return {
            "delivery": delivery,
            "outcome": None,
            "reused": True,
            "message": f"该交付已是终态 {delivery['status']}，未重复发送。",
        }
    rows = [
        {**r, "split": delivery["split"]}
        for r in storage.delivery_rows(conn, delivery_id)
    ]
    if not rows:
        raise ServiceError("交付的冻结内容缺失，无法重试。")
    outcome = receiver_client.send_delivery(
        receiver_url, delivery_id, delivery["checksum"], rows, timeout=timeout
    )
    return _apply_outcome(conn, delivery_id, outcome, origin="retry")


def _apply_outcome(
    conn: sqlite3.Connection,
    delivery_id: str,
    outcome: receiver_client.DeliveryOutcome,
    origin: str,
) -> dict[str, Any]:
    with storage.transaction(conn):
        delivery = storage.get_delivery(conn, delivery_id)
        if not delivery:
            raise ServiceError(f"交付 {delivery_id} 不存在。")
        if outcome.state in {"accepted", "rejected"} and delivery["status"] == "pending":
            payload = json.dumps(
                outcome.payload or {}, ensure_ascii=False, sort_keys=True
            )
            storage.set_delivery_terminal(
                conn,
                delivery_id,
                status=outcome.state,
                receiver_status=outcome.state,
                receipt_ref=outcome.receipt_ref,
                receipt_json=payload,
            )
            storage.store_receipt(conn, str(outcome.receipt_ref), delivery_id, payload)
            _bump_delivery_generation(conn)
            storage.log_event(
                conn,
                "delivery_terminal",
                f"交付 {delivery_id} 终态 {outcome.state}，回执 {outcome.receipt_ref}（{origin}）。",
            )
            refresh_plans(conn)
            message = f"交付 {delivery_id} 已确认终态：{_cn(outcome.state)}，回执 {outcome.receipt_ref}。"
        elif outcome.state in {"accepted", "rejected"}:
            storage.log_event(
                conn,
                "delivery_conflict",
                f"交付 {delivery_id} 本地已终态 {delivery['status']}，忽略接收服务返回 "
                f"{outcome.state}（{origin}）。",
            )
            message = (
                f"本地已记录终态 {_cn(delivery['status'])}；接收服务返回 {_cn(outcome.state)}，"
                "以首次终态为准，未覆盖。"
            )
        else:
            storage.touch_delivery(
                conn,
                delivery_id,
                receiver_status="unknown",
                error=outcome.error or "状态未明",
            )
            storage.log_event(
                conn,
                "delivery_unknown",
                f"交付 {delivery_id} 未明（{origin}）：{outcome.error}。相关切片继续暂禁入测试。",
            )
            message = (
                f"交付 {delivery_id} 状态未明：{outcome.error}。"
                "相关切片继续暂禁入测试，可稍后核实或重试。"
            )
        refreshed = storage.get_delivery(conn, delivery_id)
    return {
        "delivery": refreshed,
        "outcome": outcome,
        "reused": False,
        "message": message,
    }


def _cn(status: str) -> str:
    return {"accepted": "接纳", "rejected": "拒收", "pending": "待核实"}.get(status, status)


def startup_reconcile(
    conn: sqlite3.Connection, receiver_url: str, timeout: float = 1.5
) -> list[dict[str, Any]]:
    """Restart path: re-judge every plan, then re-query pending deliveries.

    Re-judging is unconditional because a crash may have happened after an
    upload or delivery change but before the plan status rows were refreshed.
    Re-querying never deletes a restriction: a missing record or timeout keeps
    the delivery pending/unknown.
    """
    with storage.transaction(conn):
        refresh_plans(conn)
    return reconcile_pending(conn, receiver_url, timeout=timeout)


def reconcile_pending(
    conn: sqlite3.Connection, receiver_url: str, timeout: float = 2.0
) -> list[dict[str, Any]]:
    """Only re-query pending deliveries; never drops a restriction."""
    results = []
    for d in storage.list_deliveries(conn):
        if d["status"] != "pending":
            continue
        try:
            results.append(verify_delivery(conn, d["id"], receiver_url, timeout=timeout))
        except Exception as exc:  # recovery must never abort startup
            results.append({"delivery": d, "error": f"{type(exc).__name__}: {exc}"})
    return results
