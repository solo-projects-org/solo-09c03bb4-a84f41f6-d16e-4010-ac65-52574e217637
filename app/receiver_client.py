"""Receiver client: idempotent delivery to the independent receiving service.

Protocol (implemented by ``receiver/``)
---------------------------------------
GET {receiver_url}/api/identity
  -> {"receiver_id": "RCV-…", "name": "…"}   stable per receiving data volume

POST {receiver_url}/api/deliveries
    {"delivery_id", "checksum", "slides": [{"slide_id","patient_id",
     "repeat_group","malignancy","split"}]}
  -> 200/201 {"receipt_ref", "status": "accepted"|"rejected", "checksum",
              "receiver_id", ...}

The receiver stores the delivery id once. Repeating the same POST returns the
same receipt (idempotent), so a retry after a timeout or a crash never creates
a second acceptance. Every receipt carries the receiver id, so a receipt from a
*different* volume (same address, replaced data) is never accepted as proof.

Every network failure, non-2xx response, malformed body and timeout is mapped
to ``"unknown"``. Unknown is *not* rejection: the slide stays barred from test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DeliveryOutcome:
    state: str  # 'accepted' | 'rejected' | 'unknown'
    receipt_ref: str | None = None
    payload: dict[str, Any] | None = None
    error: str | None = None
    http_status: int | None = None
    receiver_id: str | None = None
    checksum: str | None = None


def _normalise(data: dict[str, Any], expected_checksum: str) -> DeliveryOutcome:
    status = str(data.get("status", "")).lower()
    receipt = data.get("receipt_ref") or data.get("receipt")
    receiver_id = data.get("receiver_id")
    if status not in {"accepted", "rejected"}:
        return DeliveryOutcome(
            state="unknown",
            error=f"接收服务返回无法识别的状态：{data.get('status')!r}",
            payload=data,
            receiver_id=receiver_id,
        )
    if not receipt:
        return DeliveryOutcome(
            state="unknown",
            error="接收服务返回缺少回执号，视为未明交付。",
            payload=data,
            receiver_id=receiver_id,
        )
    if data.get("checksum") and data["checksum"] != expected_checksum:
        return DeliveryOutcome(
            state="unknown",
            error="接收服务回执校验和与发送内容不一致，视为未明交付。",
            payload=data,
            receiver_id=receiver_id,
        )
    return DeliveryOutcome(
        state=status,
        receipt_ref=str(receipt),
        payload=data,
        receiver_id=str(receiver_id) if receiver_id else None,
        checksum=str(data.get("checksum") or expected_checksum),
    )


def fetch_identity(receiver_url: str, timeout: float = 10.0) -> str | None:
    """Return the receiving volume's identity, or None when unreachable.

    Failure to read an identity is not proof of anything: the caller must not
    bind or confirm a delivery against a receiver of unknown identity.
    """
    import httpx

    url = receiver_url.rstrip("/") + "/api/identity"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    ident = data.get("receiver_id")
    return str(ident) if ident else None


def send_delivery(
    receiver_url: str,
    delivery_id: str,
    checksum: str,
    slides: list[dict[str, Any]],
    timeout: float = 10.0,
    expected_receiver_id: str | None = None,
) -> DeliveryOutcome:
    """POST a frozen delivery. Never raises; all failures become 'unknown'.

    ``expected_receiver_id`` guards a retry of an already-bound delivery: it is
    sent along, and the receiving service refuses to store the list when the
    id does not match its own volume. The response is checked as well, so a
    receipt from a substitute volume never confirms the delivery.
    """
    import httpx

    url = receiver_url.rstrip("/") + "/api/deliveries"
    body = {
        "delivery_id": delivery_id,
        "checksum": checksum,
        "slides": slides,
    }
    if expected_receiver_id:
        body["expected_receiver_id"] = expected_receiver_id
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body)
    except Exception as exc:  # httpx errors, DNS, connection reset, timeout ...
        return DeliveryOutcome(state="unknown", error=f"发送失败：{type(exc).__name__}: {exc}")
    if resp.status_code not in (200, 201):
        detail = resp.text[:300]
        if resp.status_code == 409 and not detail:
            detail = "接收服务拒绝了该交付（校验和或接收方身份不符）"
        return DeliveryOutcome(
            state="unknown",
            error=f"接收服务返回 HTTP {resp.status_code}：{detail}",
            http_status=resp.status_code,
        )
    try:
        data = resp.json()
    except Exception:
        return DeliveryOutcome(
            state="unknown",
            error="接收服务响应不是合法 JSON，视为未明交付。",
            http_status=resp.status_code,
        )
    if not isinstance(data, dict):
        return DeliveryOutcome(
            state="unknown", error="接收服务响应结构异常，视为未明交付。", http_status=resp.status_code
        )
    if expected_receiver_id and str(data.get("receiver_id") or "") != expected_receiver_id:
        return DeliveryOutcome(
            state="unknown",
            error=(
                "接收服务身份与交付绑定的原接收方不一致，视为未明交付；"
                "未向替代方重新登记。"
            ),
            payload=data,
            http_status=resp.status_code,
        )
    outcome = _normalise(data, checksum)
    outcome.http_status = resp.status_code
    return outcome


def query_delivery(
    receiver_url: str,
    delivery_id: str,
    expected_checksum: str | None = None,
    timeout: float = 10.0,
    expected_receiver_id: str | None = None,
) -> DeliveryOutcome:
    """GET the receiver's record. 404 / network error / bad body => unknown.

    ``expected_checksum`` (the checksum frozen at send time) must be supplied
    so a record for a different payload can never be mistaken for ours.
    ``expected_receiver_id`` additionally requires the record to come from the
    volume the delivery is bound to; a same-id record on another volume (e.g.
    a replaced data volume behind the same address) is *not* confirmation.
    """
    import httpx

    url = receiver_url.rstrip("/") + f"/api/deliveries/{delivery_id}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
    except Exception as exc:
        return DeliveryOutcome(state="unknown", error=f"核实失败：{type(exc).__name__}: {exc}")
    if resp.status_code == 404:
        return DeliveryOutcome(
            state="unknown", error="查无记录：接收服务无该交付号", http_status=404
        )
    if resp.status_code != 200:
        return DeliveryOutcome(
            state="unknown",
            error=f"核实返回 HTTP {resp.status_code}：{resp.text[:300]}",
            http_status=resp.status_code,
        )
    try:
        data = resp.json()
    except Exception:
        return DeliveryOutcome(
            state="unknown", error="核实响应不是合法 JSON。", http_status=200
        )
    if not isinstance(data, dict):
        return DeliveryOutcome(state="unknown", error="核实响应结构异常。", http_status=200)
    if not data.get("status"):
        return DeliveryOutcome(state="unknown", error="核实响应缺少状态。", payload=data, http_status=200)
    record_receiver = str(data.get("receiver_id") or "")
    if expected_receiver_id and record_receiver != expected_receiver_id:
        return DeliveryOutcome(
            state="unknown",
            error=(
                "核实到的记录不属于交付绑定的原接收方（地址或数据卷已改变），"
                "未确认为本次交付。"
            ),
            payload=data,
            http_status=200,
        )
    outcome = _normalise(data, expected_checksum or str(data.get("checksum", "")))
    outcome.http_status = resp.status_code
    return outcome
