"""Receiver client: idempotent delivery to the independent receiving service.

Protocol (implemented by ``receiver/``)
---------------------------------------
POST {receiver_url}/api/deliveries
    {"delivery_id", "checksum", "slides": [{"slide_id","patient_id",
     "repeat_group","malignancy","split"}]}
  -> 200/201 {"receipt_ref", "status": "accepted"|"rejected", "checksum", ...}

The receiver stores the delivery id once. Repeating the same POST returns the
same receipt (idempotent), so a retry after a timeout or a crash never creates
a second acceptance.

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


def _normalise(data: dict[str, Any], expected_checksum: str) -> DeliveryOutcome:
    status = str(data.get("status", "")).lower()
    receipt = data.get("receipt_ref") or data.get("receipt")
    if status not in {"accepted", "rejected"}:
        return DeliveryOutcome(
            state="unknown",
            error=f"接收服务返回无法识别的状态：{data.get('status')!r}",
            payload=data,
        )
    if not receipt:
        return DeliveryOutcome(
            state="unknown",
            error="接收服务返回缺少回执号，视为未明交付。",
            payload=data,
        )
    if data.get("checksum") and data["checksum"] != expected_checksum:
        return DeliveryOutcome(
            state="unknown",
            error="接收服务回执校验和与发送内容不一致，视为未明交付。",
            payload=data,
        )
    return DeliveryOutcome(state=status, receipt_ref=str(receipt), payload=data)


def send_delivery(
    receiver_url: str,
    delivery_id: str,
    checksum: str,
    slides: list[dict[str, Any]],
    timeout: float = 10.0,
) -> DeliveryOutcome:
    """POST a frozen delivery. Never raises; all failures become 'unknown'."""
    import httpx

    url = receiver_url.rstrip("/") + "/api/deliveries"
    body = {
        "delivery_id": delivery_id,
        "checksum": checksum,
        "slides": slides,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body)
    except Exception as exc:  # httpx errors, DNS, connection reset, timeout ...
        return DeliveryOutcome(state="unknown", error=f"发送失败：{type(exc).__name__}: {exc}")
    if resp.status_code not in (200, 201):
        return DeliveryOutcome(
            state="unknown",
            error=f"接收服务返回 HTTP {resp.status_code}：{resp.text[:300]}",
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
    outcome = _normalise(data, checksum)
    outcome.http_status = resp.status_code
    return outcome


def query_delivery(
    receiver_url: str,
    delivery_id: str,
    expected_checksum: str | None = None,
    timeout: float = 10.0,
) -> DeliveryOutcome:
    """GET the receiver's record. 404 / network error / bad body => unknown.

    ``expected_checksum`` (the checksum frozen at send time) must be supplied
    so a record for a different payload can never be mistaken for ours.
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
    outcome = _normalise(data, expected_checksum or str(data.get("checksum", "")))
    outcome.http_status = resp.status_code
    return outcome
