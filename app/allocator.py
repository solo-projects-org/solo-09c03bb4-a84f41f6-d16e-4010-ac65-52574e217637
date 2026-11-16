"""Pure allocation logic.

Connected groups
----------------
Two slides are linked when they share a patient id, or when they share the
same *non-empty* repeat group. The transitive closure of those links is a
connected group (连通组). Every connected group must be placed entirely in one
split. A missing repeat group (NULL / empty) never links anything.

Delivery-driven test restrictions
---------------------------------
Deliveries carry a *training* list to the independent receiver.

* accepted delivery  -> its slides are permanently barred from the test split;
* pending / unknown  -> its slides are temporarily barred (state not confirmed,
                        receiver record missing, or the check timed out);
* rejected delivery  -> only that delivery's restriction is lifted; the slide
                        can become test-eligible again if no other delivery
                        bars it.

A barred slide drags its whole *current* connected group with it, because a
group can never be split. The restrictions are always derived from the current
delivery table, never from a separately maintained deny list, so a crash can
not lose them and a rejection can not accidentally clear unrelated bans.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable


class Unsolvable(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Group:
    index: int
    slides: list[dict[str, Any]]
    benign: int = 0
    malignant: int = 0

    @property
    def size(self) -> int:
        return len(self.slides)

    @property
    def slide_ids(self) -> set[str]:
        return {s["slide_id"] for s in self.slides}

    @property
    def patient_ids(self) -> set[str]:
        return {s["patient_id"] for s in self.slides}

    @property
    def repeat_groups(self) -> set[str]:
        return {s["repeat_group"] for s in self.slides if s.get("repeat_group")}


def build_groups(slides: Iterable[dict[str, Any]]) -> list[Group]:
    slides = list(slides)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for s in slides:
        node = f"slide:{s['slide_id']}"
        find(node)
        union(node, f"patient:{s['patient_id']}")
        rg = s.get("repeat_group")
        if rg:
            union(node, f"group:{rg}")

    buckets: dict[str, list[dict[str, Any]]] = {}
    for s in slides:
        buckets.setdefault(find(f"slide:{s['slide_id']}"), []).append(s)

    groups: list[Group] = []
    for i, (_, members) in enumerate(sorted(buckets.items(), key=lambda kv: kv[0])):
        g = Group(index=i, slides=sorted(members, key=lambda x: x["slide_id"]))
        for s in g.slides:
            if s["malignancy"] == "benign":
                g.benign += 1
            else:
                g.malignant += 1
        groups.append(g)
    return groups


# --------------------------------------------------------------------- bans


@dataclass
class Restriction:
    slide_id: str
    code: str  # 'accepted' | 'uncertain' | 'grouped'
    detail: str


@dataclass
class RestrictionState:
    accepted: set[str] = field(default_factory=set)
    uncertain: set[str] = field(default_factory=set)
    barred: set[str] = field(default_factory=set)
    reasons: dict[str, Restriction] = field(default_factory=dict)

    def explain(self, slide_id: str) -> str | None:
        r = self.reasons.get(slide_id)
        return r.detail if r else None


def compute_restrictions(
    slides: list[dict[str, Any]],
    accepted: set[str],
    uncertain: set[str],
) -> RestrictionState:
    """Bar accepted slides and slides in unconfirmed deliveries, plus their groups."""
    state = RestrictionState(accepted=set(accepted), uncertain=set(uncertain))
    direct = state.accepted | state.uncertain
    for sid in sorted(direct):
        if sid in state.accepted:
            state.reasons[sid] = Restriction(
                sid,
                "accepted",
                "该切片被终态接纳的交付覆盖，永久禁入测试",
            )
        else:
            state.reasons[sid] = Restriction(
                sid,
                "uncertain",
                "该切片所在交付尚未确认（待核实/查无记录/超时），暂禁入测试",
            )
    state.barred = set(direct)
    groups = build_groups(slides)
    for group in groups:
        members = [s["slide_id"] for s in group.slides]
        hits = [sid for sid in members if sid in direct]
        if not hits:
            continue
        for sid in members:
            if sid in state.barred:
                continue
            state.barred.add(sid)
            state.reasons[sid] = Restriction(
                sid,
                "grouped",
                "同一传递连通组内存在禁入切片（" + "、".join(sorted(hits)) + "），整组暂禁入测试",
            )
    return state


# --------------------------------------------------------------------- solve


@dataclass
class SolveResult:
    membership: dict[str, str]  # slide_id -> 'test' | 'train'
    groups: list[Group]
    test_benign: int
    test_malignant: int
    train_benign: int
    train_malignant: int
    checksum: str
    fingerprint: str
    library_count: int
    restrictions: RestrictionState


def fingerprint(slides: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{s['slide_id']}|{s['patient_id']}|{s.get('repeat_group') or ''}|{s['malignancy']}"
        for s in sorted(slides, key=lambda x: x["slide_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_checksum(rows: Iterable[dict[str, Any]]) -> str:
    """Order-independent checksum of a fixed slide list.

    Rows must expose slide_id, patient_id, repeat_group, malignancy, split.
    """
    lines = []
    for r in rows:
        lines.append(
            "|".join(
                [
                    str(r["slide_id"]),
                    str(r["patient_id"]),
                    str(r.get("repeat_group") or ""),
                    str(r["malignancy"]),
                    str(r.get("split") or ""),
                ]
            )
        )
    lines.sort()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _find_test_groups(
    groups: list[Group],
    eligible: set[str],
    req_benign: int,
    req_malignant: int,
) -> list[Group] | None:
    """Exact 0/1 subset-sum over whole groups.

    Only groups whose every slide is eligible may enter the test split.
    Exact dynamic programming over reachable (benign, malignant) pairs; the
    library is capped at 100 slides, so the state space is at most 101x101.
    """
    optional = [g for g in groups if all(s["slide_id"] in eligible for s in g.slides)]
    total_b = sum(g.benign for g in optional)
    total_m = sum(g.malignant for g in optional)
    if total_b < req_benign or total_m < req_malignant:
        return None

    target = (req_benign, req_malignant)
    reachable: set[tuple[int, int]] = {(0, 0)}
    parent: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    for idx, g in enumerate(optional):
        pending: dict[tuple[int, int], tuple[int, int]] = {}
        for b, m in list(reachable):
            nb, nm = b + g.benign, m + g.malignant
            if nb > req_benign or nm > req_malignant:
                continue
            if (nb, nm) in reachable or (nb, nm) in pending:
                continue
            pending[(nb, nm)] = (b, m)
        reachable.update(pending)
        for cell, prev in pending.items():
            parent[cell] = (prev, idx)
        if target in reachable:
            break

    if target not in reachable:
        return None

    chosen: list[Group] = []
    cell = target
    while cell != (0, 0):
        prev, idx = parent[cell]
        chosen.append(optional[idx])
        cell = prev
    return sorted(chosen, key=lambda g: g.index)


def reachable_test_totals(
    groups: list[Group], eligible: set[str]
) -> set[tuple[int, int]]:
    """All (benign, malignant) totals obtainable from whole eligible groups."""
    optional = [g for g in groups if all(s["slide_id"] in eligible for s in g.slides)]
    reachable: set[tuple[int, int]] = {(0, 0)}
    for g in optional:
        reachable |= {(b + g.benign, m + g.malignant) for b, m in reachable}
    return reachable


def solve(
    slides: list[dict[str, Any]],
    accepted: set[str],
    uncertain: set[str],
    req_benign: int,
    req_malignant: int,
) -> SolveResult:
    if req_benign < 0 or req_malignant < 0:
        raise Unsolvable("两类测试数量都不能为负数。")
    slides = sorted(slides, key=lambda s: s["slide_id"])
    if not slides:
        raise Unsolvable("库中没有任何切片，无法划分。")

    groups = build_groups(slides)
    restrictions = compute_restrictions(slides, accepted, uncertain)
    eligible = {s["slide_id"] for s in slides} - restrictions.barred

    total_b = sum(1 for s in slides if s["malignancy"] == "benign")
    total_m = len(slides) - total_b
    eligible_b = sum(
        1 for s in slides if s["slide_id"] in eligible and s["malignancy"] == "benign"
    )
    eligible_m = sum(
        1 for s in slides if s["slide_id"] in eligible and s["malignancy"] == "malignant"
    )
    needed = req_benign + req_malignant

    if needed > len(slides):
        raise Unsolvable(
            f"要求的测试总数 {req_benign}（良性）+ {req_malignant}（恶性）= {needed} "
            f"超过全库切片总数 {len(slides)}。"
        )
    if req_benign > eligible_b or req_malignant > eligible_m:
        parts = []
        if req_benign > eligible_b:
            parts.append(
                f"良性需要 {req_benign} 张，但禁入后仅剩 {eligible_b} 张可用"
                f"（全库良性 {total_b} 张，被禁入 {total_b - eligible_b} 张）。"
            )
        if req_malignant > eligible_m:
            parts.append(
                f"恶性需要 {req_malignant} 张，但禁入后仅剩 {eligible_m} 张可用"
                f"（全库恶性 {total_m} 张，被禁入 {total_m - eligible_m} 张）。"
            )
        parts.append("禁入来源：" + _restriction_summary(restrictions) + "。")
        raise Unsolvable("".join(parts))

    chosen = _find_test_groups(groups, eligible, req_benign, req_malignant)
    if chosen is None:
        forced = [
            g for g in groups if any(s["slide_id"] in restrictions.barred for s in g.slides)
        ]
        reachable = reachable_test_totals(groups, eligible)
        nearest = _closest_totals(reachable, req_benign, req_malignant)
        hint = ""
        if nearest:
            hint = (
                "当前可行组合中最接近的是："
                + "、".join(f"{b} 良 + {m} 恶" for b, m in nearest[:4])
                + "。"
            )
        raise Unsolvable(
            "无法按连通组精确划分："
            f"要求测试良性 {req_benign} 张、恶性 {req_malignant} 张；"
            f"未被禁入的切片为良性 {eligible_b} 张、恶性 {eligible_m} 张，"
            f"共组成 {len(groups)} 个传递连通组。"
            "不存在由若干个完整连通组构成的组合能恰好匹配这两个数量。"
            + (
                f"其中 {len(forced)} 个连通组含禁入切片，只能整组留在训练集；"
                if forced
                else ""
            )
            + hint
            + "可减小测试数量、调整两类配比，或处理相关交付（拒收/核实）后重试。"
        )

    test_ids = {s["slide_id"] for g in chosen for s in g.slides}
    membership = {
        s["slide_id"]: ("test" if s["slide_id"] in test_ids else "train")
        for s in slides
    }
    test_b = sum(1 for s in slides if s["slide_id"] in test_ids and s["malignancy"] == "benign")
    test_m = len(test_ids) - test_b
    if (test_b, test_m) != (req_benign, req_malignant):
        raise Unsolvable("内部一致性检查失败：求解结果与要求不符，名单未发布。")

    rows = [
        {
            "slide_id": s["slide_id"],
            "patient_id": s["patient_id"],
            "repeat_group": s.get("repeat_group") or "",
            "malignancy": s["malignancy"],
            "split": membership[s["slide_id"]],
        }
        for s in slides
    ]
    return SolveResult(
        membership=membership,
        groups=groups,
        test_benign=test_b,
        test_malignant=test_m,
        train_benign=total_b - test_b,
        train_malignant=total_m - test_m,
        checksum=canonical_checksum(rows),
        fingerprint=fingerprint(slides),
        library_count=len(slides),
        restrictions=restrictions,
    )


def _closest_totals(
    reachable: set[tuple[int, int]], req_benign: int, req_malignant: int
) -> list[tuple[int, int]]:
    usable = [c for c in reachable if c != (0, 0)]
    usable.sort(key=lambda c: (abs(c[0] - req_benign) + abs(c[1] - req_malignant), c))
    return usable[:4]


def _restriction_summary(state: RestrictionState) -> str:
    parts = []
    if state.accepted:
        parts.append(f"{len(state.accepted)} 张被终态接纳的交付永久禁入")
    if state.uncertain:
        parts.append(f"{len(state.uncertain)} 张所在交付状态未明、暂禁入")
    if state.barred - state.accepted - state.uncertain:
        parts.append(f"{len(state.barred - state.accepted - state.uncertain)} 张因连通组牵连被禁入")
    return "；".join(parts) if parts else "无"


# ---------------------------------------------------------- plan re-checks


def evaluate_plan(
    membership: dict[str, str],
    frozen_rows: list[dict[str, Any]],
    current_slides: list[dict[str, Any]],
    restrictions: RestrictionState,
) -> list[str]:
    """Return human-readable isolation failures for a frozen plan.

    An empty list means the plan is still deliverable. Checks are made against
    the *current* library and the *current* delivery state, so later uploads
    and delivery transitions are caught. The plan's frozen content is never
    modified; a failing plan is merely suspended (stops serving downloads).
    """
    problems: list[str] = []
    by_id = {s["slide_id"]: s for s in current_slides}
    frozen = {r["slide_id"]: r for r in frozen_rows}

    # 1. every slide of the frozen list still exists and still matches
    for sid, row in sorted(frozen.items()):
        live = by_id.get(sid)
        if live is None:
            problems.append(f"名单中的切片 {sid} 已被移出总库。")
            continue
        if (
            live["patient_id"] != row["patient_id"]
            or (live.get("repeat_group") or None) != (row.get("repeat_group") or None)
            or live["malignancy"] != row["malignancy"]
        ):
            problems.append(f"名单中的切片 {sid} 在总库中的患者号/重复图组/良恶性已变化。")

    # 2. test slides must not be barred by the current delivery state
    for sid, split in sorted(membership.items()):
        if split != "test":
            continue
        if sid not in by_id:
            problems.append(f"测试切片 {sid} 已不在库中。")
            continue
        reason = restrictions.explain(sid)
        if reason:
            problems.append(f"测试切片 {sid} 被禁入：{reason}。")

    # 3. no current connected group may straddle the frozen sides, and a group
    #    containing a test slide must be fully represented in the frozen list
    for group in build_groups(current_slides):
        ids = group.slide_ids
        sides = {membership[sid] for sid in ids if sid in membership}
        if len(sides) > 1:
            problems.append(
                "当前连通组 {"
                + "、".join(sorted(ids))
                + "} 在名单中横跨训练集与测试集，违反同连通组同集要求。"
            )
        if "test" in sides:
            missing = sorted(ids - set(membership))
            if missing:
                problems.append(
                    "当前连通组含测试切片，但其中切片 "
                    + "、".join(missing)
                    + " 未纳入本名单，无法确认隔离。"
                )
    return problems
