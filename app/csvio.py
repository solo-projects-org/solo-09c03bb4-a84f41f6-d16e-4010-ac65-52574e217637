"""CSV parsing / serialisation.

CSV columns: slide_id,patient_id,repeat_group,malignancy

* header optional, case-insensitive, Chinese aliases accepted;
* repeat_group may be empty (NULL) and means "no repeat group";
* malignancy must be benign/malignant (良/恶 aliases accepted);
* a row may not be split across batches: the whole file is validated first.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Any

SLIDE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,39}$")
PATIENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,39}$")
GROUP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,39}$")

HEADERS = {
    "slide_id": {"slide_id", "slideid", "slice_id", "切片号", "切片编号"},
    "patient_id": {"patient_id", "patientid", "patient", "患者号", "患者编号"},
    "repeat_group": {"repeat_group", "repeatgroup", "group", "重复图组", "重复组"},
    "malignancy": {"malignancy", "label", "diagnosis", "良恶性", "性质"},
}

MALIGNANT_ALIASES = {"malignant", "m", "mal", "恶", "恶性", "bad", "1", "positive", "pos"}
BENIGN_ALIASES = {"benign", "b", "良", "良性", "good", "0", "negative", "neg"}


class CsvError(Exception):
    pass


@dataclass
class ParsedRow:
    slide_id: str
    patient_id: str
    repeat_group: str | None
    malignancy: str
    line: int


def _norm_header(cell: str) -> str:
    return cell.strip().lower().replace(" ", "_").replace("-", "_")


def _classify_malignancy(value: str, line: int) -> str:
    v = value.strip().lower()
    if v in MALIGNANT_ALIASES:
        return "malignant"
    if v in BENIGN_ALIASES:
        return "benign"
    raise CsvError(f"第 {line} 行：良恶性取值“{value}”无法识别（应为 benign/malignant）。")


def parse_csv_text(text: str, assumed_header: bool = True) -> list[ParsedRow]:
    if text is None:
        raise CsvError("CSV 内容为空。")
    text = text.lstrip("﻿")
    if not text.strip():
        raise CsvError("CSV 内容为空。")

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    raw = list(reader)
    if not raw:
        raise CsvError("CSV 内容为空。")

    first = [_norm_header(c) for c in raw[0]]
    has_header = assumed_header and any(
        cell in HEADERS["slide_id"] for cell in first
    )
    if has_header:
        mapping: dict[str, int] = {}
        for idx, cell in enumerate(first):
            for field, aliases in HEADERS.items():
                if cell in aliases and field not in mapping:
                    mapping[field] = idx
        missing = [f for f in ("slide_id", "patient_id", "malignancy") if f not in mapping]
        if missing:
            raise CsvError(
                "表头缺少必需列：" + "、".join(missing) + "。应包含 slide_id,patient_id,malignancy[,repeat_group]。"
            )
    else:
        mapping = {"slide_id": 0, "patient_id": 1}
        if len(raw[0]) >= 4:
            mapping["repeat_group"] = 2
            mapping["malignancy"] = 3
        elif len(raw[0]) == 3:
            mapping["malignancy"] = 2
        else:
            raise CsvError(
                "无法识别 CSV 列：无表头时需 3 列(slide,patient,malignancy) 或 4 列(slide,patient,group,malignancy)。"
            )

    rows: list[ParsedRow] = []
    seen: dict[str, int] = {}
    errors: list[str] = []
    start = 1 if has_header else 0
    max_idx = max(mapping.values())
    for line_no, cells in enumerate(raw[start:], start=start + 1):
        if not any(c.strip() for c in cells):
            continue
        if len(cells) <= max_idx:
            errors.append(f"第 {line_no} 行：列数不足（需要至少 {max_idx + 1} 列）。")
            continue
        sid = cells[mapping["slide_id"]].strip()
        pid = cells[mapping["patient_id"]].strip()
        group = cells[mapping["repeat_group"]].strip() if "repeat_group" in mapping else ""
        mal = cells[mapping["malignancy"]].strip()
        try:
            if not SLIDE_RE.match(sid):
                raise CsvError(f"第 {line_no} 行：切片号“{sid}”格式非法（1-40 位字母数字或 ._-）。")
            if not PATIENT_RE.match(pid):
                raise CsvError(f"第 {line_no} 行：患者号“{pid}”格式非法。")
            if group and not GROUP_RE.match(group):
                raise CsvError(f"第 {line_no} 行：重复图组“{group}”格式非法。")
            malignancy = _classify_malignancy(mal, line_no)
        except CsvError as exc:
            errors.append(str(exc))
            continue
        if sid in seen:
            errors.append(f"第 {line_no} 行：切片号 {sid} 与第 {seen[sid]} 行重复，本批未写入。")
            continue
        seen[sid] = line_no
        rows.append(
            ParsedRow(
                slide_id=sid,
                patient_id=pid,
                repeat_group=group or None,
                malignancy=malignancy,
                line=line_no,
            )
        )

    if errors:
        raise CsvError("CSV 校验失败（整批未写入）：\n" + "\n".join(errors))
    if not rows:
        raise CsvError("CSV 中没有可导入的数据行。")
    return rows


def render_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for r in rows:
        writer.writerow(["" if r.get(c) is None else str(r.get(c)) for c in columns])
    return out.getvalue()
