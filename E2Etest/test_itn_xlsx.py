"""Compare chinese_to_num against test_itn.xlsx (col A = ASR, col B = expected ITN).

    conda activate lingting
    python E2Etest/test_itn_xlsx.py
    python E2Etest/test_itn_xlsx.py D:\\path\\test_itn.xlsx
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chinese_itn import chinese_to_num

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_CELL_REF = re.compile(r"^([A-Z]+)(\d+)$")


def _xlsx_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    for candidate in (
        ROOT / "E2Etest" / "test_itn.xlsx",
        ROOT / "test_itn.xlsx",
        Path.cwd() / "test_itn.xlsx",
    ):
        if candidate.is_file():
            return candidate
    return ROOT / "E2Etest" / "test_itn.xlsx"


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    kind = cell.attrib.get("t")
    if kind == "inlineStr":
        return "".join((t.text or "") for t in cell.findall(".//m:t", _NS))
    value = cell.find("m:v", _NS)
    if value is None or value.text is None:
        return ""
    raw = value.text
    if kind == "s":
        return shared[int(raw)]
    return raw


def _read_rows(path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _NS):
                shared.append("".join((t.text or "") for t in si.findall(".//m:t", _NS)))
        sheet_name = next(
            (n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")),
            None,
        )
        if sheet_name is None:
            raise RuntimeError(f"no worksheet in {path}")
        sheet = ET.fromstring(zf.read(sheet_name))

    pairs: list[tuple[str, str]] = []
    for row in sheet.findall("m:sheetData/m:row", _NS):
        cols: dict[int, str] = {}
        for cell in row.findall("m:c", _NS):
            ref = cell.attrib.get("r", "")
            m = _CELL_REF.match(ref)
            if not m:
                continue
            cols[_col_index(m.group(1))] = _cell_text(cell, shared).strip()
        src = cols.get(0, "").strip()
        exp = cols.get(1, "").strip()
        if not src and not exp:
            continue
        if src in {"ASR", "输入", "原文", "asr"} or exp in {"ITN", "期望", "expected"}:
            continue
        pairs.append((src, exp))
    return pairs


def main() -> int:
    path = _xlsx_path()
    if not path.is_file():
        print(f"xlsx not found: {path}", file=sys.stderr)
        print("put test_itn.xlsx in E2Etest/ or pass the path as argv", file=sys.stderr)
        return 2

    rows = _read_rows(path)
    fails: list[tuple[int, str, str, str]] = []
    for i, (src, expected) in enumerate(rows, start=1):
        got = chinese_to_num(src)
        if got != expected:
            fails.append((i, src, expected, got))

    print(f"file={path} cases={len(rows)} ok={len(rows) - len(fails)} fail={len(fails)}")
    if not fails:
        print("all ITN cases matched")
        return 0

    print("mismatches:")
    for i, src, expected, got in fails:
        print(f"  #{i}")
        print(f"    输入: {src}")
        print(f"    期望: {expected}")
        print(f"    实际: {got}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
