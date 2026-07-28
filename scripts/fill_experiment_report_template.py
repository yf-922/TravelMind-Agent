"""Fill the supplied Word template without changing its page/table layout."""
from __future__ import annotations

import copy
import os
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "实验提交材料" / "Word实验报告"
SOURCE_DIR = ROOT / "实验提交材料"
TEMPLATE_DIR = Path(r"C:\Users\31071\xwechat_files\wxid_1imimo0axrse22_f868\msg\file\2026-07")
TEMPLATE = next(p for p in TEMPLATE_DIR.glob("*.docx") if "人工智能应用实践" in p.name and "报告模板" in p.name)

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W = "{" + NS["w"] + "}"
ET.register_namespace("w", NS["w"])

EXPERIMENTS = {
    1: "需求分析与立项",
    2: "概要设计与详细设计",
    3: "核心实现（多智能体编排）",
    4: "测试与评估",
}


def paragraph_text(p: ET.Element) -> str:
    return "".join(node.text or "" for node in p.findall(".//w:t", NS))


def set_text_keep_format(p: ET.Element, text: str) -> None:
    ts = p.findall(".//w:t", NS)
    if not ts:
        run = ET.SubElement(p, W + "r")
        ts = [ET.SubElement(run, W + "t")]
    ts[0].text = text
    for node in ts[1:]:
        node.text = ""


def make_paragraph(template_p: ET.Element, text: str, *, heading: bool = False) -> ET.Element:
    p = copy.deepcopy(template_p)
    for child in list(p):
        if child.tag != W + "pPr":
            p.remove(child)
    run = ET.SubElement(p, W + "r")
    rpr = ET.SubElement(run, W + "rPr")
    fonts = ET.SubElement(rpr, W + "rFonts")
    fonts.set(W + "eastAsia", "宋体" if not heading else "黑体")
    if heading:
        ET.SubElement(rpr, W + "b")
    size = ET.SubElement(rpr, W + "sz")
    size.set(W + "val", "24")  # 小四号
    t = ET.SubElement(run, W + "t")
    t.text = text
    return p


def extract_sections(markdown: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"一、实验目的": [], "二、实验内容": [], "三、实验步骤": [], "四、心得体会": []}
    current = None
    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            name = line[3:].strip()
            current = name if name in result else None
            continue
        if not current or not line or line.startswith("#") or line.startswith("姓名："):
            continue
        line = line.replace("**", "")
        if line.startswith(("- ", "* ")):
            line = "• " + line[2:]
        result[current].append(line)
    return result


def replace_cell(cell: ET.Element, sections: list[tuple[str, list[str]]]) -> None:
    template_p = cell.find("w:p", NS)
    if template_p is None:
        template_p = ET.Element(W + "p")
    for child in list(cell):
        if child.tag != W + "tcPr":
            cell.remove(child)
    for heading, content in sections:
        cell.append(make_paragraph(template_p, heading, heading=True))
        for line in content:
            indent = "　　" if not line.startswith("•") and not line[:2].isdigit() else ""
            cell.append(make_paragraph(template_p, indent + line))


def build(experiment: int) -> Path:
    md = (SOURCE_DIR / f"202400202042_张家乐_实验{experiment}_实验报告.md").read_text(encoding="utf-8")
    sections = extract_sections(md)
    with zipfile.ZipFile(TEMPLATE) as zf:
        document = ET.fromstring(zf.read("word/document.xml"))
        # Cover page replacement, preserving existing paragraph/run formatting.
        for p in document.findall(".//w:p", NS):
            raw = paragraph_text(p).strip()
            if raw.startswith("实验名称："):
                set_text_keep_format(p, f"实验名称：        实验{experiment}·{EXPERIMENTS[experiment]}")
            elif raw.startswith("班") and "级：" in raw:
                set_text_keep_format(p, "班    级：        2024级计算机科学与技术1班")
            elif raw.startswith("报告人："):
                set_text_keep_format(p, "报 告 人：             张家乐          学号：202400202042")
            elif raw.startswith("实验时间："):
                set_text_keep_format(p, "实验时间：   2026 年 7 月 28 日 星期二")
            elif raw.startswith("提交时间："):
                set_text_keep_format(p, "提交时间：   2026 年 7 月 28 日")

        tables = document.findall(".//w:tbl", NS)
        report_table = tables[-1]
        rows = report_table.findall("w:tr", NS)
        first_cell = rows[0].find("w:tc", NS)
        second_cell = rows[1].find("w:tc", NS)
        assert first_cell is not None and second_cell is not None
        replace_cell(first_cell, [("一、实验目的", sections["一、实验目的"]), ("二、实验内容", sections["二、实验内容"]), ("三、实验步骤", sections["三、实验步骤"])])
        replace_cell(second_cell, [("四、心得体会", sections["四、心得体会"])])

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"202400202042_张家乐_实验{experiment}_实验报告.docx"
        with tempfile.TemporaryDirectory() as td:
            patched = Path(td) / "document.xml"
            ET.ElementTree(document).write(patched, encoding="utf-8", xml_declaration=True)
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dest:
                for info in zf.infolist():
                    data = patched.read_bytes() if info.filename == "word/document.xml" else zf.read(info.filename)
                    dest.writestr(info, data)
    return out


if __name__ == "__main__":
    for number in EXPERIMENTS:
        print(build(number))
