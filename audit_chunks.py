"""只读审核原始片段；导出乱码证据及分层样本，不执行 OCR。"""

from collections import Counter
import csv
import hashlib
import html
import json
import re
import unicodedata

from build_chunks import BASE_DIR, page_text, pymupdf

OUTPUT = BASE_DIR / "output" / "chunk_audit"

# 以下为本次实际查看页面 PNG 后记录的人工图文对照，不是自动推断。
REVIEWED_SHA256 = "3e88662390ad7ace0e0546c40a1e1e77e1419cacc2acd5eb6d506c3a5c5bdcc9"
REVIEWS = {
    18: ("图表局部乱码", "ὖ㽕", "概要；准备工作；各部分的名称与基本操作", "目录结构图中文字损坏；页眉、标题正常。", None),
    52: ("图表局部乱码", "ᢨߎЏऩܗ", "拔出主单元；推压挂钩；单个安装；紧密安装", "正文正常；部分安装图标签乱码，圈号也被提取为控制字符。", None),
    84: ("图表局部乱码", "䗮ֵ", "通信；RS-485；事件输入；端子编号", "接线图重要标签和端子数字损坏，不能只当无害符号。", [0.18, 0.13, 0.93, 0.65]),
    116: ("图表局部乱码", "Ӵᛳ఼", "传感器输入/CT输入；电源；强化绝缘；功能绝缘", "前两幅绝缘块图标签乱码，第三幅 E5DC-B 的文字正常。", [0.18, 0.12, 0.72, 0.56]),
    142: ("图表局部乱码", "أᏂϟ䰤", "偏差下限；设定点；报警上限/下限", "表格的说明列可读，动作图的文字、ON/OFF 和数值损坏。", [0.63, 0.16, 0.93, 0.90]),
    143: ("图表局部乱码", "/%$Ẕ⌟䯜ؐ", "LBA检测阈值；LBA检测带；LBA检测时间", "左侧报警名称及说明正常，右侧曲线图的文字损坏。", [0.62, 0.10, 0.93, 0.95]),
    168: ("图表局部乱码", "أᏂı", "偏差≥10%FS；有限周期MV的变动范围40%", "主体操作步骤可读，曲线图及显示框标签乱码。", [0.19, 0.16, 0.69, 0.45]),
    193: ("图表局部乱码", "㟇&7 䕧ܹ", "至CT输入；负载；10A；17.3A；200V", "计算说明及公式可读，三幅 V 形接线图的标签和数值损坏。", [0.23, 0.13, 0.91, 0.50]),
    212: ("图表局部乱码", "᥹䗮⬉⑤", "接通电源；SP斜坡；报警输出ON", "正文正常，斜坡示意图的轴标、状态标签乱码。", [0.23, 0.17, 0.85, 0.78]),
    274: ("图表局部乱码", "䖤㸠", "运行/停止；报警值1；多重设定点", "操作菜单流程图名称、按键时长损坏，页首概述正常。", [0.08, 0.17, 0.94, 0.91]),
    284: ("图表局部乱码", "᠟ࡼ໡ԡؐ", "手动复位值；内部辅助继电器；滞后（加热）", "页首正文正常，调整菜单流程图的主要中文标签乱码。", [0.09, 0.33, 0.93, 0.88]),
    313: ("图表局部乱码", "ᚙᔶ", "情形1；情形2；情形3（常时ON）；H、L、SP", "下半页说明正常，上半页报警图标签与不等式损坏。", [0.17, 0.10, 0.74, 0.43]),
    344: ("图表局部乱码", "䖰⿟63", "远程SP上限；远程SP下限；2.4（-10%）；21.6（110%）", "说明和参数表可读，曲线轴标与关键数值乱码。", [0.22, 0.24, 0.80, 0.46]),
    384: ("图表局部乱码", "/('˄5'˅", "LED（RD）；250 mm；263 mm；1510 mm；2110 mm", "规格表与LED说明正常，外形尺寸图数字和标签乱码。", [0.10, 0.10, 0.93, 0.28]),
    415: ("正文说明及图表混合乱码", "ⲥ᥻", "监控/设定项目菜单；高级功能设定菜单；E5DC没有这些参数", "菜单图标签、顶部说明句和底部适用型号脚注均损坏。", [0.12, 0.17, 0.92, 0.93]),
}


def suspicious_character(char, language):
    # 英文 F06C 为项目圆点、F0A3 为版权符号：已对照英文第4、3页。
    if char in "\n\r\t" or (language == "en" and char in "\uf06c\uf0a3"):
        return False
    return char == "\ufffd" or unicodedata.category(char) in {"Cc", "Co", "Cs", "Cn"}


def retrieval_screen(rows):
    """片段级保守筛选；字体风险按文本块定位，不修改片段原文。"""
    from build_chunks import SOURCES

    decisions = []
    for filename, language in SOURCES:
        selected = [r for r in rows if r["source_file"] == filename]
        by_page = {}
        for row in selected:
            by_page.setdefault(row["pdf_page"], []).append(row)
        with pymupdf.open(BASE_DIR / filename) as doc:
            for page_number, chunks in by_page.items():
                page = doc[page_number - 1]
                span_blocks = {b["number"]: b for b in page.get_text(
                    "dict", flags=pymupdf.TEXTFLAGS_BLOCKS)["blocks"] if b["type"] == 0}
                parts, risks, position = [], [], 0
                for block in page.get_text("blocks", sort=True):
                    if block[6] != 0 or not block[4].strip():
                        continue
                    text = block[4].strip()
                    if parts:
                        position += 2
                    fonts = {s["font"] for line in span_blocks[block[5]]["lines"]
                             for s in line["spans"] if s["text"].strip()}
                    risky_fonts = sorted(f for f in fonts if "GBK-EUC-H" in f or f in {
                        "DecoNumbersLH-Circle", "CG11seg", "Wingdings3"})
                    if risky_fonts:
                        risks.append((position, position + len(text), risky_fonts))
                    parts.append(text)
                    position += len(text)
                text = "\n\n".join(parts)
                assert text == page_text(page)
                previous_start = -1
                for row in chunks:
                    start = text.find(row["text"], previous_start + 1)
                    assert start >= 0
                    end = start + len(row["text"])
                    previous_start = start
                    fonts = sorted({font for a, b, fs in risks if a < end and b > start for font in fs})
                    bad = sorted({f"U+{ord(c):04X}" for c in row["text"]
                                  if suspicious_character(c, language)})
                    reasons = []
                    if fonts:
                        reasons.append("与疑似损坏字体所在文本块重叠：" + ", ".join(fonts))
                    if bad:
                        reasons.append("含未豁免异常字符：" + ", ".join(bad))
                    if "(cid:" in row["text"]:
                        reasons.append("含 CID 占位符")
                    if language == "zh":
                        enough_text = len(re.findall(r"[\u4e00-\u9fff]", row["text"])) >= 20
                    else:
                        enough_text = len(re.findall(r"\b[A-Za-z]{2,}\b", row["text"])) >= 12
                    if not enough_text:
                        reasons.append("自然语言内容不足，暂不作为独立检索片段")
                    decisions.append({k: row[k] for k in ("source_file", "language", "pdf_page", "chunk_id")}
                                     | {"eligible": not reasons, "reasons": reasons})
    return decisions


def write_review(evidence, rows):
    assert evidence["original_sha256"] == REVIEWED_SHA256, "输入已变化，人工对照结果必须重新核查"
    assert sorted(REVIEWS) == sorted(evidence["sample_pages"])
    decisions = retrieval_screen(rows)
    with (OUTPUT / "retrieval_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for decision in decisions:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
    counts = {language: dict(Counter("eligible" if d["eligible"] else "hold"
                                    for d in decisions if d["language"] == language))
              for language in ("zh", "en")}
    flagged = {p["pdf_page"] for p in evidence["pages"]}
    categories = dict(Counter(review[0] for review in REVIEWS.values()))
    categories.update({"纯符号无重要文字损坏/误报": 0, "未逐页视觉复核": len(flagged) - len(REVIEWS)})
    ocr_plan = []
    for number, review in REVIEWS.items():
        if review[4] is not None:
            ocr_plan.append({"source_file": "中文手册内容.pdf", "pdf_page": number,
                             "suggested_crop_normalized": review[4],
                             "important_text": review[2], "confirmed_problem": review[3],
                             "status": "仅方案，未执行OCR"})
    summary = {"original_sha256": evidence["original_sha256"], "source_unchanged": True,
               "reviewed_pages": evidence["sample_pages"], "page_class_counts": categories,
               "retrieval_counts": counts,
               "eligible_total": sum(c.get("eligible", 0) for c in counts.values()),
               "flagged_page_chunks": sum(r["language"] == "zh" and r["pdf_page"] in flagged for r in rows),
               "eligible_from_flagged_pages": sum(d["language"] == "zh" and d["pdf_page"] in flagged and d["eligible"] for d in decisions),
               "held_outside_flagged_zh_pages": sum(d["language"] == "zh" and d["pdf_page"] not in flagged and not d["eligible"] for d in decisions),
               "scope": "保守编码质量筛选白名单，不代表全书人工语义核验；图表关系仍须回看PDF。"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "targeted_ocr_plan.json").write_text(json.dumps(ocr_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "flagged_pages.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pdf_page", "abnormal_count", "visible_count", "ratio", "cid", "trigger", "visual_class"])
        for page in evidence["pages"]:
            writer.writerow([page["pdf_page"], page["suspicious_characters"], page["visible_characters"],
                             f"{page['suspicious_ratio']:.4%}", page["cid_marker"],
                             "异常字符比例≥5%", REVIEWS.get(page["pdf_page"], ("未逐页视觉复核",))[0]])

    md = ["# 中文手册乱码核查", "", "原 chunks.jsonl 保持不变；仅生成核查与检索准入清单。", "",
          "## 结论与口径", "", f"105 页全部由异常字符比例≥5%触发，CID 触发为 0 页。涉及原片段 {summary['flagged_page_chunks']} 个。",
          "原规则只计非空白字符中的 U+FFFD、Cc/Co/Cs/Cn，未统计看似正常但映射错误的汉字、拉丁字母，因此 5% 以下不等于安全。",
          "样本按 PDF 实体页的前三分之一（1–142）、中三分之一（143–283）、后三分之一（284–424）各取5页。",
          "各层在标记页序列中等距抽样；不是随机抽样，不将样本比例外推至105页。", "",
          "| 视觉分类 | 页数 |", "|---|---:|"]
    md.extend(f"| {name} | {count} |" for name, count in categories.items())
    md += ["", "图表局部乱码包括重要标签及数值的真实损坏，不是无害的特殊符号；第415页的正文说明/脚注和图表均损坏，单列混合类以避免重复计数。", "",
           "## 105页及逐页触发原因", "", "| PDF页码 | 异常/非空白字符 | 比例 | CID | 视觉分类 |", "|---:|---:|---:|---|---|"]
    for page in evidence["pages"]:
        md.append(f"| {page['pdf_page']} | {page['suspicious_characters']}/{page['visible_characters']} | "
                  f"{page['suspicious_ratio']:.2%} | 无 | {REVIEWS.get(page['pdf_page'], ('未逐页视觉复核',))[0]} |")
    md += ["", "各页逐字符类别及码位计数见 evidence.json；105页原文均在 raw/，未删除或改写原文。", "",
           "## 15页原始文字与页面对照", "", "下面摘录仅为定位，raw/*.txt 保存完整的逐页文本块提取结果（与 build_chunks.py 一致）。",
           "不可见控制字符以 \\uXXXX 显示；这只是展示转义，不是修复。图片为 PDF 直接渲染，未运行 OCR。", ""]
    sections = []
    for number, (category, excerpt, reading, note, crop) in REVIEWS.items():
        with (OUTPUT / "raw" / f"zh_p{number:04d}.txt").open(encoding="utf-8", newline="") as stream:
            raw = stream.read()
        visible = "".join(f"\\u{ord(c):04x}" if unicodedata.category(c) == "Cc" and c != "\n" else c for c in raw)
        md += [f"### PDF第{number}页：{category}", "", f"提取原文定位：`{excerpt}`；页面可读内容：{reading}。", "", note, "",
               f"[完整原文](raw/zh_p{number:04d}.txt) · [转义原文](raw/zh_p{number:04d}.escaped.txt) · [页面图片](images/zh_p{number:04d}.png)", "",
               "```text", visible, "```", ""]
        sections.append(f'<section><h2>PDF 第 {number} 页：{html.escape(category)}</h2><p>{html.escape(note)}</p>'
                        f'<p>对照页面读文：{html.escape(reading)}</p><div class="pair">'
                        f'<a href="images/zh_p{number:04d}.png"><img src="images/zh_p{number:04d}.png" alt="PDF第{number}页"></a>'
                        f'<pre>{html.escape(visible)}</pre></div></section>')
    md += ["## 定向 OCR 方案（未执行）", "", "仅建议处理下面已视觉确认损坏、且包含重要文字或数值的区域；90个未视觉复核页面不自动进入OCR队列。",
           "第18页为目录结构，第52页主要正文与部分图示文字仍可读，本轮不优先OCR。", "",
           "| PDF页码 | 目标区域/重要文字 |", "|---:|---|"]
    md.extend(f"| {entry['pdf_page']} | {entry['important_text']} |" for entry in ocr_plan)
    md += ["", "按 targeted_ocr_plan.json 的归一化坐标裁剪，先核对边界，再以300–400 DPI渲染；仅使用本地中英文OCR。",
           "第168页可另外裁剪操作步骤右侧显示框；第384页除顶部尺寸图还需底部两幅尺寸图。坐标为建议范围，不能当作已精确标注的文字边界。",
           "保留原页码、区域坐标、OCR原文和置信度；逐条核对端子号、正负号、单位、不等号和小数点。接线图不凭OCR文本推断连线关系。",
           "复核后另建版本化修复片段，仅替换损坏区域，保留本来可读的正文；原 chunks.jsonl、原chunk_id来源记录始终保留。", "",
           "## 检索准入计数", "", "不按整页删除：用原片段在页内的位置，与疑似损坏字体文本块求交；有交集即暂缓。",
           "风险字体为含 GBK-EUC-H 的字体、DecoNumbersLH-Circle、CG11seg、Wingdings3；另拒绝未豁免控制/私用/未分配字符及CID标记。",
           "英文第4页的 Wingdings F06C 圆点、第3页的 Wingdings2 F0A3 版权符号已看图确认，不作为正文损坏；只豁免这两个码位。",
           "独立片段另要求至少20个常用区汉字（中文）或12个长度≥2的英文单词（英文），避免只有页眉、代码、数字的片段。",
           "按整个文本块暂缓可能误伤正常文字，这里选择保守保留待核，不进行删除。", "",
           "| 语言 | 可先用于检索 | 暂缓 | 原总数 |", "|---|---:|---:|---:|"]
    for language, count in counts.items():
        md.append(f"| {language} | {count.get('eligible', 0)} | {count.get('hold', 0)} | {sum(count.values())} |")
    md += ["", f"合计可先用于检索：**{summary['eligible_total']}**。这是保守编码质量筛选结果，不是全书逐页语义正确性保证。",
           f"105个标记页内仍有 {summary['eligible_from_flagged_pages']} 个片段通过；未标记中文页另有 {summary['held_outside_flagged_zh_pages']} 个片段暂缓。",
           "每个chunk_id的准入结果和原因均在 retrieval_manifest.jsonl；无需改动原文件，只在后续检索读取时按 eligible 过滤。", "",
           "原文件SHA-256（审核前后相同）：", "", f"`{evidence['original_sha256']}`", ""]
    (OUTPUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    html_page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>PDF图文对照</title><style>body{font-family:system-ui;margin:24px;background:#f7f7f7}section{background:white;padding:20px;margin-bottom:24px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 monospace;margin:0}h1,h2{color:#23384a}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>15页PDF与原始提取文字对照</h1><p>左：直接渲染的PDF页面；右：完整提取文字，控制字符转义显示。未执行OCR。</p>' + "".join(sections) + '</html>'
    (OUTPUT / "review.html").write_text(html_page, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def character_stats(text):
    visible = [c for c in text if not c.isspace()]
    bad = Counter(c for c in visible if c == "\ufffd" or
                  unicodedata.category(c) in {"Cc", "Co", "Cs", "Cn"})
    return {
        "visible_characters": len(visible),
        "suspicious_characters": sum(bad.values()),
        "suspicious_ratio": sum(bad.values()) / max(1, len(visible)),
        "cid_marker": "(cid:" in text,
        "characters": {f"U+{ord(c):04X} ({unicodedata.category(c)})": n
                       for c, n in bad.most_common()},
    }


def main():
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "raw").mkdir(exist_ok=True)
    (OUTPUT / "images").mkdir(exist_ok=True)
    original = BASE_DIR / "output" / "chunks.jsonl"
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    report = json.loads((BASE_DIR / "output" / "chunks_report.json").read_text(encoding="utf-8"))
    warnings = [w for w in report["warnings"] if w["source_file"] == "中文手册内容.pdf"]
    rows = [json.loads(line) for line in original.read_text(encoding="utf-8").splitlines()]
    counts = Counter(r["pdf_page"] for r in rows if r["language"] == "zh")
    page_numbers = sorted(w["pdf_page"] for w in warnings)
    with pymupdf.open(BASE_DIR / "中文手册内容.pdf") as doc:
        groups = [[p for p in page_numbers if (p - 1) * 3 // len(doc) == group]
                  for group in range(3)]
        samples = [group[round(i * (len(group) - 1) / 4)]
                   for group in groups for i in range(5)]
        pages = []
        for warning in warnings:
            number = warning["pdf_page"]
            page = doc[number - 1]
            text = page_text(page)
            raw_path = OUTPUT / "raw" / f"zh_p{number:04d}.txt"
            raw_path.write_text(text, encoding="utf-8", newline="\n")
            # JSON 转义展示控制字符；原始 TXT 完整保留原字符。
            raw_path.with_suffix(".escaped.txt").write_text(
                json.dumps(text, ensure_ascii=False, indent=2), encoding="utf-8")
            pages.append({**warning, **character_stats(text),
                          "chunk_count": counts[number], "sampled": number in samples})
            if number in samples:
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(
                    OUTPUT / "images" / f"zh_p{number:04d}.png")
        evidence = {"original_sha256": digest, "sample_pages": samples,
                    "strata": groups, "pages": pages,
                    "chunk_character_stats": [
                        {"chunk_id": r["chunk_id"], "language": r["language"],
                         "pdf_page": r["pdf_page"], **character_stats(r["text"])}
                        for r in rows]}
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    (OUTPUT / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(evidence, rows)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    print("SHA256:", digest)
    print("Sample pages:", samples)
    print("Flagged pages / chunks:", len(pages), sum(p["chunk_count"] for p in pages))
    for page in pages:
        print(page["pdf_page"], f"{page['suspicious_characters']}/{page['visible_characters']}",
              f"{page['suspicious_ratio']:.2%}", "CID=" + str(page["cid_marker"]))


if __name__ == "__main__":
    main()
