"""全量逐页分段：本地文字提取，不执行 OCR 或调用 API。"""

import json
import random
import re

from check_pdf import BASE_DIR, page_warning, pymupdf


SOURCES = (("中文手册内容.pdf", "zh"), ("英文手册内容.pdf", "en"))
MIN_SIZE = 500
MAX_SIZE = 800
TARGET_SIZE = 700
OVERLAP = 100


def page_text(page) -> str:
    # PDF 往往没有空行：使用文本块近似自然段，保留块内原始换行。
    blocks = page.get_text("blocks", sort=True)
    return "\n\n".join(block[4].strip() for block in blocks
                       if block[6] == 0 and block[4].strip())


def split_page(text: str) -> list[str]:
    """优先在自然段末尾切分，超长段落退到句末、空白或字符边界。

    邻接片段重叠约 100 字符；短页面及页尾允许少于 500 字符。
    所有片段都是本页规范化文本的连续子串，不跨页。
    """
    boundaries = [
        [m.end() for m in re.finditer(r"\n\n", text)],
        [m.end() for m in re.finditer(r"[。！？.!?](?:[\s]|$)", text)],
        [m.end() for m in re.finditer(r"\s+", text)],
    ]
    chunks = []
    start = 0
    while start < len(text):
        if len(text) - start <= MAX_SIZE:
            end = len(text)
        else:
            end = start + TARGET_SIZE
            for positions in boundaries:
                candidates = [p for p in positions
                              if start + MIN_SIZE <= p <= start + MAX_SIZE]
                if candidates:
                    end = min(candidates, key=lambda p: abs(p - start - TARGET_SIZE))
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        # 尽量从词边界开始重叠，避免英文从词中间开始。
        next_start = end - OVERLAP
        nearby = [p for p in boundaries[2]
                  if end - OVERLAP - 30 <= p <= end - OVERLAP + 30]
        if nearby:
            next_start = min(nearby, key=lambda p: abs(p - next_start))
        start = max(start + 1, next_start)
    return chunks


def main() -> None:
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    records = []
    summaries = []
    warnings = []
    for source_file, language in SOURCES:
        blank_pages = []
        count = 0
        with pymupdf.open(BASE_DIR / source_file) as document:
            if document.needs_pass:
                raise ValueError(f"{source_file} 已加密，需要密码")
            for index, page in enumerate(document):
                text = page_text(page)
                warning = page_warning(text)
                if warning:
                    warnings.append({"source_file": source_file,
                                     "pdf_page": index + 1, "warning": warning})
                    print(f"提示：{source_file} 第 {index + 1} 页：{warning}")
                if not text.strip():
                    blank_pages.append(index + 1)
                    continue
                for number, chunk in enumerate(split_page(text), start=1):
                    records.append({
                        "source_file": source_file,
                        "language": language,
                        "pdf_page": index + 1,
                        "chunk_id": f"{language}_p{index + 1:04d}_c{number:03d}",
                        "text": chunk,
                    })
                    count += 1
            summaries.append({"source_file": source_file, "language": language,
                              "total_pages": len(document),
                              "blank_page_count": len(blank_pages),
                              "blank_pages": blank_pages, "chunk_count": count})

    target = output_dir / "chunks.jsonl"
    with target.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 从实际落盘数据验证字段、唯一 ID、字符上限及来源页文字。
    loaded = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert loaded == records
    assert len({r["chunk_id"] for r in loaded}) == len(loaded)
    for source_file, language in SOURCES:
        with pymupdf.open(BASE_DIR / source_file) as document:
            texts = {}
            for record in loaded:
                if record["source_file"] != source_file:
                    continue
                page_number = record["pdf_page"]
                assert record["language"] == language
                assert 1 <= page_number <= len(document)
                if page_number not in texts:
                    texts[page_number] = page_text(document[page_number - 1])
                assert 0 < len(record["text"]) <= MAX_SIZE
                assert record["text"] in texts[page_number]

    # 固定种子随机抽样，便于重复核对；完整样本同时落盘。
    rng = random.Random(42)
    samples = []
    for _, language in SOURCES:
        candidates = [r for r in loaded if r["language"] == language]
        samples.extend(rng.sample(candidates, min(3, len(candidates))))
    report = {"sources": summaries, "warnings": warnings,
              "validation": "全部片段已核验：UTF-8 JSONL、唯一 ID、长度和来源页文字",
              "sample_seed": 42, "samples": samples}
    (output_dir / "chunks_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for summary in summaries:
        print(f"\n{summary['source_file']}：总页数 {summary['total_pages']}，"
              f"空白/无文字页 {summary['blank_page_count']}，片段数 {summary['chunk_count']}")
    for sample in samples:
        print(f"\n--- {sample['chunk_id']} | {sample['source_file']} | "
              f"PDF 第 {sample['pdf_page']} 页 | {len(sample['text'])} 字符 ---")
        print(sample["text"])
    print(f"\n{report['validation']}\n已保存：{target}")


if __name__ == "__main__":
    main()
