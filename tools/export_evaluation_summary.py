"""Export evaluation metadata only, never extracted manual text."""
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((BASE / "output/retrieval_hybrid/comparison_report.json").read_text(encoding="utf-8"))
    out = BASE / "docs"
    out.mkdir(exist_ok=True)
    (out / "evaluation_summary.json").write_text(json.dumps(report["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 固定20题检索评测", "", "评测对象：实际Top 5的检索依据是否覆盖问题要求，不是最终回答准确率。",
        "20题参与了词表与排序调整，是开发集复测，不是独立测试集。没有调用API作答案质量评测。",
        "", "混合检索：充分15题、部分4题、未找到1题；基线：充分9题、部分2题、未找到9题。",
        "", "| ID | 类别 | 提问语言 | 原结果 | 新结果 | 需看图 | 新Top 5语言:PDF物理页码 |",
        "|---|---|---|---|---|---|---|"]
    for q in report["results"]:
        after, before = q["after"], q["before"]
        pages = ", ".join(f"{h['language']}:{h['pdf_page']}" for h in after["hits"])
        lines.append(f"| {q['id']} | {q['category']} | {q['language']} | {before['assessment']['status']} | {after['assessment']['status']} | {'是' if after['assessment'].get('requires_pdf_image') else '否'} | {pages} |")
    lines += ["", "Q07退步为部分，Q08仍未找到完整依据；Q12/Q15需查看PDF原图，不推断端子对应关系。",
        "Q01新英文155页直接依据排第1；其原中文398页目标仍未进Top 5。Q06中文175页与Q18英文404页目标排第1。",
        "本文件仅保存评测元数据；完整片段报告位于本地output目录，不随仓库发布。", ""]
    (out / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
