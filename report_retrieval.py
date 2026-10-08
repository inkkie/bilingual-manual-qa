"""将已经逐题内容复核的判断与原始Top-5检索结果合并。"""
from collections import Counter, defaultdict
import csv
import html
import json

from local_vector_index import BASE, INDEX, sha256, write_json

OUT = BASE / "output" / "retrieval_eval"


def main():
    results_path = OUT / "retrieval_results.json"
    raw = json.loads(results_path.read_text(encoding="utf-8"))
    assessment = json.loads((BASE / "retrieval_assessments.json").read_text(encoding="utf-8"))
    if sha256(results_path) != assessment["results_sha256"]:
        raise ValueError("检索结果已变化，需重新逐题核对，不能复用旧判断")
    if sha256(BASE / "retrieval_questions.json") != raw["questions_sha256"]:
        raise ValueError("测试问题已变化，需重新检索和评定")
    reviews = {r["id"]: r for r in assessment["assessments"]}
    if len(reviews) != 20 or set(reviews) != {q["id"] for q in raw["results"]}:
        raise ValueError("评定与20题不匹配")
    combined = []
    for question in raw["results"]:
        review = reviews[question["id"]]
        if review["status"] not in {"充分", "部分", "未找到"}:
            raise ValueError("无效评定")
        hit_ids = {h["chunk_id"] for h in question["hits"]}
        if not set(review["evidence_chunk_ids"]).issubset(hit_ids):
            raise ValueError("依据必须来自本题实际Top-5")
        if (review["status"] != "未找到") != bool(review["evidence_chunk_ids"]):
            raise ValueError("依据列表与判断不一致")
        combined.append({**question, "assessment": {**review,
                         "found_effective_evidence": review["status"] == "充分",
                         "has_partial_evidence": review["status"] == "部分"}})
    stats = Counter(q["assessment"]["status"] for q in combined)
    breakdown = {}
    for key in ("category", "language"):
        groups = defaultdict(Counter)
        for q in combined:
            groups[q[key]][q["assessment"]["status"]] += 1
        breakdown[key] = {k: dict(v) for k, v in groups.items()}
    summary = {"questions": 20, "top_k": raw["top_k"], "status_counts": dict(stats),
               "full_evidence_rate": stats["充分"] / 20,
               "partial_or_full_rate": (stats["充分"] + stats["部分"]) / 20,
               "breakdown": breakdown, "review_method": assessment["review_method"],
               "meaning": "是否找到有效依据=true仅表示Top-5覆盖本题全部关键要求；部分依据单列。未找到不表示整本手册没有答案。",
               "evaluation_scope": "20题本地冒烟评测，不是独立标注的大规模基准；未测全库Recall。"}
    write_json(OUT / "evaluation_report.json", {"summary": summary, "retrieval_config": {
        k: v for k, v in raw.items() if k != "results"}, "results": combined})
    write_json(OUT / "evaluation_summary.json", summary)

    md = ["# 本地向量检索：20题结果与依据复核", "",
          "索引仅含当前准入清单通过的1596个片段（中文276、英文1320），399个暂缓片段未入库。原始PDF、chunks.jsonl及准入清单未改动。",
          "所有页码均为PDF物理页码（从1开始），不是手册印刷页码。每题同时检索中英文，精确余弦Top-5，不翻译、不扩展问题、不重排。", "",
          "## 评定口径", "", assessment["review_method"],
          "充分：检索结果覆盖题目全部关键要求；部分：有直接依据但缺少关键条件；未找到：无有效直接依据，目录/主题相似不算。",
          "布尔值found_effective_evidence只在充分时为true。判定只使用本题实际Top-5，不把其他题或额外搜索的结果补算为命中。",
          "分数不是正确概率。未找到只表示本次Top-5未提供有效依据，不证明手册或索引里没有答案。", "",
          f"充分：{stats['充分']}/20；部分：{stats['部分']}/20；未找到：{stats['未找到']}/20。充分依据率{summary['full_evidence_rate']:.0%}，含部分依据为{summary['partial_or_full_rate']:.0%}。", "",
          "| 分类 | 充分 | 部分 | 未找到 |", "|---|---:|---:|---:|"]
    for category, counts in breakdown["category"].items():
        md.append(f"| {category} | {counts.get('充分', 0)} | {counts.get('部分', 0)} | {counts.get('未找到', 0)} |")
    md += ["", "## 逐题汇总", "", "| ID | 语言 | 测试问题 | 实际检索PDF页码（排名顺序） | 有效依据 |", "|---|---|---|---|---|"]
    csv_rows = []
    cards = []
    for q in combined:
        review = q["assessment"]
        pages = "; ".join(f"{h['language']}:{h['pdf_page']}" for h in q["hits"])
        md.append(f"| {q['id']} | {q['language']} | {q['question']} | {pages} | {review['status']} |")
        csv_rows.append([q["id"], q["category"], q["language"], q["question"], review["status"],
                         review["found_effective_evidence"], "; ".join(review["evidence_chunk_ids"]), pages,
                         "; ".join(h["chunk_id"] for h in q["hits"]), review["reason"]])
    md += ["", "## 每题完整片段与判断", ""]
    for q in combined:
        review = q["assessment"]
        md += [f"### {q['id']} · {q['category']} · {q['language']} · {review['status']}", "", q["question"], "",
               "判定要求：" + q["evidence_requirement"], "", "判断：" + review["reason"], "",
               "有效/部分依据片段：" + ("、".join(review["evidence_chunk_ids"]) or "无"), ""]
        details = []
        for hit in q["hits"]:
            selected = hit["chunk_id"] in review["evidence_chunk_ids"]
            label = f"#{hit['rank']} · {hit['chunk_id']} · {hit['source_file']} · PDF第{hit['pdf_page']}页 · 相似度{hit['score']:.6f}"
            if selected:
                label += " · 依据片段"
            md += [label, "", "```text", hit["text"], "```", ""]
            details.append(f'<details{" open" if selected else ""}><summary>{html.escape(label)}</summary><pre>{html.escape(hit["text"])}</pre></details>')
        cards.append(f'<article data-lang="{q["language"]}" data-status="{review["status"]}" data-category="{q["category"]}">'
                     f'<h2>{q["id"]} · {q["category"]} · {q["language"]} · {review["status"]}</h2>'
                     f'<h3>{html.escape(q["question"])}</h3><p>判定要求：{html.escape(q["evidence_requirement"])}</p>'
                     f'<p><strong>复核：</strong>{html.escape(review["reason"])}</p>' + "".join(details) + '</article>')
    md += ["## 观察到的限制", "",
           "- 中文仅276个片段，且目录/附录文字占据部分高排名结果；本次中文题充分2题、部分1题、未找到7题。",
           "- Q01、Q06、Q11、Q18有可定位的库内相关依据，却未进入本题Top-5，说明不能将失败全归因于待OCR页面。",
           "- Q12的接线图即使文字通过编码筛选，线性文本仍无法可靠保留连线关系。",
           "- Q14能区分端子螺钉和安装螺钉的扭矩；Q18与Q20没有把相邻错误或相邻表格行误算为有效依据。",
           "- 当前是未加混合检索、重排或跨语言查询扩展的向量基线。样本少，不能将充分依据率当作全库准确率。", ""]
    (OUT / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    with (OUT / "evaluation_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["question_id", "category", "language", "question", "evidence_status", "found_effective_evidence",
                         "evidence_chunk_ids", "retrieved_pdf_pages", "retrieved_chunk_ids", "review_reason"])
        writer.writerows(csv_rows)
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>20题本地检索评测</title>
<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:28px auto;padding:0 20px;background:#f4f6f8;color:#172b3a}article{background:white;padding:22px;margin:20px 0;border-radius:10px}h2{font-size:20px}h3{font-size:18px}summary{cursor:pointer;font-weight:600;padding:12px;background:#edf2f7}details{margin:10px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 monospace;padding:16px;border:1px solid #ddd}nav{position:sticky;top:0;background:#f4f6f8;padding:15px 0}select,input{padding:8px;margin-right:8px}#count{font-weight:bold}</style>
<h1>20题本地检索评测</h1><p>索引1596个片段，Top-5精确向量检索。页码均为PDF物理页码。所有命中均保留完整原文；相似度不是正确概率。</p>'''
    page += f'<p>充分依据 {stats["充分"]} 题 · 部分依据 {stats["部分"]} 题 · 未找到 {stats["未找到"]} 题。没有调用推理/裁判API或执行OCR。</p>'
    page += '''<nav><select id="lang"><option value="">全部语言</option><option>zh</option><option>en</option></select><select id="status"><option value="">全部结果</option><option>充分</option><option>部分</option><option>未找到</option></select><select id="category"><option value="">全部类别</option><option>参数设置</option><option>报警</option><option>接线</option><option>故障排查</option></select><input id="search" placeholder="搜索问题或片段"><span id="count"></span></nav>'''
    page += "".join(cards)
    page += '''<script>const fields=['lang','status','category'];function filter(){let n=0;const search=document.getElementById('search').value.toLowerCase();document.querySelectorAll('article').forEach(a=>{const ok=fields.every(k=>!document.getElementById(k).value||a.dataset[k]===document.getElementById(k).value)&&a.textContent.toLowerCase().includes(search);a.hidden=!ok;if(ok)n++});document.getElementById('count').textContent=n+' / 20 题'}document.querySelectorAll('select,input').forEach(e=>e.addEventListener('input',filter));filter();</script></html>'''
    (OUT / "evaluation_report.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
