"""固定20题的混合检索前后对比；评判绑定实际结果哈希。"""
from collections import Counter
import csv
import html
import json

from local_vector_index import BASE, sha256, write_json

OUT = BASE / "output/retrieval_hybrid"
LEVEL = {"未找到": 0, "部分": 1, "充分": 2}


def read(path):
    return json.loads((BASE / path).read_text(encoding="utf-8"))


def main():
    current = read("output/retrieval_hybrid/retrieval_results.json")
    baseline = read("output/retrieval_eval/retrieval_results.json")
    review = read("hybrid_assessments.json")
    old_review = read("retrieval_assessments.json")
    audit = read("output/retrieval_hybrid/embedding_audit.json")
    questions = read("retrieval_questions.json")
    assert sha256(OUT / "retrieval_results.json") == review["results_sha256"], "结果变化，需重新逐题核对"
    assert sha256(BASE / "output/retrieval_eval/retrieval_results.json") == old_review["results_sha256"] == current["baseline_sha256"]
    assert sha256(BASE / "retrieval_questions.json") == current["questions_sha256"] == baseline["questions_sha256"]
    old = {q["id"]: q for q in baseline["results"]}
    new = {q["id"]: q for q in current["results"]}
    reviews = {r["id"]: r for r in review["assessments"]}
    old_reviews = {r["id"]: r for r in old_review["assessments"]}
    assert len(questions) == len(old) == len(new) == len(reviews) == len(old_reviews) == 20
    before_counts, after_counts, changes = Counter(), Counter(), Counter()
    combined = []
    for q in questions:
        qid = q["id"]
        for result in (old[qid], new[qid]):
            assert all(result[k] == v for k, v in q.items()), "问题或判定要求不能改动"
            assert len(result["hits"]) == 5
        r, previous = reviews[qid], old_reviews[qid]
        assert r["status"] in LEVEL
        assert set(r["evidence_chunk_ids"]) <= {h["chunk_id"] for h in new[qid]["hits"]}
        assert bool(r["evidence_chunk_ids"]) == (r["status"] != "未找到")
        change = "改善" if LEVEL[r["status"]] > LEVEL[previous["status"]] else "退步" if LEVEL[r["status"]] < LEVEL[previous["status"]] else "不变"
        before_counts[previous["status"]] += 1
        after_counts[r["status"]] += 1
        changes[change] += 1
        combined.append({**q, "change": change,
            "before": {"assessment": previous, "hits": old[qid]["hits"]},
            "after": {"assessment": {**r, "found_effective_evidence": r["status"] == "充分",
                "requires_pdf_image": r.get("requires_pdf_image", False)},
                "hits": new[qid]["hits"], "candidate_count": new[qid]["candidate_count"]}})
    summary = {"questions": 20, "top_k": 5, "indexed_chunks": 1596,
        "before": dict(before_counts), "after": dict(after_counts), "changes": dict(changes),
        "before_full_evidence_rate": before_counts["充分"] / 20,
        "after_full_evidence_rate": after_counts["充分"] / 20,
        "requires_pdf_image": [q["id"] for q in combined if q["after"]["assessment"]["requires_pdf_image"]],
        "review_method": review["review_method"],
        "scope": "同一20题开发集复测，词表/排序调整参考了这些问题，不是独立测试集；不代表全库准确率。"}
    notes = [
        "沿用原20题及全部判定要求。充分才表示找到有效依据；部分单列。判断只依据本题实际Top-5，目录或相似主题不算答案。",
        "仍只有1596条通过筛选的片段（zh 276、en 1320），399条损坏/待复核片段未入库。原始chunks、PDF、原索引及基线报告均保留。全程离线，没有OCR或DeepSeek/其他模型API。",
        "查询使用query:，文档使用passage:。原实现单独分词带尾空格前缀，多出空白token；已修正并另建派生向量。1596条文档和20个查询均验证与整句分词一致。",
        "50条长片段用64-token重叠分窗，共1646窗口；315219个正文token完整覆盖，最长窗口512 tokens，无静默截断。一片段仍对应一个加权聚合向量，细节信号稀释仍可能发生。",
        "默认没有语言过滤。每路取100条向量/BM25/精确标识符候选，并补入每种语言各50条向量和BM25候选，合并去重后排序。双语术语表扩展查询，BM25按语言归一化，融合向量秩、关键词秩、术语覆盖及参数/故障码匹配。",
        "目录/索引采用点引线密度检测，排序分数乘0.35；没有删除页面，也不把包含参数说明的附录一律当目录。每页最多2条进入Top-5，结果显式标注来源语言。检测是启发式，未识别的无点引线目录可能残留。",
        "Q01旧目标zh_p0398_c001：原向量第18、修正前缀后仍第18；进入混合候选但仍未进Top-5。新的第1名en_p0155_c001（英文PDF155页）直接提供完整依据。",
        "Q06目标zh_p0175_c002：原向量第6、修正后第6、混合第1。Q18目标en_p0404_c003：原向量第193、修正后第209、混合第1。改善主要来自混合召回/排序，不能归功于前缀修正 alone。",
        "Q07从充分退为部分：偏差上限说明退出Top-5。Q08仍未找到完整待机序列说明。Q20仅部分支持检查方向，不能从AT参数不显示的条目推断加热输出OFF的原因。",
        "Q12、Q15：需查看PDF原图，未推断接线关系。Q11：新Top-5有逐型号文字明确输入电源端子对，以及E5CC螺丝端子机型说明，因此判为充分；不推断额外极性、电压或其他机型接法。",
        summary["scope"],
    ]
    notes[7] = notes[7].replace("前缀修正 alone", "前缀修正本身")
    write_json(OUT / "comparison_report.json", {"summary": summary, "notes": notes,
        "embedding_audit": audit, "retrieval_config": {k: v for k, v in current.items() if k != "results"}, "results": combined})
    write_json(OUT / "evaluation_summary.json", summary)
    md = ["# 20题混合检索前后对比", "", f"充分 {before_counts['充分']} → {after_counts['充分']}；部分 {before_counts['部分']} → {after_counts['部分']}；未找到 {before_counts['未找到']} → {after_counts['未找到']}。", ""]
    md += ["- " + n for n in notes]
    md += ["", "所有页码为从1开始的PDF物理页码。融合分数与旧余弦分数不可直接比较，均不是正确概率。", "",
        "| 题号 | 原结果 | 新结果 | 变化 | 新Top-5来源语言:PDF页码 |", "|---|---|---|---|---|"]
    cards, rows, table = [], [], []
    for q in combined:
        r = q["after"]["assessment"]
        status = r["status"] + ("；需查看 PDF 原图" if r["requires_pdf_image"] else "")
        pages = "; ".join(f"{h['language']}:{h['pdf_page']}" for h in q["after"]["hits"])
        md.append(f"| {q['id']} | {q['before']['assessment']['status']} | {status} | {q['change']} | {pages} |")
        table.append(f'<tr><td><a href="#{q["id"]}">{q["id"]}</a></td><td>{q["before"]["assessment"]["status"]}</td><td>{status}</td><td>{q["change"]}</td><td>{pages}</td></tr>')
        rows.append([q["id"], q["question"], q["evidence_requirement"], q["before"]["assessment"]["status"], r["status"], q["change"], r["found_effective_evidence"], r["requires_pdf_image"], pages, "; ".join(r["evidence_chunk_ids"]), r["reason"]])
    md += ["", "## 逐题Top-5全文与前后判断", ""]
    for q in combined:
        md += [f"### {q['id']} · {q['category']} · {q['language']} · {q['change']}", "", q["question"], "", "判定要求：" + q["evidence_requirement"], ""]
        columns = []
        for name, title in (("before", "原向量检索"), ("after", "混合检索")):
            group = q[name]
            r = group["assessment"]
            status = r["status"] + ("；需查看 PDF 原图" if r.get("requires_pdf_image") else "")
            md += [f"#### {title}：{status}", "", r["reason"], ""]
            details = []
            for h in group["hits"]:
                selected = h["chunk_id"] in r["evidence_chunk_ids"]
                label = f"#{h['rank']} · {h['source_file']} · 来源语言 {h['language']} · PDF {h['pdf_page']} · {h['chunk_id']} · {'融合分数' if name == 'after' else '余弦分数'} {h['score']:.6f}"
                if selected:
                    label += " · 依据片段"
                diagnostics = ""
                if name == "after":
                    diagnostics = f"向量秩 {h['dense_rank']}；关键词秩 {h['keyword_rank']}；BM25 {h['bm25_score']:.3f}；目录系数 {h['navigation_penalty']}；标识符 {', '.join(h['matched_anchors']) or '无'}"
                md += [label, "", diagnostics, "", "```text", h["text"], "```", ""]
                pdf_link = "../../" + h["source_file"] + "#page=" + str(h["pdf_page"])
                details.append(f'<details{" open" if selected else ""}><summary>{html.escape(label)}</summary><p>{html.escape(diagnostics)} <a href="{html.escape(pdf_link, quote=True)}">打开PDF本页</a></p><pre>{html.escape(h["text"])}</pre></details>')
            columns.append(f'<section><h3>{title}：{status}</h3><p>{html.escape(r["reason"])}</p>' + "".join(details) + '</section>')
        cards.append(f'<article id="{q["id"]}" data-change="{q["change"]}" data-status="{q["after"]["assessment"]["status"]}"><h2>{q["id"]} · {q["category"]} · 提问语言 {q["language"]} · {q["change"]}</h2><p><strong>{html.escape(q["question"])}</strong></p><p>原判定要求：{html.escape(q["evidence_requirement"])}</p><p>本次合并候选 {q["after"]["candidate_count"]} 条；按同一口径复核Top-5。</p><div class="columns">' + "".join(columns) + '</div></article>')
    (OUT / "comparison_report.md").write_text("\n".join(md), encoding="utf-8")
    with (OUT / "comparison_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "question", "evidence_requirement", "before", "after", "change", "found_effective_evidence", "requires_pdf_image", "new_top5_language_pdf_pages", "evidence_chunk_ids", "reason"])
        writer.writerows(rows)
    page = '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>20题混合检索前后对比</title>
<style>body{font:16px/1.65 system-ui;margin:24px auto;max-width:1560px;padding:0 20px;background:#f4f6f8;color:#172b3a}article{background:#fff;padding:20px;margin:24px 0;border-radius:10px;scroll-margin-top:90px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}section{min-width:0}summary{cursor:pointer;background:#edf2f7;padding:10px;overflow-wrap:anywhere}details{margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;font:14px/1.6 monospace;border:1px solid #ddd}nav{position:sticky;top:0;padding:12px;background:#f4f6f8}select,input{padding:9px}td,th{border-bottom:1px solid #ccd;padding:8px;text-align:left}table{border-collapse:collapse;width:100%}.table{overflow:auto}@media(max-width:900px){.columns{grid-template-columns:1fr}body{padding:0 10px}}</style></head><body><h1>20题混合检索前后对比</h1>'''
    page += f'<p><strong>充分依据：{before_counts["充分"]}/20 → {after_counts["充分"]}/20（45% → {after_counts["充分"]/20:.0%}）；部分：{before_counts["部分"]} → {after_counts["部分"]}；未找到：{before_counts["未找到"]} → {after_counts["未找到"]}。</strong></p>'
    page += '<details><summary>实现、前缀/截断检查与评测限制</summary><ul>' + ''.join('<li>' + html.escape(n) + '</li>' for n in notes) + '</ul></details>'
    page += '<p>所有页码均为PDF物理页码。左右完整展示原/新Top-5；来源语言独立于提问语言。分数不是正确概率，两种分数不可直接比较。</p>'
    page += '<div class="table"><table><thead><tr><th>题号</th><th>原结果</th><th>新结果</th><th>变化</th><th>新Top-5语言:PDF页码</th></tr></thead><tbody>' + ''.join(table) + '</tbody></table></div>'
    page += '''<nav><select id="change"><option value="">全部变化</option><option>改善</option><option>退步</option><option>不变</option></select> <select id="status"><option value="">全部新结果</option><option>充分</option><option>部分</option><option>未找到</option></select> <input id="search" placeholder="搜索问题或片段"> <span id="count"></span></nav>'''
    page += ''.join(cards)
    page += '''<script>function filter(){let n=0;const s=document.getElementById('search').value.toLowerCase();document.querySelectorAll('article').forEach(a=>{const ok=['change','status'].every(k=>!document.getElementById(k).value||document.getElementById(k).value===a.dataset[k])&&a.textContent.toLowerCase().includes(s);a.hidden=!ok;if(ok)n++});document.getElementById('count').textContent=n+' / 20题'}document.querySelectorAll('input,select').forEach(e=>e.addEventListener('input',filter));filter();</script></body></html>'''
    (OUT / "comparison_report.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
