"""核验混合检索的数据隔离、固定评测、文字依据和排序边界。"""
from collections import Counter
import json
from html.parser import HTMLParser

import numpy as np

from hybrid_retrieval import HybridIndex, anchor_present, is_navigation
from local_vector_index import BASE, eligible_chunks, read_jsonl, sha256, write_json


def read(path):
    return json.loads((BASE / path).read_text(encoding="utf-8"))


class ReportCounter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.articles = 0
        self.pre = 0

    def handle_starttag(self, tag, attrs):
        self.articles += tag == "article"
        self.pre += tag == "pre"


def main():
    expected = {
        "output/chunks.jsonl": "3e88662390ad7ace0e0546c40a1e1e77e1419cacc2acd5eb6d506c3a5c5bdcc9",
        "output/vector_index/vectors.npy": "a53c4f85f17632c1dfed95e5ecff5bf6b5aba4e1fee7b5f206d04df996472b8a",
        "output/vector_index/metadata.jsonl": "085aa9659966d867b266907cc8caf3d8f214509ce1e6445d1906641d4aed249f",
        "output/retrieval_eval/retrieval_results.json": "6d989972d66067693ec984485e34925cf74847cb0a926164324c0681cee0a9f5",
    }
    for file, digest in expected.items():
        assert sha256(BASE / file) == digest, file + "原文件发生变化"
    info = read("output/vector_index_hybrid/index_info.json")
    for file in ("vectors.npy", "metadata.jsonl"):
        assert sha256(BASE / "output/vector_index_hybrid" / file) == info[file.split('.')[0] + "_sha256"]
    for file, digest in info["source_pdf_sha256"].items():
        assert sha256(BASE / file) == digest
    assert sha256(BASE / "output/chunk_audit/retrieval_manifest.jsonl") == info["manifest_sha256"]
    records = read_jsonl(BASE / "output/vector_index_hybrid/metadata.jsonl")
    lookup = {r["chunk_id"]: r for r in records}
    allowed = eligible_chunks()
    assert len(records) == len(lookup) == len(allowed) == 1596
    assert set(lookup) == {r["chunk_id"] for r in allowed}
    for r in allowed:
        assert all(lookup[r["chunk_id"]][k] == v for k, v in r.items())
    vectors = np.load(BASE / "output/vector_index_hybrid/vectors.npy", allow_pickle=False)
    assert vectors.shape == (1596, 384) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert sum(r["embedding_windows"] for r in records) == 1646
    audit = read("output/retrieval_hybrid/embedding_audit.json")
    assert audit["covered_document_tokens"] == sum(r["token_count"] for r in records)
    assert audit["max_window_tokens"] <= 512 and audit["truncated_tokens"] == 0
    assert all(c["checked"] == c["corrected_tokenization_identical"] for c in audit["prefix_checks"])
    results = read("output/retrieval_hybrid/retrieval_results.json")
    assert results["index_vectors_sha256"] == info["vectors_sha256"]
    assert results["algorithm_source_sha256"] == sha256(BASE / "hybrid_retrieval.py")
    assert results["questions_sha256"] == sha256(BASE / "retrieval_questions.json")
    assert results["language_filter"] is None and results["top_k"] == 5
    review = read("hybrid_assessments.json")
    assert review["results_sha256"] == sha256(BASE / "output/retrieval_hybrid/retrieval_results.json")
    reviews = {q["id"]: q for q in review["assessments"]}
    questions = read("retrieval_questions.json")
    assert len(questions) == len(results["results"]) == len(reviews) == 20
    for q, result in zip(questions, results["results"]):
        assert all(result[k] == v for k, v in q.items())
        hits = result["hits"]
        assert len(hits) == len({h["chunk_id"] for h in hits}) == 5
        assert [h["rank"] for h in hits] == list(range(1, 6))
        assert [h["score"] for h in hits] == sorted([h["score"] for h in hits], reverse=True)
        assert max(Counter((h["source_file"], h["pdf_page"]) for h in hits).values()) <= 2
        assert result["candidate_count"] == len(set(result["candidate_chunk_ids"])) >= 100
        assert set(result["candidate_chunk_ids"]) <= set(lookup)
        for h in hits:
            assert h["chunk_id"] in result["candidate_chunk_ids"]
            assert all(h[k] == v for k, v in lookup[h["chunk_id"]].items())
            assert h["source_language"] == h["language"]
        assert set(reviews[q["id"]]["evidence_chunk_ids"]) <= {h["chunk_id"] for h in hits}
    assert reviews["Q12"]["requires_pdf_image"] and reviews["Q15"]["requires_pdf_image"]
    assert "输入电源(C/D)\nE5CC：11/12" in lookup["zh_p0371_c001"]["text"]
    # 型号必须区分后缀；故障码不能匹配更长编号。
    assert anchor_present("E5CC: power", "e5cc")
    assert not anchor_present("E5CC-U E5CC-B", "e5cc")
    assert anchor_present("e333 AD converter", "e333")
    assert not anchor_present("e3333", "e333")
    assert is_navigation("Input type ........ 12\nAlarm ......... 24\nWiring ........ 35")
    assert not is_navigation("Input type\nin-t\nK\n5\nAlarm\n2")
    # 在真实库验证目录标记存在、默认跨语言候选及显式过滤行为。
    engine = HybridIndex(vectors, records)
    assert engine.navigation.sum() > 0
    query = questions[0]["question"]
    probe = engine.search(query, vectors[0])
    assert {lookup[c]["language"] for c in probe["candidate_chunk_ids"]} == {"zh", "en"}
    filtered = engine.search(query, vectors[0], language="en")
    assert all(h["language"] == "en" for h in filtered["hits"])
    assert all(lookup[c]["language"] == "en" for c in filtered["candidate_chunk_ids"])
    report = read("output/retrieval_hybrid/comparison_report.json")
    assert len(report["results"]) == 20
    assert report["summary"]["after"] == dict(Counter(r["status"] for r in reviews.values()))
    parser = ReportCounter()
    parser.feed((BASE / "output/retrieval_hybrid/comparison_report.html").read_text(encoding="utf-8"))
    assert parser.articles == 20 and parser.pre == 200
    output = {"passed": True, "indexed": len(records), "excluded": 399,
        "navigation_chunks_detected": int(engine.navigation.sum()), "questions": 20,
        "reported_before_and_after_chunks": parser.pre, "summary": report["summary"],
        "checks": ["原始片段/PDF/准入清单/原索引/原检索结果哈希不变", "准入集合与原文元数据一致",
            "向量维度、有限值及归一化", "1596文档/20查询标准前缀与长片段覆盖审计",
            "固定20题与判定要求、100条Top-5文字/来源/页码一致", "依据只来自实际Top-5",
            "型号后缀/故障码精确匹配边界", "目录降权识别、默认跨语言、显式语言过滤",
            "Q12/Q15保留看图标记，Q11编号来自明确文字", "HTML包含20题及200条前后全文"]}
    write_json(BASE / "output/retrieval_hybrid/validation.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
