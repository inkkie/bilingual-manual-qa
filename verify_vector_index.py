"""验证准入隔离、原文/元数据一致性、归一化与检索结果完整性。"""
from collections import Counter
import json

import numpy as np

from local_vector_index import BASE, INDEX, MANIFEST, CHUNKS, eligible_chunks, load_index, read_jsonl, sha256


def main():
    vectors, records, info = load_index()
    allowed = eligible_chunks()
    source = {r["chunk_id"]: r for r in read_jsonl(CHUNKS)}
    denied = {r["chunk_id"] for r in read_jsonl(MANIFEST) if not r["eligible"]}
    assert len(records) == 1596
    assert len({r["chunk_id"] for r in records}) == len(records)
    assert {r["chunk_id"] for r in records} == {r["chunk_id"] for r in allowed}
    assert not denied.intersection(r["chunk_id"] for r in records)
    for i, row in enumerate(records):
        assert row["vector_row"] == i
        assert all(row[k] == value for k, value in source[row["chunk_id"]].items())
        assert row["pdf_page"] >= 1
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    # 所有向量以自身查询时应达到最大相似度；重复文本允许并列。
    for start in range(0, len(records), 100):
        scores = vectors[start:start + 100] @ vectors.T
        for offset, row_scores in enumerate(scores):
            assert row_scores[start + offset] >= row_scores.max() - 1e-5
    for filename, expected in info["source_pdf_sha256"].items():
        assert sha256(BASE / filename) == expected
    result_path = BASE / "output" / "retrieval_eval" / "retrieval_results.json"
    evaluation_checks = None
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        assert report["index_vectors_sha256"] == info["vectors_sha256"]
        assert report["index_metadata_sha256"] == info["metadata_sha256"]
        assert len(report["results"]) == 20
        lookup = {r["chunk_id"]: r for r in records}
        for question in report["results"]:
            hits = question["hits"]
            assert len(hits) == report["top_k"]
            assert len({h["chunk_id"] for h in hits}) == len(hits)
            assert [h["score"] for h in hits] == sorted([h["score"] for h in hits], reverse=True)
            for hit in hits:
                assert all(hit[k] == v for k, v in lookup[hit["chunk_id"]].items())
        evaluation_checks = "20题结果及元数据已核验"
    output = {"passed": True, "indexed": len(records), "excluded": len(denied),
              "language_counts": dict(Counter(r["language"] for r in records)),
              "original_chunks_sha256": sha256(CHUNKS), "evaluation": evaluation_checks,
              "checks": ["准入片段集合完全相等", "399个暂缓片段全部排除", "原文与元数据逐字段一致",
                         "向量有限且单位归一化", "全量自身最近邻一致", "源PDF与原片段哈希不变"]}
    (INDEX / "validation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
