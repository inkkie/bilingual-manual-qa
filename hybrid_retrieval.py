"""离线混合检索：双语词表、BM25、向量候选融合及目录降权。"""
import argparse
from collections import Counter
import json
import re
import unicodedata
from datetime import datetime, timezone
import time

import numpy as np

from local_vector_index import BASE, LocalEmbedder, load_index, sha256, write_json

OUT = BASE / "output" / "retrieval_hybrid"
# 通用手册术语，不含题号、答案、页码或chunk_id。仅扩展查询，不改变测试问题。
CONCEPTS = [
    ["输入类型", "input type", "in-t"], ["热电偶", "thermocouple"],
    ["k型", "k 型", "type k", "k thermocouple"],
    ["初始设定", "initial setting"], ["温度单位", "temperature unit", "celsius", "fahrenheit", "d-u"],
    ["设定点", "set point", "setpoint"], ["sp上限", "sp 上限", "set point upper limit", "sl-h"],
    ["sp下限", "sp 下限", "set point lower limit", "sl-l"],
    ["限制", "limiter", "limit"], ["sp斜坡", "sp 斜坡", "sp ramp", "sprt"],
    ["每分钟", "per minute", "time unit", "时间单位", "斜坡时间单位"],
    ["自动调节", "auto-tuning", "autotuning", "100% at", "at-2"],
    ["偏差上限", "deviation upper-limit"],
    ["绝对值上限", "absolute-value upper-limit"], ["报警值", "alarm value"],
    ["待机序列", "standby sequence"], ["报警闩锁", "latched alarm", "alarm latch"],
    ["加热器断线", "heater burnout", "failed heater"], ["电流", "current"],
    ["电源", "power supply", "input power"], ["螺丝端子", "screw terminal"],
    ["铂电阻", "platinum resistance thermometer", "resistance thermometer"],
    ["三线", "three-wire", "three lead wires"], ["补偿导线", "compensating wires"],
    ["紧固力矩", "tightening torque", "torque"], ["接线", "wiring", "wire", "connected"],
    ["通信", "communications", "rs-485"], ["输入异常", "输入错误", "input error", "s.err"],
    ["存储器错误", "memory error", "e111"], ["转换器错误", "a/d", "ad converter", "e333"],
    ["保护", "protection", "protect"], ["不能修改", "cannot change", "setting change protect"],
    ["加热输出", "heating output"], ["停止", "stopped", "run/stop"],
    ["手动", "manual"], ["输出分配", "output assignment"],
]
STOP = set("how do i the a an to and or of for is are be what which when it in on from with can should does then if has have its at by as my me this that between why into after before".split())
PARAMETER_CODES = {"输入类型": ["in-t"], "温度单位": ["d-u"],
                   "sp上限": ["sl-h"], "sp下限": ["sl-l"], "sp斜坡": ["sprt", "spru"],
                   "自动调节": ["at-2"], "报警闩锁": ["a1lt", "lat"],
                   "不能修改": ["wtpt", "oapt"], "待机序列": ["rest"]}


def normalize(text):
    return unicodedata.normalize("NFKC", text).lower().replace("−", "-").replace("–", "-")


def compact(text):
    return re.sub(r"\s+", "", normalize(text))


def tokens(text):
    text = normalize(text)
    result = [t for t in re.findall(r"[a-z]+(?:[./-][a-z0-9]+)*|\d+(?:\.\d+)?", text) if t not in STOP]
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        result.extend(run[i:i+2] for i in range(len(run)-1))
    return result


def phrase_present(text, phrase):
    if re.search(r"[\u4e00-\u9fff]", phrase):
        return compact(phrase) in compact(text)
    # 保留词边界，防止 at、on 等短串匹配任意单词。
    pattern = r"(?<![a-z0-9])" + re.escape(normalize(phrase)).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, normalize(text)) is not None


def is_navigation(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dotted = sum(bool(re.search(r"(?:\.\s*){4,}|…{2,}|\.{4,}", line)) for line in lines)
    return dotted >= 3 and dotted / max(1, len(lines)) >= .18


def anchor_present(text, anchor):
    if anchor.startswith("e5"):
        return re.search(r"(?<![a-z0-9-])" + re.escape(anchor) + r"(?![a-z0-9-])", normalize(text)) is not None
    return phrase_present(text, anchor)


def anchors(query):
    q = normalize(query)
    # 传感器单字母类型单独处理；其余为故障码、型号、通信协议等。
    result = re.findall(r"\b(?:e\d{3}|s\.err|e5[a-z]+(?:-[a-z]+)?|rs-485)\b", q)
    if re.search(r"\bk\s*型|\btype\s+k\b|\bk\s+thermocouple", q):
        result.append("k")
    return list(dict.fromkeys(result))


class HybridIndex:
    def __init__(self, vectors, records):
        self.vectors, self.records = vectors, records
        self.texts = [r["text"] for r in records]
        self.counts = [Counter(tokens(t)) for t in self.texts]
        self.lengths = np.array([sum(c.values()) for c in self.counts], dtype=float)
        self.average_length = self.lengths.mean()
        df = Counter(term for c in self.counts for term in c)
        self.idf = {term: np.log(1 + (len(records) - n + .5) / (n + .5)) for term, n in df.items()}
        self.navigation = np.array([is_navigation(t) for t in self.texts])
        self.concept_matches = np.array([[any(phrase_present(t, alias) for alias in concept)
                                         for concept in CONCEPTS] for t in self.texts], dtype=float)

    def query_spec(self, query):
        active = [i for i, concept in enumerate(CONCEPTS) if any(phrase_present(query, a) for a in concept)]
        expansion = " ".join(alias for i in active for alias in CONCEPTS[i])
        return active, expansion, anchors(query)

    def bm25(self, query, expansion):
        original_terms = set(tokens(query))
        expanded_terms = set(tokens(expansion))
        scores = np.zeros(len(self.records))
        norm = 1.2 * (.25 + .75 * self.lengths / self.average_length)
        for term in original_terms | expanded_terms:
            tf = np.array([c.get(term, 0) for c in self.counts], dtype=float)
            weight = 1.0 if term in original_terms else .8
            scores += weight * self.idf.get(term, 0) * tf * 2.2 / (tf + norm)
        return scores

    def search(self, query, vector, top_k=5, language=None, candidate_k=100):
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("需要 candidate_k >= top_k >= 1")
        active, expansion, requested = self.query_spec(query)
        dense = self.vectors @ vector
        sparse = self.bm25(query, expansion)
        permitted = np.array([i for i, r in enumerate(self.records) if language is None or r["language"] == language], dtype=int)
        dense_order = permitted[np.argsort(-dense[permitted], kind="stable")]
        sparse_order = permitted[np.argsort(-sparse[permitted], kind="stable")]
        exact = np.array([sum(anchor_present(t, anchor) for anchor in requested) / max(1, len(requested)) for t in self.texts])
        exact_order = [int(i) for i in sparse_order if exact[i] > 0][:candidate_k]
        candidates = set(dense_order[:candidate_k]) | set(sparse_order[:candidate_k]) | set(exact_order)
        local_sparse_ranks, sparse_normalized = {}, {}
        for lang in {self.records[i]["language"] for i in permitted}:
            language_sparse = [int(i) for i in sparse_order if self.records[i]["language"] == lang]
            language_dense = [int(i) for i in dense_order if self.records[i]["language"] == lang]
            candidates.update(language_sparse[:candidate_k // 2])
            candidates.update(language_dense[:candidate_k // 2])
            maximum = max((sparse[i] for i in language_sparse), default=1) or 1
            for rank, i in enumerate(language_sparse, 1):
                local_sparse_ranks[i] = rank
                sparse_normalized[i] = float(sparse[i] / maximum)
        candidates = sorted(candidates)
        dense_rank = {int(i): rank + 1 for rank, i in enumerate(dense_order)}
        sparse_rank = {int(i): rank + 1 for rank, i in enumerate(sparse_order)}
        scored = []
        for i in candidates:
            coverage = float(self.concept_matches[i, active].mean()) if active else 0.0
            # 三路候选汇合后按秩融合、术语覆盖和明确标识符匹配排序。
            code_matches = [code for c in active for code in PARAMETER_CODES.get(CONCEPTS[c][0], [])
                            if phrase_present(self.texts[i], code)]
            base = (.20 * 61 / (60 + dense_rank[i]) + .30 * 61 / (60 + local_sparse_ranks[i])
                    + .25 * sparse_normalized[i] + .15 * coverage + (.22 if code_matches else 0))
            if requested:
                base += .30 * exact[i]
            penalty = .35 if self.navigation[i] else 1.0
            score = base * penalty
            scored.append((score, i, coverage, penalty, code_matches))
        scored.sort(key=lambda row: (-row[0], row[1]))
        hits = []
        page_counts = Counter()
        for score, i, coverage, penalty, code_matches in scored:
            page_key = (self.records[i]["source_file"], self.records[i]["pdf_page"])
            if page_counts[page_key] >= 2:
                continue
            page_counts[page_key] += 1
            rank = len(hits) + 1
            hits.append({**self.records[i], "rank": rank, "score": round(score, 6),
                         "vector_score": round(float(dense[i]), 6), "bm25_score": round(float(sparse[i]), 6),
                         "dense_rank": dense_rank[i], "keyword_rank": sparse_rank[i],
                         "matched_anchors": [a for a in requested if anchor_present(self.texts[i], a)],
                         "matched_parameter_codes": code_matches,
                         "concept_coverage": round(coverage, 4), "navigation_penalty": penalty,
                         "source_language": self.records[i]["language"]})
            if len(hits) >= top_k:
                break
        return {"hits": hits, "candidate_count": len(candidates), "candidate_k_per_channel": candidate_k,
                "query_expansion_terms": expansion, "exact_anchors": requested,
                "candidate_chunk_ids": [self.records[i]["chunk_id"] for i in candidates]}


def audit_embeddings(model, vectors, records, questions, query_vectors):
    checks = []
    covered = 0
    max_length = 0
    for kind, texts in (("passage", [r["text"] for r in records]), ("query", [q["question"] for q in questions])):
        prefix = model.tokenizer.encode(kind + ":", add_special_tokens=False)
        old_prefix = model.tokenizer.encode(kind + ": ", add_special_tokens=False)
        capacity = 512 - len(prefix) - model.tokenizer.num_special_tokens_to_add(pair=False)
        equal, old_equal = 0, 0
        for number, text in enumerate(texts):
            ids = model.tokenizer.encode(text, add_special_tokens=False)
            combined = model.tokenizer.encode(kind + ": " + text, add_special_tokens=False)
            equal += combined == prefix + ids
            old_equal += combined == old_prefix + ids
            if kind == "passage":
                assert len(ids) == records[number]["token_count"]
                count, start, covered_until = 0, 0, 0
                while start < len(ids):
                    end = min(start + capacity, len(ids))
                    assert start <= covered_until
                    covered += end - covered_until
                    covered_until = end
                    max_length = max(max_length, end-start+len(prefix)+2)
                    count += 1
                    if end == len(ids):
                        break
                    start = end - 64
                assert covered_until == len(ids)
                assert count == records[number]["embedding_windows"]
        assert equal == len(texts), "修正前缀必须与整句分词完全一致"
        checks.append({"kind": kind, "prefix": kind + ": ", "checked": len(texts),
                       "old_tokenization_identical": old_equal, "corrected_tokenization_identical": equal})
    probes = {"Q01": "zh_p0398_c001", "Q06": "zh_p0175_c002", "Q18": "en_p0404_c003"}
    lookup = {r["chunk_id"]: i for i, r in enumerate(records)}
    ranks = []
    for q, vector in zip(questions, query_vectors):
        if q["id"] in probes:
            target = lookup[probes[q["id"]]]
            scores = vectors @ vector
            order = np.argsort(-scores, kind="stable")
            ranks.append({"id": q["id"], "target_chunk_id": probes[q["id"]],
                          "corrected_dense_rank": int(np.flatnonzero(order == target)[0]) + 1,
                          "target_language": records[target]["language"]})
    return {"prefix_checks": checks, "covered_document_tokens": covered, "max_window_tokens": max_length,
            "truncated_tokens": 0, "default_language_filter": None,
            "conclusion": "原前缀单独分词多出空白token，已修正并重建派生向量；原分窗无截断、默认未限制语言。",
            "focus_questions": ranks}


def corrected_index(model, original_records, original_info):
    target = BASE / "output/vector_index_hybrid"
    config_file = target / "index_info.json"
    if config_file.exists():
        info = json.loads(config_file.read_text(encoding="utf-8"))
        assert info["baseline_vectors_sha256"] == original_info["vectors_sha256"]
        assert info["baseline_metadata_sha256"] == original_info["metadata_sha256"]
        assert info["model_weights_sha256"] == original_info["model_weights_sha256"]
        assert sha256(target / "vectors.npy") == info["vectors_sha256"]
        assert sha256(target / "metadata.jsonl") == info["metadata_sha256"]
        records = [json.loads(line) for line in (target / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
        return np.load(target / "vectors.npy", allow_pickle=False), records, info
    started = time.perf_counter()
    vectors, stats = model.encode([r["text"] for r in original_records], "passage")
    records = [{**r, "token_count": stats["token_lengths"][i], "embedding_windows": stats["window_counts"][i]}
               for i, r in enumerate(original_records)]
    target.mkdir(exist_ok=True)
    np.save(target / "vectors.npy", vectors, allow_pickle=False)
    (target / "metadata.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    info = {**original_info, "baseline_vectors_sha256": original_info["vectors_sha256"],
            "created_at": datetime.now(timezone.utc).isoformat(), "seconds": round(time.perf_counter() - started, 3),
            "baseline_metadata_sha256": original_info["metadata_sha256"],
            "vectors_sha256": sha256(target / "vectors.npy"), "metadata_sha256": sha256(target / "metadata.jsonl"),
            "prefix_tokenization": "encode(kind + ':') + encode(text); asserted equal to encode(kind + ': ' + text)",
            "windowing": {k: v for k, v in stats.items() if k not in {"token_lengths", "window_counts"}},
            "multi_window_chunks": sum(n > 1 for n in stats["window_counts"])}
    write_json(config_file, info)
    return vectors, records, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["query", "evaluate"])
    parser.add_argument("--text")
    parser.add_argument("--language", choices=["zh", "en"])
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    old_vectors, old_records, old_info = load_index()
    model = LocalEmbedder(canonical_prefix=True)
    vectors, records, info = corrected_index(model, old_records, old_info)
    engine = HybridIndex(vectors, records)
    if args.command == "query":
        if not args.text:
            parser.error("query需要--text")
        encoded, _ = model.encode([args.text], "query")
        print(json.dumps(engine.search(args.text, encoded[0], args.top_k, args.language, args.candidate_k), ensure_ascii=False, indent=2))
        return
    questions = json.loads((BASE / "retrieval_questions.json").read_text(encoding="utf-8"))
    baseline_path = BASE / "output/retrieval_eval/retrieval_results.json"
    baseline_hash = sha256(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["questions_sha256"] == sha256(BASE / "retrieval_questions.json")
    assert baseline["index_vectors_sha256"] == old_info["vectors_sha256"]
    encoded, _ = model.encode([q["question"] for q in questions], "query")
    audit = audit_embeddings(model, vectors, records, questions, encoded)
    model.canonical_prefix = False
    old_encoded, _ = model.encode([q["question"] for q in questions], "query")
    model.canonical_prefix = True
    for probe in audit["focus_questions"]:
        qi = next(i for i, q in enumerate(questions) if q["id"] == probe["id"])
        ti = next(i for i, r in enumerate(old_records) if r["chunk_id"] == probe["target_chunk_id"])
        order = np.argsort(-(old_vectors @ old_encoded[qi]), kind="stable")
        probe["original_dense_rank"] = int(np.flatnonzero(order == ti)[0]) + 1
    results = [{**q, **engine.search(q["question"], v, args.top_k, None, args.candidate_k)} for q, v in zip(questions, encoded)]
    for probe in audit["focus_questions"]:
        result = next(q for q in results if q["id"] == probe["id"])
        probe["hybrid_rank"] = next((h["rank"] for h in result["hits"] if h["chunk_id"] == probe["target_chunk_id"]), None)
        probe["in_candidates"] = probe["target_chunk_id"] in result["candidate_chunk_ids"]
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "embedding_audit.json", audit)
    write_json(OUT / "retrieval_results.json", {"top_k": args.top_k, "language_filter": None,
        "baseline_sha256": baseline_hash, "questions_sha256": baseline["questions_sha256"],
        "index_vectors_sha256": info["vectors_sha256"], "algorithm_source_sha256": sha256(__file__),
        "config": {"candidate_k_per_channel": args.candidate_k, "channels": ["dense", "bm25_bilingual", "exact_identifiers"],
                   "per_language_dense_and_bm25_candidates": args.candidate_k // 2, "max_chunks_per_page": 2,
                   "bm25_k1": 1.2, "bm25_b": .75, "expansion_term_weight": .8,
                   "ranking_weights": {"dense_rrf": .20, "language_bm25_rrf": .30,
                       "language_normalized_bm25": .25, "concept_coverage": .15,
                       "parameter_code_bonus": .22, "identifier_coverage": .30}, "rrf_offset": 60,
                   "navigation_multiplier": .35, "reranker": "deterministic rank fusion + concept/identifier coverage", "remote_api": False},
        "results": results})
    assert sha256(baseline_path) == baseline_hash
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    for result in results:
        print(result["id"], [(h["chunk_id"], h["score"]) for h in result["hits"]])


if __name__ == "__main__":
    main()
