"""本地 E5 向量索引：build / query / evaluate，全程离线、不调用模型API。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(BASE / ".hf-cache")

import numpy as np

INDEX = BASE / "output" / "vector_index"
MODEL = BASE / "models" / "multilingual-e5-small"
CHUNKS = BASE / "output" / "chunks.jsonl"
MANIFEST = BASE / "output" / "chunk_audit" / "retrieval_manifest.jsonl"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def eligible_chunks():
    summary = json.loads((MANIFEST.parent / "summary.json").read_text(encoding="utf-8"))
    if sha256(CHUNKS) != summary["original_sha256"]:
        raise ValueError("原片段已改变，需要先重新审核")
    rows, decisions = read_jsonl(CHUNKS), read_jsonl(MANIFEST)
    by_id = {row["chunk_id"]: row for row in rows}
    if len(by_id) != len(rows) or len({d["chunk_id"] for d in decisions}) != len(decisions):
        raise ValueError("重复chunk_id")
    if set(by_id) != {d["chunk_id"] for d in decisions}:
        raise ValueError("准入清单与原片段不匹配")
    allowed = set()
    for decision in decisions:
        if type(decision["eligible"]) is not bool:
            raise ValueError("eligible 必须为布尔值")
        row = by_id[decision["chunk_id"]]
        if any(row[k] != decision[k] for k in ("source_file", "language", "pdf_page")):
            raise ValueError("准入清单元数据不匹配")
        if decision["eligible"]:
            if decision["reasons"]:
                raise ValueError("通过条目仍带有暂缓原因")
            allowed.add(decision["chunk_id"])
    selected = [row for row in rows if row["chunk_id"] in allowed]
    if len(selected) != 1596 or Counter(r["language"] for r in selected) != {"zh": 276, "en": 1320}:
        raise ValueError("筛选数量不再是本次授权的1596条（zh276/en1320），请先重新核查")
    return selected


class LocalEmbedder:
    def __init__(self, batch_size=16, canonical_prefix=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not (MODEL / "model.safetensors").exists():
            raise FileNotFoundError("请先运行 download_embedding_model.py 下载本地模型")
        self.torch = torch
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        self.tokenizer.model_max_length = 1000000
        self.model = AutoModel.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True).to(self.device).eval()
        self.batch_size = batch_size
        # 旧默认值用于复现基线；新混合检索显式启用标准拼接分词。
        self.canonical_prefix = canonical_prefix
        self.max_tokens = 512
        self.window_overlap = 64
        print(f"Local model loaded: device={self.device}, dimension={self.model.config.hidden_size}", flush=True)

    def encode(self, texts, kind):
        if kind not in {"query", "passage"}:
            raise ValueError(kind)
        torch = self.torch
        prefix = self.tokenizer.encode(kind + (":" if self.canonical_prefix else ": "), add_special_tokens=False)
        capacity = self.max_tokens - len(prefix) - self.tokenizer.num_special_tokens_to_add(pair=False)
        features, owners, weights, window_counts = [], [], [], []
        lengths = []
        for owner, text in enumerate(texts):
            tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if not tokens:
                raise ValueError("不能嵌入空文本")
            lengths.append(len(tokens))
            start, windows, previous_end = 0, 0, 0
            while start < len(tokens):
                end = min(start + capacity, len(tokens))
                ids = self.tokenizer.build_inputs_with_special_tokens(prefix + tokens[start:end])
                if len(ids) > self.max_tokens:
                    raise ValueError("窗口超过模型限制")
                features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
                owners.append(owner)
                weights.append(end - previous_end)
                windows += 1
                if end == len(tokens):
                    break
                previous_end = end
                start = end - self.window_overlap
            window_counts.append(windows)
        vectors = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        weight_sums = np.zeros(len(texts), dtype=np.float32)
        with torch.inference_mode():
            for offset in range(0, len(features), self.batch_size):
                batch = self.tokenizer.pad(features[offset:offset + self.batch_size],
                                           padding=True, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0).sum(1) / mask.sum(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
                for j, vector in enumerate(pooled):
                    i = offset + j
                    vectors[owners[i]] += vector * weights[i]
                    weight_sums[owners[i]] += weights[i]
                if offset % (self.batch_size * 10) == 0 or offset + self.batch_size >= len(features):
                    print(f"Embedded windows {min(offset + self.batch_size, len(features))}/{len(features)}", flush=True)
        if not np.array_equal(weight_sums.astype(int), np.array(lengths)):
            raise RuntimeError("分窗覆盖不完整")
        vectors /= weight_sums[:, None]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms <= 0):
            raise RuntimeError("无效向量")
        vectors /= norms
        return vectors, {"window_counts": window_counts, "token_lengths": lengths,
                         "total_windows": len(features), "max_tokens_per_window": self.max_tokens,
                         "window_overlap": self.window_overlap, "truncated_tokens": 0}


def build(batch_size):
    rows = eligible_chunks()
    original_hash, manifest_hash = sha256(CHUNKS), sha256(MANIFEST)
    model = LocalEmbedder(batch_size)
    start = time.perf_counter()
    vectors, stats = model.encode([r["text"] for r in rows], "passage")
    if vectors.shape != (1596, 384):
        raise RuntimeError(f"意外向量尺寸：{vectors.shape}")
    INDEX.mkdir(parents=True, exist_ok=True)
    np.save(INDEX / "vectors.npy", vectors, allow_pickle=False)
    with (INDEX / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(rows):
            record = {**row, "vector_row": i, "manual_name": Path(row["source_file"]).stem,
                      "token_count": stats["token_lengths"][i],
                      "embedding_windows": stats["window_counts"][i]}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    source_hashes = {name: sha256(BASE / name) for name in {r["source_file"] for r in rows}}
    info = {"created_at": datetime.now(timezone.utc).isoformat(),
            "model": json.loads((MODEL / "download_info.json").read_text(encoding="utf-8")),
            "model_weights_sha256": sha256(MODEL / "model.safetensors"),
            "dimension": 384, "count": len(rows), "language_counts": dict(Counter(r["language"] for r in rows)),
            "metric": "cosine / normalized inner product", "search": "exact flat NumPy",
            "chunks_sha256": original_hash, "manifest_sha256": manifest_hash,
            "source_pdf_sha256": source_hashes,
            "vectors_sha256": sha256(INDEX / "vectors.npy"),
            "metadata_sha256": sha256(INDEX / "metadata.jsonl"),
            "device": model.device, "seconds": round(time.perf_counter() - start, 2),
            "windowing": {k: v for k, v in stats.items() if k not in {"window_counts", "token_lengths"}},
            "multi_window_chunks": sum(c > 1 for c in stats["window_counts"]),
            "pooling": "attention masked mean; token-coverage-weighted mean of normalized windows; L2 normalize",
            "prefixes": {"query": "query: ", "passage": "passage: "},
            "offline": True, "ocr": False}
    if original_hash != sha256(CHUNKS) or manifest_hash != sha256(MANIFEST):
        raise RuntimeError("输入在建库期间发生变化")
    write_json(INDEX / "index_info.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def load_index():
    info = json.loads((INDEX / "index_info.json").read_text(encoding="utf-8"))
    for filename, key in (("vectors.npy", "vectors_sha256"), ("metadata.jsonl", "metadata_sha256")):
        if sha256(INDEX / filename) != info[key]:
            raise ValueError("索引文件已改变：" + filename)
    if sha256(CHUNKS) != info["chunks_sha256"] or sha256(MANIFEST) != info["manifest_sha256"]:
        raise ValueError("索引来源已变化，请重新审核并建库")
    if sha256(MODEL / "model.safetensors") != info["model_weights_sha256"]:
        raise ValueError("当前查询模型与建库模型不同，不能混用向量")
    records = read_jsonl(INDEX / "metadata.jsonl")
    vectors = np.load(INDEX / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if vectors.shape != (len(records), info["dimension"]):
        raise ValueError("向量与元数据不匹配")
    return vectors, records, info


def search(vector, vectors, records, top_k=5, language=None):
    if top_k < 1:
        raise ValueError("top_k 必须为正整数")
    indices = np.array([i for i, row in enumerate(records) if language is None or row["language"] == language])
    if len(indices) == 0:
        return []
    scores = vectors[indices] @ vector
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [{"rank": rank + 1, "score": round(float(scores[j]), 6), **records[int(indices[j])]}
            for rank, j in enumerate(order)]


def evaluate(batch_size, top_k):
    questions = json.loads((BASE / "retrieval_questions.json").read_text(encoding="utf-8"))
    if len(questions) != 20 or Counter(q["language"] for q in questions) != {"zh": 10, "en": 10}:
        raise ValueError("需要中英文各10题")
    if sorted(Counter(q["category"] for q in questions).values()) != [5, 5, 5, 5]:
        raise ValueError("需要四类各5题")
    vectors, records, info = load_index()
    model = LocalEmbedder(batch_size)
    query_vectors, _ = model.encode([q["question"] for q in questions], "query")
    results = [{**question, "hits": search(vector, vectors, records, top_k)}
               for question, vector in zip(questions, query_vectors)]
    output = BASE / "output" / "retrieval_eval"
    output.mkdir(exist_ok=True)
    write_json(output / "retrieval_results.json", {
        "top_k": top_k, "language_filter": None, "query_expansion": False,
        "reranking": False, "index_vectors_sha256": info["vectors_sha256"],
        "index_metadata_sha256": info["metadata_sha256"],
        "questions_sha256": sha256(BASE / "retrieval_questions.json"), "results": results})
    for result in results:
        print(result["id"], result["question"])
        print([(hit["chunk_id"], hit["pdf_page"], hit["score"]) for hit in result["hits"]])
    print("Results saved. 有效依据由逐题内容复核判断，不能由相似度阈值决定。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "query", "evaluate"])
    parser.add_argument("--text")
    parser.add_argument("--language", choices=["zh", "en"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1 or args.top_k < 1:
        parser.error("--batch-size 和 --top-k 必须为正整数")
    if args.command == "build":
        build(args.batch_size)
    elif args.command == "evaluate":
        evaluate(args.batch_size, args.top_k)
    else:
        if not args.text or not args.text.strip():
            parser.error("query 需要 --text 非空问题")
        vectors, records, _ = load_index()
        model = LocalEmbedder(args.batch_size)
        encoded, _ = model.encode([args.text], "query")
        print(json.dumps(search(encoded[0], vectors, records, args.top_k, args.language),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
