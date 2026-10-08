"""只下载公开模型文件；不读取或上传手册、问题、片段。"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(BASE / ".hf-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_DIR = BASE / "models" / "multilingual-e5-small"

if __name__ == "__main__":
    revision = MODEL_REVISION
    snapshot_download(
        repo_id=MODEL_ID, revision=revision, local_dir=MODEL_DIR,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json",
                        "sentencepiece.bpe.model", "README.md"],
        max_workers=3,
    )
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    for filename in required:
        if not (MODEL_DIR / filename).is_file():
            raise RuntimeError(f"缺少模型文件：{filename}")
    (MODEL_DIR / "download_info.json").write_text(
        json.dumps({"model_id": MODEL_ID, "revision": revision}, indent=2), encoding="utf-8")
    print(f"Downloaded {MODEL_ID} revision {revision} to {MODEL_DIR}")
