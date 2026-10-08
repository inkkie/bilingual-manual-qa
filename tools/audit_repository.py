"""Read-only secret audit. Never prints matching values or exception details."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

BASE = Path(__file__).resolve().parents[1]
PATTERNS = [re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
            re.compile(rb"(?i)(?:api[_-]?key|authorization)[\s\"']*[:=][\s\"']*(?:bearer\s+)?[a-z0-9_-]{32,}")]


def main():
    files = (BASE / "docs/proposed_files.txt").read_text(encoding="utf-8").splitlines()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip().encode()
    findings = []
    scanned = []

    def inspect(data, location):
        reasons = []
        if any(p.search(data) for p in PATTERNS):
            reasons.append("credential-shaped value")
        if key and key in data:
            reasons.append("current environment credential exact match")
        if reasons:
            findings.append({"location": location, "rules": reasons})

    for name in files:
        if not name.strip() or name == "docs/repository_audit.json":
            continue  # generated report contains metadata only; avoid self-hash recursion
        path = BASE / name
        if not path.is_file():
            raise RuntimeError("A proposed file is missing")
        data = path.read_bytes()
        inspect(data, name)
        scanned.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
    repo_marker = next((p / ".git" for p in [BASE, *BASE.parents] if (p / ".git").exists()), None)
    history = {"status": "no_local_git_repository", "objects_scanned": 0,
               "note": "No local .git found in workspace or ancestors; no local history exists to inspect. Remote history not inspected."}
    git = shutil.which("git")
    if repo_marker and not git:
        history = {"status": "blocked_git_unavailable", "objects_scanned": 0}
    elif repo_marker:
        def run(*args):
            return subprocess.check_output([git, "-C", str(BASE), *args], stderr=subprocess.DEVNULL)
        history = {"status": "scanned", "objects_scanned": 0, "scope": "all local objects including unreachable objects and reflogs; not unfetched remotes"}
        objects = run("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)").decode().splitlines()
        for line in objects:
            oid, kind = line.split()
            if kind in {"blob", "commit", "tag"}:
                inspect(run("cat-file", kind, oid), "git-object:" + oid)
                history["objects_scanned"] += 1
        inspect(run("reflog", "show", "--all", "--format=%H %gs"), "git-reflog")
        # Check actual tracked/staged/untracked nonignored files, not only planned allowlist.
        for name in run("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0"):
            if name and (BASE / name).is_file():
                inspect((BASE / name).read_bytes(), "git-working:" + name)
        for line in run("ls-files", "--stage", "-z").decode().split("\0"):
            if line:
                fields, name = line.split("\t", 1)
                oid = fields.split()[1]
                inspect(run("cat-file", "blob", oid), "git-index:" + name)
    report = {"findings_count": len(findings), "findings": findings, "proposed_files_scanned": len(scanned),
              "history": history, "files": scanned,
              "limits": "Heuristic and optional environment-value matching; no guarantee of detecting all credentials. No matched secret values are output."}
    target = BASE / "docs/repository_audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inspect(target.read_bytes(), "docs/repository_audit.json")
    print(json.dumps({"findings_count": len(findings), "files_scanned": len(scanned) + 1, "history": history}, ensure_ascii=True))
    if findings or history["status"].startswith("blocked"):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError):
        print("Audit could not complete. No exception detail printed to protect credentials.")
        raise SystemExit(2)
