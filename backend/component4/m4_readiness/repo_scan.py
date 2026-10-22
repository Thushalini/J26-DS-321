"""Reads a few key files from the group's GitHub repository, so readiness and reproducibility judge content and
not just file names: how many tests exist, whether CI runs them, whether the README explains setup, whether
package versions are pinned, whether the Dockerfile installs and runs the project, whether random seeds are set.

Only those files are fetched (README, tests, CI workflows, requirements, Dockerfiles, a few scripts and
notebooks); nothing else is read or stored except the counts below. Results are cached per group and refreshed
after a day or a new sync. When GitHub can't be reached, the last scan (or file names alone) is used.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

from ..common import config

SETTINGS = config.REPO_SCAN
TABLE = "component4_repo_scans"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
GITHUB_REPO = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")

KINDS = {
    "tests": [r"(^|/)tests?/.*\.py$", r"(^|/)test_[^/]*\.py$", r"_test\.py$"],
    "workflows": [r"(^|/)\.github/workflows/[^/]+\.ya?ml$", r"(^|/)\.gitlab-ci\.yml$", r"(^|/)jenkinsfile$",
                  r"(^|/)azure-pipelines\.ya?ml$"],
    "readme": [r"^readme(\.[a-z]+)?$"],
    "requirements": [r"(^|/)requirements[^/]*\.txt$", r"(^|/)environment\.ya?ml$"],
    "docker": [r"(^|/)dockerfile[^/]*$"],
    "scripts": [r"\.py$"],
    "notebooks": [r"\.ipynb$"],
}
PREFERRED_SCRIPTS = r"(^|/)(src|scripts)/|(^|/)(train|run|main|model|predict)[^/]*\.py$"


class ScanError(RuntimeError):
    pass


def _get(url: str, raw: bool = False):
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv(SETTINGS["token_env"])
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.get(url, headers=headers, timeout=SETTINGS["timeout_seconds"])
    except requests.RequestException as error:
        raise ScanError(f"GitHub could not be reached ({type(error).__name__}).") from None
    if response.status_code != 200:
        raise ScanError(f"GitHub answered {response.status_code}.")
    return response.text if raw else response.json()


def github_fetcher(owner: str, repo: str):
    """Returns ({path: size} for every file on the default branch, a function that reads one file, repo details)."""
    meta = _get(f"{SETTINGS['api']}/repos/{owner}/{repo}")
    branch = meta["default_branch"]
    tree = _get(f"{SETTINGS['api']}/repos/{owner}/{repo}/git/trees/{quote(branch)}?recursive=1")
    files = {item["path"]: item.get("size", 0) for item in tree.get("tree", []) if item.get("type") == "blob"}
    return (files, lambda path: _get(f"{SETTINGS['raw']}/{owner}/{repo}/{quote(branch)}/{quote(path)}", raw=True),
            {"created_at": meta.get("created_at")})


def _pick(files: dict[str, int], kind: str) -> list[str]:
    limit = SETTINGS["max_files"][kind]
    small = [p for p, size in sorted(files.items()) if size <= SETTINGS["max_file_bytes"]]
    picked = [p for p in small if any(re.search(pattern, p.lower()) for pattern in KINDS[kind])]
    if kind == "scripts":
        tests = set(_pick(files, "tests"))
        picked = sorted((p for p in picked if p not in tests), key=lambda p: (not re.search(PREFERRED_SCRIPTS, p.lower()), p))
    return picked[:limit]


def _read_all(paths: list[str], read) -> dict[str, str]:
    def one(path):
        try:
            return read(path)
        except ScanError:
            return None

    with ThreadPoolExecutor(max_workers=SETTINGS["parallel_reads"]) as pool:
        return {path: text for path, text in zip(paths, pool.map(one, paths)) if text is not None}


def notebook_code(text: str) -> str:
    try:
        cells = json.loads(text).get("cells", [])
    except (ValueError, AttributeError):
        return ""
    return "\n".join("".join(c.get("source", "")) if isinstance(c.get("source"), list) else c.get("source", "")
                     for c in cells if c.get("cell_type") == "code")


def pinned_share(texts: list[str]) -> float | None:
    packages = pinned = 0
    for text in texts:
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("- "):
                line = line[2:].strip()
            elif line.startswith("-"):
                continue
            if not line or line.endswith(":") or line.startswith(("name:", "channels")):
                continue
            packages += 1
            if "==" in line or "~=" in line or " @ " in line or re.match(r"^[\w.\-\[\]]+=\d", line):
                pinned += 1
    return round(pinned / packages, 3) if packages else None


def evidence(files: dict[str, int], read) -> dict:
    patterns = SETTINGS["patterns"]
    picked = {kind: _pick(files, kind) for kind in KINDS}
    contents = _read_all(sorted({path for paths in picked.values() for path in paths}), read)
    texts = {kind: [contents[path] for path in paths if path in contents] for kind, paths in picked.items()}
    code = texts["scripts"] + [notebook_code(t) for t in texts["notebooks"]]
    readme = texts["readme"][0] if texts["readme"] else ""
    words = len(readme.split())
    has_lock = any(re.search(p, path.lower()) for path in files for p in SETTINGS["lock_files"])
    return {
        "test_count": sum(len(re.findall(patterns["test_function"], t)) for t in texts["tests"]),
        "ci_runs_tests": any(re.search(patterns["ci_test_command"], t) for t in texts["workflows"]),
        "readme_words": words,
        "readme_ok": words >= SETTINGS["readme_min_words"] and bool(re.search(patterns["readme_usage"], readme)),
        "pinned_share": 1.0 if has_lock else pinned_share(texts["requirements"]),
        "docker_ok": any(all(re.search(patterns[k], t) for k in ("docker_from", "docker_install", "docker_run"))
                         for t in texts["docker"]),
        "seeds_set": any(re.search(patterns["seed"], t) for t in code),
        "tracking_in_code": any(re.search(patterns["tracking"], t) for t in code),
        "files_read": len(contents),
    }


def load(conn, team_id: str) -> dict | None:
    row = conn.execute(f"SELECT repo_url, scanned_at, files, evidence FROM {TABLE} WHERE team_id = ?", (team_id,)).fetchone()
    if not row:
        return None
    return {"repo_url": row[0], "scanned_at": row[1], "files": json.loads(row[2]), "evidence": json.loads(row[3])}


def save(conn, team_id: str, scan: dict) -> None:
    conn.execute(f"INSERT INTO {TABLE} (team_id, repo_url, scanned_at, files, evidence) VALUES (?, ?, ?, ?, ?) "
                 "ON CONFLICT (team_id) DO UPDATE SET repo_url = excluded.repo_url, scanned_at = excluded.scanned_at, "
                 "files = excluded.files, evidence = excluded.evidence",
                 (team_id, scan["repo_url"], scan["scanned_at"], json.dumps(scan["files"]), json.dumps(scan["evidence"])))
    conn.commit()


def _fresh(scan: dict, repo_url: str, last_synced: str | None) -> bool:
    scanned = datetime.strptime(scan["scanned_at"], TIME_FORMAT).replace(tzinfo=timezone.utc)
    young = datetime.now(timezone.utc) - scanned < timedelta(hours=SETTINGS["refresh_hours"])
    return scan["repo_url"] == repo_url and young and not (last_synced and last_synced > scan["scanned_at"])


def get(conn, team_id: str, repo_url: str | None, last_synced: str | None = None,
        allow_network: bool = True, fetcher=None) -> dict | None:
    match = GITHUB_REPO.match((repo_url or "").strip())
    if not match:
        return None
    cached = load(conn, team_id)
    cached = cached if cached and cached["repo_url"] == repo_url else None
    if not allow_network or (cached and _fresh(cached, repo_url, last_synced)):
        return cached
    try:
        found = (fetcher or github_fetcher)(*match.groups())
        files, read, info = found[0], found[1], (found[2] if len(found) > 2 else {})
        scan = {"repo_url": repo_url, "scanned_at": datetime.now(timezone.utc).strftime(TIME_FORMAT),
                "files": sorted(files), "evidence": {**evidence(files, read), "repo_created_at": info.get("created_at")}}
    except (ScanError, KeyError, ValueError):
        return cached
    save(conn, team_id, scan)
    return scan
