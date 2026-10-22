import json
import sqlite3
from pathlib import Path

from component4.m1_feature_integration import project_health
from component4.m4_readiness import repo_scan
from component4.m4_readiness.rubric import assess

MIGRATION = Path(__file__).resolve().parents[2] / "buddy" / "migrations" / "0007_component4_repo_scans.sql"
URL = "https://github.com/team/project"

FILES = {
    "README.md": "# Churn\n\n## Setup\npip install -r requirements.txt\n" + "word " * 120,
    "requirements.txt": "pandas==2.2.0\nscikit-learn==1.5.0\nnumpy>=1.26\n-r extra.txt\n# comment\n",
    "tests/test_model.py": "def test_a():\n    pass\n\ndef test_b():\n    pass\n\nasync def test_c():\n    pass\n",
    ".github/workflows/ci.yml": "jobs:\n  test:\n    steps:\n      - run: pytest -q\n",
    "Dockerfile": "FROM python:3.12\nRUN pip install -r requirements.txt\nCMD [\"python\", \"src/train.py\"]\n",
    "src/train.py": "import mlflow\nmodel = RandomForestClassifier(random_state=42)\n",
    "notebooks/eda.ipynb": json.dumps({"cells": [{"cell_type": "code", "source": ["import numpy as np\n", "np.random.seed(1)"]}]}),
}


def fake_fetcher(owner, repo, files=FILES):
    return {path: len(text) for path, text in files.items()}, lambda path: files[path]


def test_reads_what_the_key_files_contain():
    found = repo_scan.evidence(*fake_fetcher("team", "project"))
    assert found["test_count"] == 3 and found["ci_runs_tests"] and found["readme_ok"] and found["docker_ok"]
    assert found["pinned_share"] == 0.667 and found["seeds_set"] and found["tracking_in_code"]


def test_thin_files_do_not_count():
    thin = {"README.md": "# Hi", "tests/test_x.py": "def test_one():\n    pass\n",
            ".github/workflows/lint.yml": "run: flake8", "Dockerfile": "FROM python:3.12\n", "requirements.txt": "pandas\nnumpy\n"}
    found = repo_scan.evidence(*fake_fetcher("t", "p", thin))
    assert found["test_count"] == 1 and not found["ci_runs_tests"] and not found["readme_ok"]
    assert not found["docker_ok"] and found["pinned_share"] == 0.0 and not found["seeds_set"]


def test_a_lock_file_counts_as_fully_pinned():
    files = {**FILES, "poetry.lock": "x"}
    assert repo_scan.evidence(*fake_fetcher("t", "p", files))["pinned_share"] == 1.0


def test_environment_yml_pins_are_understood():
    assert repo_scan.pinned_share(["name: env\ndependencies:\n  - python=3.11\n  - pandas\n  - pip:\n    - shap==0.47\n"]) == 0.667


def test_contents_decide_when_a_scan_exists():
    thin = repo_scan.evidence(*fake_fetcher("t", "p", {"tests/test_x.py": "def test_one():\n    pass\n"}))
    pipeline = next(a for a in assess(["tests/test_x.py"], thin)["areas"] if a["id"] == "pipeline")
    assert "automated tests (1 tests)" in pipeline["missing"]
    assert assess(["tests/test_x.py"])["basis"] == "names" and assess(["tests/test_x.py"], thin)["basis"] == "contents"


def test_reproducibility_uses_seeds_only_when_contents_were_read():
    files = list(FILES)
    with_contents = project_health.reproducibility(files, repo_scan.evidence(*fake_fetcher("t", "p")))
    assert project_health.reproducibility(files) == 0.5
    assert with_contents == 0.4


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.executescript(MIGRATION.read_text(encoding="utf-8"))
    return conn


def test_scans_are_cached_and_reused():
    conn, calls = _conn(), []

    def counting(owner, repo):
        calls.append((owner, repo))
        return fake_fetcher(owner, repo)

    first = repo_scan.get(conn, "t1", URL, fetcher=counting)
    second = repo_scan.get(conn, "t1", URL, fetcher=counting)
    assert calls == [("team", "project")] and first["evidence"] == second["evidence"]


def test_no_network_means_cache_only_and_failures_fall_back():
    conn = _conn()

    def broken(owner, repo):
        raise repo_scan.ScanError("down")

    assert repo_scan.get(conn, "t1", URL, fetcher=broken) is None
    assert repo_scan.get(conn, "t1", URL, allow_network=False) is None
    repo_scan.get(conn, "t1", URL, fetcher=fake_fetcher)
    assert repo_scan.get(conn, "t1", URL, last_synced="2999-01-01T00:00:00Z", fetcher=broken)["evidence"]["test_count"] == 3
    assert repo_scan.get(conn, "t1", "not a github link") is None
