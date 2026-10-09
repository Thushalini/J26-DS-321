"""
Component 2 - Data collection, part 1: find public DS team repos on GitHub.

Keeps only repos with >= 2 contributors and >= 20 commits, because
Component 2 needs real teamwork to learn from.

Run from the backend folder:
    python -m component2.collect.search_repos
Output:
    component2/data/candidate_repos.json
"""
import json
import os
import re
import time

import requests
from dotenv import load_dotenv

from component2.config import CANDIDATES_FILE, DATA_DIR

load_dotenv(override=True)  # reads backend/.env
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

SEARCH_QUERIES = [
    "data science capstone project language:Jupyter-Notebook",
    "final year data science project language:Jupyter-Notebook",
    "machine learning group project language:Jupyter-Notebook",
    "kaggle competition team language:Jupyter-Notebook",
    "data science team project language:Jupyter-Notebook",
]

MIN_CONTRIBUTORS = 2
MIN_COMMITS = 20
MIN_STARS = 0
RESULTS_PER_QUERY = 30


def api_get(url, params=None, max_retries=3):
    """GET with rate-limit handling. Returns the response, or None if it keeps failing."""
    for attempt in range(max_retries):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset - time.time(), 5) if reset else 60
            print(f"    Rate limited. Waiting {int(wait)}s ...")
            time.sleep(min(wait, 300) + 1)
            continue
        return resp
    return None


def search_repos(query):
    resp = api_get("https://api.github.com/search/repositories",
                   params={"q": query, "sort": "updated", "order": "desc",
                           "per_page": RESULTS_PER_QUERY})
    if resp is None:
        print(f"  Search failed for: {query} (rate limited 3 times)")
        return []
    if resp.status_code != 200:
        print(f"  Search failed for: {query} -> {resp.status_code}: {resp.json().get('message')}")
        return []
    return resp.json().get("items", [])


def get_contributor_count(full_name):
    resp = api_get(f"https://api.github.com/repos/{full_name}/contributors",
                   params={"per_page": 100, "anon": "false"})
    if resp is None or resp.status_code != 200:
        return None
    return len(resp.json())


def get_commit_count_estimate(full_name):
    resp = api_get(f"https://api.github.com/repos/{full_name}/commits", params={"per_page": 1})
    if resp is None or resp.status_code != 200:
        return None
    match = re.search(r'page=(\d+)>; rel="last"', resp.headers.get("Link", ""))
    return int(match.group(1)) if match else len(resp.json())


def main():
    if not GITHUB_TOKEN:
        print("WARNING: no GITHUB_TOKEN in backend/.env - you will hit rate limits quickly.")

    seen = {}
    for q in SEARCH_QUERIES:
        print(f"Searching: {q}")
        for item in search_repos(q):
            name = item["full_name"]
            if name in seen or item.get("fork") or item.get("stargazers_count", 0) < MIN_STARS:
                continue
            seen[name] = {
                "full_name": name,
                "html_url": item["html_url"],
                "clone_url": item["clone_url"],
                "description": item.get("description"),
                "stars": item.get("stargazers_count", 0),
                "updated_at": item.get("updated_at"),
            }
        time.sleep(2)

    print(f"\nFound {len(seen)} unique repos. Checking contributors and commits...")
    qualified = []
    for i, (name, info) in enumerate(seen.items(), start=1):
        n_contrib = get_contributor_count(name)
        n_commits = get_commit_count_estimate(name)
        time.sleep(0.5 if GITHUB_TOKEN else 1.5)
        if n_contrib is None or n_commits is None:
            print(f"  [{i}/{len(seen)}] {name}: check failed, skipping")
            continue
        print(f"  [{i}/{len(seen)}] {name}: {n_contrib} contributors, ~{n_commits} commits")
        if n_contrib >= MIN_CONTRIBUTORS and n_commits >= MIN_COMMITS:
            info["contributors"] = n_contrib
            info["commit_count_estimate"] = n_commits
            qualified.append(info)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(qualified, f, indent=2)
    print(f"\n{len(qualified)} repos passed the filters. Saved to {CANDIDATES_FILE}")


if __name__ == "__main__":
    main()